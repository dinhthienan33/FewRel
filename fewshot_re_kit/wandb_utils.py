import os
import json
import sys
import site
import importlib
import importlib.util

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _load_real_wandb_package():
    """Prefer site-packages wandb when local ./wandb folder shadows the module."""
    global wandb
    if wandb is not None and hasattr(wandb, "login") and hasattr(wandb, "init"):
        return

    try:
        if "wandb" in sys.modules:
            del sys.modules["wandb"]
    except Exception:
        pass

    search_paths = []
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        search_paths.append(user_site)

    for p in search_paths:
        if p and os.path.isdir(os.path.join(p, "wandb")):
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                wandb = importlib.import_module("wandb")
                if hasattr(wandb, "login") and hasattr(wandb, "init"):
                    return
            except Exception:
                continue

    # Last-resort loader: import site-packages/wandb directly by file path.
    for p in search_paths:
        pkg_dir = os.path.join(p, "wandb") if p else ""
        init_py = os.path.join(pkg_dir, "__init__.py")
        if os.path.isfile(init_py):
            try:
                spec = importlib.util.spec_from_file_location(
                    "wandb",
                    init_py,
                    submodule_search_locations=[pkg_dir],
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["wandb"] = module
                spec.loader.exec_module(module)
                wandb = module
                if hasattr(wandb, "login") and hasattr(wandb, "init"):
                    return
            except Exception:
                continue


_load_real_wandb_package()


def _wandb_run():
    return getattr(wandb, "run", None)


def _wandb_has_api():
    return hasattr(wandb, "init") and hasattr(wandb, "login")


def _is_permission_error(exc):
    msg = str(exc).lower()
    return (
        ("403" in msg)
        or ("forbidden" in msg)
        or ("permission" in msg)
        or ("unauthorized" in msg)
        or ("invalid api key" in msg)
        or ("api key" in msg and "invalid" in msg)
    )


def _resolve_api_key(explicit_key, cfg_key):
    arg_key = str(explicit_key).strip() if explicit_key is not None else ""
    env_key = str(os.environ.get("WANDB_API_KEY", "")).strip()
    config_key = str(cfg_key).strip()
    if arg_key:
        return arg_key, "argument"
    if env_key:
        return env_key, "env"
    if config_key:
        return config_key, "config"
    return "", "none"


def _load_config_wandb_defaults():
    try:
        from config import Config  # local project config
        project = str(getattr(Config, "wandb_project", "")).strip()
        api_key = str(getattr(Config, "wandb_key", "")).strip()
        return project, api_key
    except Exception:
        return "", ""

def init_wandb_run(
    name,
    config,
    project="FewRel",
    entity=None,
    api_key=None,
    mode="online",
    allow_offline_fallback=False,
):
    try:
        if not _wandb_has_api():
            print("⚠️ wandb package API not available (likely shadowed by local folder). Logging disabled.")
            return False

        cfg_project, cfg_key = _load_config_wandb_defaults()
        if (not project) or str(project).strip() in {"", "DocRED", "FewRel"}:
            project = cfg_project or project or "FewRel"
        key, key_source = _resolve_api_key(api_key, cfg_key)

        requested_mode = str(mode or "online").strip().lower()
        if requested_mode not in {"online", "offline", "auto"}:
            requested_mode = "online"

        if requested_mode == "auto":
            effective_mode = "online"
        elif requested_mode == "offline":
            effective_mode = "offline"
        else:
            # Default behavior: prioritize online logging unless explicitly disabled.
            effective_mode = "online"
            if os.environ.get("WANDB_MODE", "").strip().lower() == "offline":
                print("[W&B] Overriding WANDB_MODE=offline -> online.")

        if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
            print("[W&B] Clearing WANDB_DISABLED so cloud logging can start.")
            os.environ.pop("WANDB_DISABLED", None)

        os.environ["WANDB_MODE"] = effective_mode
        # fewshot_re_kit/ -> FewRel/ root for local wandb runs
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_dir = os.environ.get("WANDB_DIR", os.path.join(workspace_root, "wandb"))
        os.makedirs(run_dir, exist_ok=True)

        if key:
            os.environ["WANDB_API_KEY"] = key
            key_suffix = key[-6:] if len(key) >= 6 else key
            print(f"[W&B] Using project='{project}' with API key source={key_source} suffix='***{key_suffix}'")
            try:
                wandb.login(key=key, relogin=True)
            except Exception as login_exc:
                # If config/env key is stale, allow retry via existing wandb login session.
                print(f"[W&B] Login with {key_source} key failed: {login_exc}")
                if key_source == "argument":
                    raise
                os.environ.pop("WANDB_API_KEY", None)
                key = ""
                key_source = "none"
        else:
            print("[W&B] No API key provided via args/env/config; using existing wandb login session if available.")

        init_kwargs = dict(project=project, name=name, config=config, mode=effective_mode, dir=run_dir)
        if entity and str(entity).strip():
            init_kwargs["entity"] = str(entity).strip()

        init_ok = False
        init_exc = None
        try:
            wandb.init(**init_kwargs)
            init_ok = True
        except Exception as exc:
            init_exc = exc

        # Common failure mode: stale key in config.py overrides a valid local wandb session.
        if (not init_ok) and effective_mode == "online" and _is_permission_error(init_exc) and key_source == "config":
            print("[W&B] Online init failed with config key. Retrying using existing wandb login session.")
            os.environ.pop("WANDB_API_KEY", None)
            try:
                wandb.init(**init_kwargs)
                init_ok = True
            except Exception as retry_exc:
                init_exc = retry_exc

        if not init_ok:
            if effective_mode != "online":
                raise init_exc

            if _is_permission_error(init_exc):
                print(f"[W&B] Online init permission error with project='{project}': {init_exc}")
                if not allow_offline_fallback:
                    raise init_exc
                print("⚠️ Falling back to offline mode because allow_offline_fallback=True.")
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(project=project, name=name, config=config, mode="offline", dir=run_dir)
            else:
                if not allow_offline_fallback:
                    raise init_exc
                print(f"⚠️ W&B online init failed: {init_exc}. Retrying offline mode.")
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(project=project, name=name, config=config, mode="offline", dir=run_dir)

        run = _wandb_run()
        if run is not None:
            run_mode = getattr(getattr(run, "settings", None), "mode", os.environ.get("WANDB_MODE", "unknown"))
            run_url = getattr(run, "url", None)
            print(f"[W&B] Initialized mode={run_mode} project={project} run={getattr(run, 'name', name)}")
            if run_url:
                print(f"[W&B] Run URL: {run_url}")
        return run is not None
    except Exception as e:
        print(f"⚠️ Failed to initialize wandb: {e}. Logging disabled.")
        return False

def log_metrics(metrics):
    if _wandb_run() is not None and hasattr(wandb, "log"):
        wandb.log(metrics)

def log_system_metrics():
    # wandb handles system metrics automatically if initialized
    pass

def save_model_artifact(path, name, artifact_type="model"):
    run = _wandb_run()
    if run is not None and hasattr(wandb, "Artifact"):
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if os.path.isdir(path):
            root = os.path.abspath(path)
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    relpath = os.path.relpath(fpath, root).replace("\\", "/")
                    try:
                        artifact.add_file(fpath, name=relpath)
                    except (FileNotFoundError, OSError, ValueError):
                        # Runtime logs can rotate while walking the workspace.
                        continue
        else:
            artifact.add_file(path)
        run.log_artifact(artifact)
        return True
    return False

def save_evaluation_artifact(results, path, artifact_name="evaluation"):
    run = _wandb_run()
    if run is not None and hasattr(wandb, "Artifact"):
        with open(path, 'w') as f:
            json.dump(results, f)
        artifact = wandb.Artifact(name=artifact_name, type="evaluation")
        artifact.add_file(path)
        run.log_artifact(artifact)
        return True
    return False

def finish_run():
    if _wandb_run() is not None and hasattr(wandb, "finish"):
        wandb.finish()
