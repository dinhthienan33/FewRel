import os
from pathlib import Path


def load_dotenv_file(path=None):
    """Load FewRel/.env into os.environ without overriding existing vars.

    Tries python-dotenv if installed; otherwise parses KEY=VALUE lines manually.
    Returns the path loaded, or None.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / ".env"
    else:
        path = Path(path)
    if not path.is_file():
        return None

    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return str(path)
    except Exception:
        pass

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if not key:
            continue
        if key not in os.environ or not str(os.environ.get(key, "")).strip():
            os.environ[key] = val
    return str(path)


def first_nonempty(*values):
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""
