import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

ENTITY_MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

class LLMSentenceEncoder(nn.Module):
    """Causal-LLM sentence encoder for the graph + MoE FewRel model.

    Unlike the CNN/BERT encoders (which return a single pooled vector),
    this encoder returns the full sequence of token hidden states so the
    downstream graph expert can build a token-level graph. Head/tail spans
    are wrapped with ``[E1]/[/E1]`` and ``[E2]/[/E2]`` markers; the token
    index of ``[E1]`` and ``[E2]`` is returned as ``pos1`` / ``pos2``.
    """

    def __init__(
        self,
        pretrain_path,
        max_length,
        load_4bit=False,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        freeze_llm=True,
    ):
        nn.Module.__init__(self)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path, add_prefix_space=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ENTITY_MARKERS})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model(pretrain_path, load_4bit)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.model.config.hidden_size

        self.pad_id = self.tokenizer.pad_token_id or 0
        self.e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        self._maybe_apply_lora(load_4bit, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm)

    def _load_model(self, pretrain_path, load_4bit):
        if load_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                return AutoModelForCausalLM.from_pretrained(
                    pretrain_path,
                    quantization_config=quant_cfg,
                    device_map={"": 0},
                )
            except Exception as e:  # pragma: no cover - depends on env
                print("[WARN] 4-bit load failed ({}); falling back to fp32/fp16.".format(e))

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return AutoModelForCausalLM.from_pretrained(pretrain_path, torch_dtype=dtype)

    def _maybe_apply_lora(self, load_4bit, use_lora, lora_r, lora_alpha, lora_dropout, freeze_llm):
        self.lora_enabled = False
        if not use_lora:
            if freeze_llm:
                for p in self.model.parameters():
                    p.requires_grad = False
            return
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

            if load_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            target = self._lora_targets()
            if not target:
                print("[WARN] No known LoRA target modules; freezing LLM instead.")
                if freeze_llm:
                    for p in self.model.parameters():
                        p.requires_grad = False
                return
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target,
                ),
            )
            self.lora_enabled = True
        except Exception as e:  # pragma: no cover - depends on env
            print("[WARN] peft/LoRA unavailable ({}); freezing LLM.".format(e))
            if freeze_llm:
                for p in self.model.parameters():
                    p.requires_grad = False

    def _lora_targets(self):
        names = [n for n, _ in self.model.named_modules()]
        if any(n.endswith("q_proj") for n in names) and any(n.endswith("v_proj") for n in names):
            return ["q_proj", "v_proj"]
        if any(n.endswith("c_attn") for n in names):
            return ["c_attn"]
        return []

    def forward(self, inputs):
        outputs = self.model(
            input_ids=inputs["word"],
            attention_mask=inputs["mask"],
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        return hidden.float()

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        h_start, h_end = pos_head[0], pos_head[-1]
        t_start, t_end = pos_tail[0], pos_tail[-1]

        marked = []
        for i, w in enumerate(raw_tokens):
            if i == h_start:
                marked.append("[E1]")
            if i == t_start:
                marked.append("[E2]")
            marked.append(w)
            if i == h_end:
                marked.append("[/E1]")
            if i == t_end:
                marked.append("[/E2]")

        enc = self.tokenizer(marked, is_split_into_words=True, add_special_tokens=False)
        ids = enc["input_ids"][: self.max_length]

        e1_pos = ids.index(self.e1_id) if self.e1_id in ids else 0
        e2_pos = ids.index(self.e2_id) if self.e2_id in ids else 0

        mask = [1] * len(ids)
        while len(ids) < self.max_length:
            ids.append(self.pad_id)
            mask.append(0)

        e1_pos = min(e1_pos, self.max_length - 1)
        e2_pos = min(e2_pos, self.max_length - 1)
        return ids, e1_pos, e2_pos, mask
