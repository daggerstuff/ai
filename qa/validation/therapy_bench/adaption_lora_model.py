"""TherapyBench ``GenerativeModel`` backed by the Adaption run-1 LoRA checkpoint.

Run 1 of the Adaption AutoScientist track (cf1ebae0-1197-4362-92a1-3228c38ce96e)
trained a LoRA adapter (r=8, alpha=8, dropout 0, ``q_proj``/``v_proj`` of the ten
full-attention layers) against the text-only backbone of Qwen3.6-35B-A3B,
bitsandbytes-NF4 quantized at training time. The downloaded checkpoint ships the
text-only base config, the tokenizer, and the authoritative Qwen3-style chat
template (with a thinking block); this module loads that stack and answers bench
prompts through it.

Two load strategies, both probe-verified offline (pilot log §13.8):

- ``remap`` (default): instantiate the text-only CausalLM from the checkpoint's
  own config, stream the public multimodal repo's text weights in with a prefix
  remap (its 692 ``model.language_model.*`` tensors plus the top-level
  ``lm_head.weight`` cover the text model exactly), then attach the adapter with
  native PEFT — keys align perfectly. Needs device memory for BF16 weights
  (~70 GiB: an 80 GiB GPU, or a ~96 GiB-RAM machine on CPU).
- ``mm``: load the full multimodal model through ``from_pretrained`` — required
  for bitsandbytes 4-bit, which only quantizes on that path (a ~24 GiB GPU works
  with ``load_4bit``) — then inject the LoRA into the language-model subtree
  with a name-restricted regex and copy the adapter tensors in under a key
  rewrite. Vision towers are excluded; load-time accounting asserts exactly the
  20 modules the run-1 adapter covers.

Heavy imports (torch/transformers/peft) are deferred to construction so this
module stays importable — and its pure helpers unit-testable — without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

# PEFT saved the adapter rooted at the text-only CausalLM module tree; under the
# multimodal tree the same modules sit one segment deeper.
_ADAPTER_TEXT_PREFIX = "base_model.model.model.layers."
_ADAPTER_MM_PREFIX = "base_model.model.model.language_model.layers."

# Run-1 adapter coverage: q_proj + v_proj of the ten full-attention layers.
_EXPECTED_ADAPTED_MODULES = 20

_LOAD_STRATEGIES = ("remap", "mm")


class ChatTemplateRenderer(Protocol):
    """The minimal tokenizer surface the thinking-marker helpers need."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str: ...


@dataclass(frozen=True)
class ModelLoadConfig:
    """Construction knobs for :class:`AdaptionLoraModel` (see module docstring)."""

    load_strategy: str = "remap"
    load_4bit: bool = False
    device: str = "cuda"
    device_map: str = "auto"
    max_new_tokens: int = 512
    enable_thinking: bool = False
    system_prompt: str | None = None
    seed: int = 42


def build_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Build chat messages for one bench vignette (training rows carry no system turn)."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def derive_thinking_markers(renderer: ChatTemplateRenderer) -> tuple[str, str]:
    """Derive the chat template's thinking open/close markers by rendering it both ways.

    The checkpoint's template appends an opening marker after the generation
    prompt; rendering with ``enable_thinking=False`` additionally emits the
    closing marker. Comparing the two renders extracts both markers without
    hardcoding template bytes. Returns ``("", "")`` for a template with no
    thinking block.
    """
    messages = [{"role": "user", "content": "probe"}]
    thinking = renderer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    no_thinking = renderer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    open_marker = thinking.rsplit("assistant\n", 1)[-1]
    if open_marker.endswith("\n"):
        open_marker = open_marker[:-1]
    close_marker = ""
    if no_thinking.startswith(thinking) and len(no_thinking) > len(thinking):
        close_marker = no_thinking[len(thinking):].strip("\n")
    return open_marker, close_marker


def strip_thinking(text: str, open_marker: str, close_marker: str) -> str:
    """Strip a thinking block the model emitted despite the no-thinking prefill.

    Takes the span after the last closing marker (the final answer); an
    unterminated thinking block yields ``""`` so the judge scores it honestly.
    """
    if not close_marker or close_marker not in text:
        if open_marker and text.lstrip().startswith(open_marker):
            return ""
        return text.strip()
    return text.rsplit(close_marker, 1)[-1].strip()


class AdaptionLoraModel:
    """TherapyBench ``GenerativeModel`` over the run-1 checkpoint stack."""

    def __init__(
        self,
        name: str,
        *,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        adapter_path: str | None = None,
        tokenizer_path: str | None = None,
        config: ModelLoadConfig | None = None,
    ) -> None:
        cfg = config or ModelLoadConfig()
        if cfg.load_strategy not in _LOAD_STRATEGIES:
            msg = f"load_strategy must be one of {_LOAD_STRATEGIES}, got {cfg.load_strategy!r}"
            raise ValueError(msg)
        tokenizer_source = tokenizer_path or adapter_path
        if not tokenizer_source:
            msg = "adapter_path or tokenizer_path is required: the checkpoint ships the authoritative tokenizer"
            raise ValueError(msg)
        self.name = name
        self._cfg = cfg
        self._load_tokenizer(str(tokenizer_source))
        if cfg.load_strategy == "remap":
            self._model = self._load_remap(base_model_id, adapter_path, str(tokenizer_source), cfg)
        else:
            self._model = self._load_mm(base_model_id, adapter_path, cfg)
        import torch

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        self._open_marker, self._close_marker = derive_thinking_markers(self._tokenizer)
        self._model.eval()

    # ── construction helpers ───────────────────────────────────────────────

    def _load_tokenizer(self, source: str) -> None:
        """Load the checkpoint tokenizer and make sure its chat template is set."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(source)
        if not tok.chat_template:
            template_path = Path(source) / "chat_template.jinja"
            if template_path.is_file():
                tok.chat_template = template_path.read_text(encoding="utf-8")
        if not tok.chat_template:
            msg = f"no chat template found for tokenizer at {source}"
            raise ValueError(msg)
        self._tokenizer = tok

    def _load_remap(
        self,
        base_model_id: str,
        adapter_path: str | None,
        config_source: str,
        cfg: ModelLoadConfig,
    ) -> Any:
        """Text-only load: checkpoint config + prefix-remapped public weights + native PEFT."""
        import json

        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM

        model_config = AutoConfig.from_pretrained(config_source)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(cfg.device):
                model = AutoModelForCausalLM.from_config(model_config)
        finally:
            torch.set_default_dtype(previous_dtype)

        index_path = hf_hub_download(base_model_id, "model.safetensors.index.json")
        weight_map: dict[str, str] = json.loads(
            Path(index_path).read_text(encoding="utf-8")
        )["weight_map"]
        loaded: set[str] = set()
        for shard in sorted(set(weight_map.values())):
            shard_state = load_file(hf_hub_download(base_model_id, shard))
            remapped: dict[str, Any] = {}
            for key, tensor in shard_state.items():
                if key.startswith("model.language_model."):
                    remapped[key.replace("model.language_model.", "model.", 1)] = tensor
                elif key == "lm_head.weight":
                    remapped[key] = tensor
                # else: vision / MTP weights — not part of the text backbone.
            if not remapped:
                continue
            result = model.load_state_dict(remapped, strict=False)
            if result.unexpected_keys:
                msg = (
                    f"unexpected text-backbone keys in shard {shard}: "
                    f"{result.unexpected_keys[:3]}"
                )
                raise RuntimeError(msg)
            loaded.update(remapped)
        missing = set(dict(model.state_dict()).keys()) - loaded
        if missing:
            msg = f"remap load incomplete: {len(missing)} unfilled keys, e.g. {sorted(missing)[:3]}"
            raise RuntimeError(msg)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        return model

    def _load_mm(self, base_model_id: str, adapter_path: str | None, cfg: ModelLoadConfig) -> Any:
        """Multimodal load via ``from_pretrained`` (bnb-quantizable) + subtree-restricted PEFT."""
        import torch
        from transformers import BitsAndBytesConfig

        kwargs: dict[str, Any] = {"device_map": cfg.device_map}
        if cfg.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        model = self._from_pretrained_multimodal(base_model_id, kwargs)
        if adapter_path:
            model = self._inject_adapter_mm(model, adapter_path)
        return model

    @staticmethod
    def _from_pretrained_multimodal(base_model_id: str, kwargs: dict[str, Any]) -> Any:
        """Load the multimodal repo, falling back to its concrete architecture class."""
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(base_model_id, **kwargs)
        except (ImportError, ValueError):
            import importlib

            from transformers import AutoConfig

            architectures = getattr(
                AutoConfig.from_pretrained(base_model_id), "architectures", None
            )
            if not architectures:
                raise
            model_cls = getattr(importlib.import_module("transformers"), architectures[0])
            return model_cls.from_pretrained(base_model_id, **kwargs)

    @staticmethod
    def _inject_adapter_mm(model: Any, adapter_path: str) -> Any:
        """Inject the run-1 adapter into the language-model subtree with verified accounting.

        The saved config targets ``q_proj``/``v_proj`` by suffix — on the
        multimodal tree that would also hit vision-tower projections the adapter
        was never trained for. Restrict the match to the language-model subtree,
        copy the tensors in under a key rewrite, and assert the exact module
        coverage the run-1 adapter has (10 q_proj + 10 v_proj).
        """
        from peft import LoraConfig, get_peft_model
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        saved = LoraConfig.from_pretrained(adapter_path)
        saved.target_modules = r".*language_model.*\.(?:q_proj|v_proj)$"
        model = get_peft_model(model, saved)

        adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
        if len(adapted) != _EXPECTED_ADAPTED_MODULES:
            msg = (
                f"expected {_EXPECTED_ADAPTED_MODULES} adapted modules (q_proj/v_proj of the "
                f"ten full-attention layers), injected {len(adapted)}"
            )
            raise RuntimeError(msg)
        escaped = [n for n in adapted if "language_model" not in n]
        if escaped:
            msg = f"adapter injection escaped the language-model subtree: {escaped[:3]}"
            raise RuntimeError(msg)

        adapter_state = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
        remapped: dict[str, Any] = {}
        for key, tensor in adapter_state.items():
            if not key.startswith(_ADAPTER_TEXT_PREFIX):
                msg = f"adapter key outside the expected prefix: {key}"
                raise ValueError(msg)
            remapped[_ADAPTER_MM_PREFIX + key[len(_ADAPTER_TEXT_PREFIX):]] = tensor
        result = set_peft_model_state_dict(model, remapped)
        lost = [k for k in result.missing_keys if "lora" in k]
        stray = [k for k in result.unexpected_keys if "lora" in k]
        if lost or stray:
            msg = f"adapter tensor accounting failed — missing: {lost[:3]}, unexpected: {stray[:3]}"
            raise RuntimeError(msg)
        return model

    # ── GenerativeModel protocol ───────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Answer one bench prompt through the checkpoint's chat template (greedy)."""
        import torch

        messages = build_messages(prompt, self._cfg.system_prompt)
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._cfg.enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return strip_thinking(raw, self._open_marker, self._close_marker)
