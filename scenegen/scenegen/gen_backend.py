# from __future__ import annotations
# # =============================================================================
# # gen_backend.py  --  probe_backend WITHOUT activation capture.
# # -----------------------------------------------------------------------------
# # Identical generation path to probe_backend.VLMProbe.complete (same chat
# # template with enable_thinking=False, same per-call torch seeding
# # `DET_SEED + call_counter`, same do_sample/temperature/top_p kwargs), but it
# # does NOT run the second _forward_hidden pass and stores NO activations / RECORDS.
# #
# # Because probe_backend re-seeds torch immediately BEFORE model.generate on every
# # call, and the hidden-state pass runs only AFTER the tokens are produced, removing
# # that pass cannot change the sampled tokens. Scenes generated here are the same
# # scenes you were probing (given the same DET_SEED, spec order, and decode mode).
# #
# # Drop-in for Heuristic_wo_dspy.llm_complete via gen_complete().
# # Set GEN_MOCK=1 to skip loading the model (returns a canned object list) for
# # wiring / dry tests.
# # =============================================================================

# import os
# import re
# from typing import Optional

# _MODEL: "GenModel | None" = None
# DET_SEED = None            # if set, seed torch by DET_SEED + call_index (deterministic)

# DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"
# DEFAULT_TEMPERATURE = 0.5
# DEFAULT_MAX_TOKENS = 8192
# JSON_SYSTEM = (
#     "You are a strict JSON generator. Output ONLY a single valid JSON value "
#     "and nothing else. Do NOT include any reasoning, thinking, explanation, "
#     "preamble, markdown, or code fences. Your entire response must be parseable "
#     "by json.loads()."
# )


# def set_det_seed(s):
#     """Enable deterministic per-call seeding (None disables -> greedy or seed-from-prompt)."""
#     global DET_SEED
#     DET_SEED = s


# def reset_call_counter():
#     """Reset the per-call counter so every seed's run replays the same seed sequence."""
#     if _MODEL is not None:
#         _MODEL._call_counter = 0


# def _parse_seed_from_prompt(prompt: str) -> Optional[int]:
#     m = re.search(r"Random seed:\s*(\d+)", prompt)
#     return int(m.group(1)) if m else None


# class GenModel:
#     def __init__(self, model_path: str, dtype: str = "bfloat16",
#                  device_map: str = "auto", max_seq_len: int = 8192):
#         self.model_path = model_path
#         self.max_seq_len = max_seq_len
#         self.mock = os.environ.get("GEN_MOCK", "0") == "1"
#         self.model = None
#         self.tok = None
#         self._call_counter = 0
#         if not self.mock:
#             self._load(dtype, device_map)
#         else:
#             print("[gen_backend] GEN_MOCK=1 -> no model loaded (canned outputs)")

#     def _load(self, dtype: str, device_map: str):
#         import torch
#         from transformers import AutoModelForCausalLM, AutoTokenizer
#         td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
#               "float32": torch.float32}[dtype]
#         self.tok = AutoTokenizer.from_pretrained(
#             self.model_path, trust_remote_code=True, use_fast=True)
#         if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
#             self.tok.pad_token = self.tok.eos_token
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path, trust_remote_code=True, torch_dtype=td,
#             device_map=device_map)
#         self.model.config.use_cache = True
#         self.model.eval()
#         nL = self.model.config.num_hidden_layers
#         print(f"[gen_backend] loaded {self.model_path}: num_hidden_layers={nL}")

#     def complete(self, prompt: str, images=None, system: Optional[str] = None,
#                  temperature: float = DEFAULT_TEMPERATURE,
#                  max_tokens: int = DEFAULT_MAX_TOKENS,
#                  force_json: bool = True) -> str:
#         if images:
#             raise NotImplementedError(
#                 "gen_backend is text-only; run the pipeline with orientation=False "
#                 "(the orientation refiner is the only image call).")

#         seed = _parse_seed_from_prompt(prompt)
#         self._call_counter += 1

#         if self.mock:
#             return ('[{"label": "sofa", "object_type": "bulk", "position": [0.0, 1.0], '
#                     '"bbox": [-1.0, 0.0, 2.0, 1.0], "rotation_angle": 0.0}]')

#         import torch
#         sys_msg = system or (JSON_SYSTEM if force_json else None)
#         messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
#                    [{"role": "user", "content": prompt}]
#         try:
#             rendered = self.tok.apply_chat_template(
#                 messages, add_generation_prompt=True, tokenize=False,
#                 enable_thinking=False)
#         except TypeError:
#             rendered = self.tok.apply_chat_template(
#                 messages, add_generation_prompt=True, tokenize=False)

#         enc = self.tok(rendered, return_tensors="pt", truncation=True,
#                        max_length=self.max_seq_len)
#         dev = self.model.get_input_embeddings().weight.device
#         input_ids = enc["input_ids"].to(dev)
#         attn = enc.get("attention_mask")
#         attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
#         prompt_len = int(input_ids.shape[1])

#         # Same per-call seeding as probe_backend: reproducible AND identical across
#         # baseline/adversarial (or across adjectives) for a given DET_SEED.
#         if DET_SEED is not None:
#             torch.manual_seed(DET_SEED + self._call_counter)
#         elif seed is not None:
#             torch.manual_seed(seed)

#         import time
#         gen_kwargs = dict(max_new_tokens=max_tokens,
#                           pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
#         if temperature and temperature > 0:
#             gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
#         else:
#             gen_kwargs.update(do_sample=False)
#         t0 = time.time()
#         with torch.inference_mode():
#             full = self.model.generate(input_ids=input_ids, attention_mask=attn,
#                                        **gen_kwargs)
#         new_ids = full[0][prompt_len:]
#         output_text = self.tok.decode(new_ids, skip_special_tokens=True)

#         dt = time.time() - t0
#         n_new = int(new_ids.shape[0])
#         if self._call_counter <= 5 or n_new >= max_tokens:
#             tps = n_new / dt if dt > 0 else 0.0
#             note = ""
#             if n_new >= max_tokens:
#                 note = "  <-- hit max_new_tokens (truncated / no EOS; raise --max-tokens)"
#             if "<think>" in output_text or "</think>" in output_text:
#                 note += "  <-- OUTPUT CONTAINS <think>: thinking not suppressed"
#             print(f"[gen_backend] call {self._call_counter}: {n_new} tok in {dt:.1f}s "
#                   f"({tps:.1f} tok/s){note}")
#         return output_text


# def init_gen(model_path: str = DEFAULT_MODEL_PATH, dtype: str = "bfloat16",
#              device_map: str = "auto", max_seq_len: int = 8192) -> GenModel:
#     global _MODEL
#     _MODEL = GenModel(model_path, dtype, device_map, max_seq_len)
#     return _MODEL


# def gen_complete(prompt: str, images=None, system: Optional[str] = None,
#                  temperature: float = DEFAULT_TEMPERATURE,
#                  max_tokens: int = DEFAULT_MAX_TOKENS,
#                  force_json: bool = True) -> str:
#     """Exact drop-in for Heuristic_wo_dspy.llm_complete (no probing)."""
#     if _MODEL is None:
#         raise RuntimeError("call gen_backend.init_gen(...) first")
#     return _MODEL.complete(prompt, images, system, temperature, max_tokens, force_json)

from __future__ import annotations
# =============================================================================
# gen_backend.py  --  probe_backend WITHOUT activation capture.
# -----------------------------------------------------------------------------
# Identical generation path to probe_backend.VLMProbe.complete (same chat
# template with enable_thinking=False, same per-call torch seeding
# `DET_SEED + call_counter`, same do_sample/temperature/top_p kwargs), but it
# does NOT run the second _forward_hidden pass and stores NO activations / RECORDS.
#
# Because probe_backend re-seeds torch immediately BEFORE model.generate on every
# call, and the hidden-state pass runs only AFTER the tokens are produced, removing
# that pass cannot change the sampled tokens. Scenes generated here are the same
# scenes you were probing (given the same DET_SEED, spec order, and decode mode).
#
# Drop-in for Heuristic_wo_dspy.llm_complete via gen_complete().
# Set GEN_MOCK=1 to skip loading the model (returns a canned object list) for
# wiring / dry tests.
# =============================================================================

import os
import re
from typing import Optional

_MODEL: "GenModel | None" = None
DET_SEED = None            # if set, seed torch by DET_SEED + call_index (deterministic)

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 8192
JSON_SYSTEM = (
    "You are a strict JSON generator. Output ONLY a single valid JSON value "
    "and nothing else. Do NOT include any reasoning, thinking, explanation, "
    "preamble, markdown, or code fences. Your entire response must be parseable "
    "by json.loads()."
)


def set_det_seed(s):
    """Enable deterministic per-call seeding (None disables -> greedy or seed-from-prompt)."""
    global DET_SEED
    DET_SEED = s


def reset_call_counter():
    """Reset the per-call counter so every seed's run replays the same seed sequence."""
    if _MODEL is not None:
        _MODEL._call_counter = 0


def _parse_seed_from_prompt(prompt: str) -> Optional[int]:
    m = re.search(r"Random seed:\s*(\d+)", prompt)
    return int(m.group(1)) if m else None


class GenModel:
    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 device_map: str = "auto", max_seq_len: int = 8192):
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.mock = os.environ.get("GEN_MOCK", "0") == "1"
        self.model = None
        self.tok = None
        self._call_counter = 0
        if not self.mock:
            self._load(dtype, device_map)
        else:
            print("[gen_backend] GEN_MOCK=1 -> no model loaded (canned outputs)")

    def _load(self, dtype: str, device_map: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=td,
            device_map=device_map)
        self.model.config.use_cache = True
        self.model.eval()
        nL = self.model.config.num_hidden_layers
        print(f"[gen_backend] loaded {self.model_path}: num_hidden_layers={nL}")

    def complete(self, prompt: str, images=None, system: Optional[str] = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 force_json: bool = True) -> str:
        if images:
            raise NotImplementedError(
                "gen_backend is text-only; run the pipeline with orientation=False "
                "(the orientation refiner is the only image call).")

        seed = _parse_seed_from_prompt(prompt)
        self._call_counter += 1

        if self.mock:
            return ('[{"label": "sofa", "object_type": "bulk", "position": [0.0, 1.0], '
                    '"bbox": [-1.0, 0.0, 2.0, 1.0], "rotation_angle": 0.0}]')

        import torch
        sys_msg = system or (JSON_SYSTEM if force_json else None)
        messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
                   [{"role": "user", "content": prompt}]
        try:
            rendered = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            rendered = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

        enc = self.tok(rendered, return_tensors="pt", truncation=True,
                       max_length=self.max_seq_len)
        dev = self.model.get_input_embeddings().weight.device
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
        prompt_len = int(input_ids.shape[1])

        # Same per-call seeding as probe_backend: reproducible AND identical across
        # baseline/adversarial (or across adjectives) for a given DET_SEED.
        if DET_SEED is not None:
            torch.manual_seed(DET_SEED + self._call_counter)
        elif seed is not None:
            torch.manual_seed(seed)

        import time
        gen_kwargs = dict(max_new_tokens=max_tokens,
                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        t0 = time.time()
        with torch.inference_mode():
            full = self.model.generate(input_ids=input_ids, attention_mask=attn,
                                       **gen_kwargs)
        new_ids = full[0][prompt_len:]
        output_text = self.tok.decode(new_ids, skip_special_tokens=True)

        dt = time.time() - t0
        n_new = int(new_ids.shape[0])
        if self._call_counter <= 5 or n_new >= max_tokens:
            tps = n_new / dt if dt > 0 else 0.0
            note = ""
            if n_new >= max_tokens:
                note = "  <-- hit max_new_tokens (truncated / no EOS; raise --max-tokens)"
            if "<think>" in output_text or "</think>" in output_text:
                note += "  <-- OUTPUT CONTAINS <think>: thinking not suppressed"
            print(f"[gen_backend] call {self._call_counter}: {n_new} tok in {dt:.1f}s "
                  f"({tps:.1f} tok/s){note}")
        return output_text


def init_gen(model_path: str = DEFAULT_MODEL_PATH, dtype: str = "bfloat16",
             device_map: str = "auto", max_seq_len: int = 8192) -> GenModel:
    global _MODEL
    _MODEL = GenModel(model_path, dtype, device_map, max_seq_len)
    return _MODEL


def gen_complete(prompt: str, images=None, system: Optional[str] = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 force_json: bool = True) -> str:
    """Exact drop-in for Heuristic_wo_dspy.llm_complete (no probing)."""
    if _MODEL is None:
        raise RuntimeError("call gen_backend.init_gen(...) first")
    return _MODEL.complete(prompt, images, system, temperature, max_tokens, force_json)