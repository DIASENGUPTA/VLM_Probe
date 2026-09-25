# # #!/usr/bin/env python3
# # # =============================================================================
# # # probe_backend.py
# # # -----------------------------------------------------------------------------
# # # A LOCAL HuggingFace backend that (a) generates like the vLLM endpoint did and
# # # (b) captures residual-stream activations during that generation. It is a
# # # drop-in for Heuristic_wo_dspy.llm_complete so the whole SAGE pipeline drives a
# # # locally-loaded copy of the model and we can read hidden states.
# # #
# # # Why this is needed
# # # ------------------
# # # The pipeline routes every model call through llm_complete -> litellm ->
# # # the vLLM OpenAI endpoint, which returns TEXT ONLY. Hidden states are not
# # # available on that path. Probing therefore requires a locally-loaded model,
# # # exactly like collect_probe_data.py already does for the read-back experiment.
# # #
# # # What we capture per call (the Othello analog)
# # # ---------------------------------------------
# # # For each placement-candidate generation we record the residual stream at TWO
# # # token positions and several depths:
# # #   * "prompt_end" : the last prompt token, BEFORE any coordinate is emitted.
# # #                    Use this for accumulated-state / plan-vs-execution / a
# # #                    forward-looking legality probe (the model has not yet
# # #                    written the placement it is about to commit).
# # #   * "delta_end"  : the last generated token, AFTER the candidate placement is
# # #                    written. This is the direct Othello "probe at the move
# # #                    token" position; use it for board-state and legality of the
# # #                    candidate the model just proposed.
# # # Ground-truth labels are attached later, by probe_capture.py, from the parsed
# # # candidate + the prior committed objects (external geometry = "the rules").
# # #
# # # Architecture caveats (read once)
# # # --------------------------------
# # #   * Qwen3.5-27B is a unified multimodal, hybrid (Gated-DeltaNet + MoE) model.
# # #     output_hidden_states still yields a per-layer tuple (embeddings + L
# # #     blocks), and we treat hidden_states[i] as the residual stream after block
# # #     i. The clean additive-stream picture that linear Othello probes rely on is
# # #     not guaranteed for hybrid/MoE blocks -- validate, do not assume.
# # #   * This backend is TEXT-ONLY on purpose (you chose to probe LLM capability).
# # #     If a call passes images (the orientation refiner does), we raise, so run
# # #     the pipeline with orientation=False. Placement + scale are text-only.
# # #
# # # Set PROBE_MOCK=1 to exercise the wiring without a GPU: generation returns a
# # # tiny canned JSON and activations are random. Lets you validate probe_capture
# # # end-to-end before spending H200 time.
# # # =============================================================================
# # from __future__ import annotations

# # import os
# # import re
# # import json
# # from typing import Any, Dict, List, Optional

# # import numpy as np

# # # ---- module state -----------------------------------------------------------
# # _PROBE: "VLMProbe | None" = None
# # RECORDS: List[Dict[str, Any]] = []      # one dict per probed generation call
# # LAST_RECORD_IDX: int = -1               # index of the most recent record
# # _CTX: Dict[str, Any] = {}               # context set by the caller (scene/region/iter)

# # # Defaults mirror Heuristic_wo_dspy config (minus the litellm "openai/" prefix).
# # # NOTE: the generator used MAX_TOKENS=12000 for the *served* endpoint, where the
# # # server streams cheaply. LOCAL autoregressive decode pays per token, so 12000 is
# # # the stall. Placement/plan JSON is small; 2048 is generous. Raise per-call only
# # # if a JSON object actually gets truncated.
# # DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"
# # DEFAULT_TEMPERATURE = 0.5
# # DEFAULT_MAX_TOKENS = 2048
# # JSON_SYSTEM = (
# #     "You are a strict JSON generator. Output ONLY a single valid JSON value "
# #     "and nothing else. Do NOT include any reasoning, thinking, explanation, "
# #     "preamble, markdown, or code fences. Your entire response must be parseable "
# #     "by json.loads()."
# # )


# # def _select_layer_indices(num_hidden_layers: int, num_probe_layers: int) -> List[int]:
# #     """Evenly spaced block outputs in [1..L] (skip the embedding layer 0)."""
# #     L = num_hidden_layers
# #     k = min(num_probe_layers, L)
# #     idx = np.linspace(1, L, k).round().astype(int).tolist()
# #     return sorted(set(int(i) for i in idx))


# # def _parse_seed_from_prompt(prompt: str) -> Optional[int]:
# #     """The pipeline prepends 'Random seed: N'; reuse it to seed local sampling
# #     so best-of-N candidates diverge deterministically like the served run."""
# #     m = re.search(r"Random seed:\s*(\d+)", prompt)
# #     return int(m.group(1)) if m else None


# # class VLMProbe:
# #     def __init__(self, model_path: str, dtype: str = "bfloat16",
# #                  device_map: str = "auto", num_probe_layers: int = 8,
# #                  max_seq_len: int = 8192, random_init: bool = False):
# #         self.model_path = model_path
# #         self.num_probe_layers = num_probe_layers
# #         self.max_seq_len = max_seq_len
# #         self.random_init = random_init
# #         self.mock = os.environ.get("PROBE_MOCK", "0") == "1"
# #         self.model = None
# #         self.tok = None
# #         self.layers: List[int] = []
# #         self.hidden_size: int = 0
# #         self._call_counter = 0
# #         if not self.mock:
# #             self._load(dtype, device_map)
# #         else:
# #             # mock config so probe_capture can run without a GPU
# #             self.layers = _select_layer_indices(48, num_probe_layers)
# #             self.hidden_size = 64
# #             print(f"[probe_backend] PROBE_MOCK=1 -> layers {self.layers}, H={self.hidden_size}")

# #     # ---- model load (mirrors collect_probe_data.load_model) -----------------
# #     def _load(self, dtype: str, device_map: str):
# #         import torch
# #         from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# #         td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
# #               "float32": torch.float32}[dtype]
# #         self.tok = AutoTokenizer.from_pretrained(
# #             self.model_path, trust_remote_code=True, use_fast=True)
# #         if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
# #             self.tok.pad_token = self.tok.eos_token
# #         if self.random_init:
# #             # control model: same architecture, random weights. Heavy for 27B;
# #             # for a cheap control point --model-path at a small checkpoint.
# #             cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
# #             self.model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(td)
# #         else:
# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 self.model_path, trust_remote_code=True, torch_dtype=td,
# #                 device_map=device_map)
# #         self.model.config.output_hidden_states = True
# #         self.model.config.use_cache = True   # cache on for generation
# #         self.model.eval()
# #         nL = self.model.config.num_hidden_layers
# #         self.layers = _select_layer_indices(nL, self.num_probe_layers)
# #         self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))
# #         print(f"[probe_backend] loaded {self.model_path}: "
# #               f"num_hidden_layers={nL}; probing {self.layers}; H={self.hidden_size}")

# #     # ---- robust hidden-state forward (handles wrapper variants) -------------
# #     def _forward_hidden(self, input_ids, attention_mask):
# #         import torch
# #         with torch.inference_mode():
# #             # Try the top-level ForCausalLM forward first.
# #             try:
# #                 out = self.model(input_ids=input_ids, attention_mask=attention_mask,
# #                                  output_hidden_states=True, use_cache=False,
# #                                  return_dict=True)
# #                 if getattr(out, "hidden_states", None) is not None:
# #                     return out.hidden_states
# #             except TypeError:
# #                 pass
# #             # Fall back to the text backbone submodule (VL wrappers).
# #             for attr in ("model", "language_model"):
# #                 sub = getattr(self.model, attr, None)
# #                 if sub is None:
# #                     continue
# #                 inner = getattr(sub, "model", sub)
# #                 try:
# #                     out = inner(input_ids=input_ids, attention_mask=attention_mask,
# #                                 output_hidden_states=True, use_cache=False,
# #                                 return_dict=True)
# #                     if getattr(out, "hidden_states", None) is not None:
# #                         return out.hidden_states
# #                 except TypeError:
# #                     continue
# #         raise RuntimeError("Could not obtain hidden_states from the model; "
# #                            "check the wrapper's forward signature.")

# #     # ---- the drop-in --------------------------------------------------------
# #     def complete(self, prompt: str, images=None, system: Optional[str] = None,
# #                  temperature: float = DEFAULT_TEMPERATURE,
# #                  max_tokens: int = DEFAULT_MAX_TOKENS,
# #                  force_json: bool = True) -> str:
# #         global LAST_RECORD_IDX
# #         if images:
# #             raise NotImplementedError(
# #                 "probe_backend is text-only; run the pipeline with orientation=False "
# #                 "(the orientation refiner is the only image call).")

# #         seed = _parse_seed_from_prompt(prompt)
# #         self._call_counter += 1

# #         if self.mock:
# #             text = '[{"label": "sofa", "object_type": "bulk", "position": [0.0, 1.0], ' \
# #                    '"bbox": [-1.0, 0.0, 2.0, 1.0], "rotation_angle": 0.0}]'
# #             acts = {p: {li: np.random.randn(self.hidden_size).astype(np.float16)
# #                         for li in self.layers} for p in ("prompt_end", "delta_end")}
# #             RECORDS.append({"activations": acts, "output_text": text,
# #                             "ctx": dict(_CTX), "seed": seed, "prompt": prompt,
# #                             "system": system, "force_json": force_json,
# #                             "call_index": self._call_counter, "labels": {}})
# #             LAST_RECORD_IDX = len(RECORDS) - 1
# #             return text

# #         import torch
# #         sys_msg = system or (JSON_SYSTEM if force_json else None)
# #         messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
# #                    [{"role": "user", "content": prompt}]

# #         # Qwen3.x chat template: disable "thinking" so output is pure JSON.
# #         try:
# #             rendered = self.tok.apply_chat_template(
# #                 messages, add_generation_prompt=True, tokenize=False,
# #                 enable_thinking=False)
# #         except TypeError:
# #             rendered = self.tok.apply_chat_template(
# #                 messages, add_generation_prompt=True, tokenize=False)

# #         enc = self.tok(rendered, return_tensors="pt", truncation=True,
# #                        max_length=self.max_seq_len)
# #         dev = self.model.get_input_embeddings().weight.device
# #         input_ids = enc["input_ids"].to(dev)
# #         attn = enc.get("attention_mask")
# #         attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
# #         prompt_len = int(input_ids.shape[1])

# #         if seed is not None:
# #             torch.manual_seed(seed)

# #         # ---- generate (text out, drop-in) ----
# #         import time
# #         gen_kwargs = dict(max_new_tokens=max_tokens,
# #                           pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
# #         if temperature and temperature > 0:
# #             gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
# #         else:
# #             gen_kwargs.update(do_sample=False)
# #         t0 = time.time()
# #         with torch.inference_mode():
# #             full = self.model.generate(input_ids=input_ids, attention_mask=attn,
# #                                        **gen_kwargs)
# #         new_ids = full[0][prompt_len:]
# #         output_text = self.tok.decode(new_ids, skip_special_tokens=True)

# #         # visibility: first few calls print throughput + flag a runaway/thinking leak
# #         dt = time.time() - t0
# #         n_new = int(new_ids.shape[0])
# #         if self._call_counter <= 5 or n_new >= max_tokens:
# #             tps = n_new / dt if dt > 0 else 0.0
# #             note = ""
# #             if n_new >= max_tokens:
# #                 note = "  <-- hit max_new_tokens (truncated / no EOS; raise cap or check stop)"
# #             if "<think>" in output_text or "</think>" in output_text:
# #                 note += "  <-- OUTPUT CONTAINS <think>: thinking not suppressed, wasting tokens"
# #             print(f"[probe_backend] call {self._call_counter}: {n_new} tok in {dt:.1f}s "
# #                   f"({tps:.1f} tok/s){note}")

# #         # ---- capture activations at prompt_end and delta_end ----
# #         full_ids = full[:, : self.max_seq_len]
# #         full_attn = torch.ones_like(full_ids)
# #         hidden = self._forward_hidden(full_ids, full_attn)  # tuple len L+1, each [1,S,H]
# #         S = full_ids.shape[1]
# #         pos_prompt_end = prompt_len - 1
# #         # last non-eos generated token; fall back to last token
# #         eos = self.tok.eos_token_id
# #         pos_delta_end = S - 1
# #         while pos_delta_end > prompt_len and eos is not None and int(full_ids[0, pos_delta_end]) == eos:
# #             pos_delta_end -= 1

# #         acts: Dict[str, Dict[int, np.ndarray]] = {"prompt_end": {}, "delta_end": {}}
# #         for li in self.layers:
# #             h = hidden[li][0]  # [S, H]
# #             acts["prompt_end"][li] = h[pos_prompt_end].to(torch.float16).cpu().numpy()
# #             acts["delta_end"][li] = h[pos_delta_end].to(torch.float16).cpu().numpy()

# #         RECORDS.append({"activations": acts, "output_text": output_text,
# #                         "ctx": dict(_CTX), "seed": seed, "prompt": prompt,
# #                         "system": system, "force_json": force_json,
# #                         "call_index": self._call_counter, "labels": {}})
# #         LAST_RECORD_IDX = len(RECORDS) - 1
# #         return output_text


# # # ---- module-level API (what probe_capture / the pipeline use) ---------------
# # def init_probe(model_path: str = DEFAULT_MODEL_PATH, dtype: str = "bfloat16",
# #                device_map: str = "auto", num_probe_layers: int = 8,
# #                random_init: bool = False) -> VLMProbe:
# #     global _PROBE
# #     _PROBE = VLMProbe(model_path, dtype, device_map, num_probe_layers,
# #                       random_init=random_init)
# #     return _PROBE


# # def probed_complete(prompt: str, images=None, system: Optional[str] = None,
# #                     temperature: float = DEFAULT_TEMPERATURE,
# #                     max_tokens: int = DEFAULT_MAX_TOKENS,
# #                     force_json: bool = True) -> str:
# #     """Exact drop-in for Heuristic_wo_dspy.llm_complete."""
# #     if _PROBE is None:
# #         raise RuntimeError("call probe_backend.init_probe(...) first")
# #     return _PROBE.complete(prompt, images, system, temperature, max_tokens, force_json)


# # def set_context(**kw):
# #     """Caller sets scene_id/region/iteration/etc. before a probed call."""
# #     _CTX.clear()
# #     _CTX.update(kw)


# # def attach_labels(idx: int, **labels):
# #     """probe_capture calls this after computing geometric GT for record idx."""
# #     if 0 <= idx < len(RECORDS):
# #         RECORDS[idx]["labels"].update(labels)


# # def reset_records():
# #     RECORDS.clear()


# # def get_probe() -> "VLMProbe | None":
# #     return _PROBE


# #!/usr/bin/env python3
# # =============================================================================
# # probe_backend.py
# # -----------------------------------------------------------------------------
# # A LOCAL HuggingFace backend that (a) generates like the vLLM endpoint did and
# # (b) captures residual-stream activations during that generation. It is a
# # drop-in for Heuristic_wo_dspy.llm_complete so the whole SAGE pipeline drives a
# # locally-loaded copy of the model and we can read hidden states.
# #
# # Why this is needed
# # ------------------
# # The pipeline routes every model call through llm_complete -> litellm ->
# # the vLLM OpenAI endpoint, which returns TEXT ONLY. Hidden states are not
# # available on that path. Probing therefore requires a locally-loaded model,
# # exactly like collect_probe_data.py already does for the read-back experiment.
# #
# # What we capture per call (the Othello analog)
# # ---------------------------------------------
# # For each placement-candidate generation we record the residual stream at TWO
# # token positions and several depths:
# #   * "prompt_end" : the last prompt token, BEFORE any coordinate is emitted.
# #                    Use this for accumulated-state / plan-vs-execution / a
# #                    forward-looking legality probe (the model has not yet
# #                    written the placement it is about to commit).
# #   * "delta_end"  : the last generated token, AFTER the candidate placement is
# #                    written. This is the direct Othello "probe at the move
# #                    token" position; use it for board-state and legality of the
# #                    candidate the model just proposed.
# # Ground-truth labels are attached later, by probe_capture.py, from the parsed
# # candidate + the prior committed objects (external geometry = "the rules").
# #
# # Architecture caveats (read once)
# # --------------------------------
# #   * Qwen3.5-27B is a unified multimodal, hybrid (Gated-DeltaNet + MoE) model.
# #     output_hidden_states still yields a per-layer tuple (embeddings + L
# #     blocks), and we treat hidden_states[i] as the residual stream after block
# #     i. The clean additive-stream picture that linear Othello probes rely on is
# #     not guaranteed for hybrid/MoE blocks -- validate, do not assume.
# #   * This backend is TEXT-ONLY on purpose (you chose to probe LLM capability).
# #     If a call passes images (the orientation refiner does), we raise, so run
# #     the pipeline with orientation=False. Placement + scale are text-only.
# #
# # Set PROBE_MOCK=1 to exercise the wiring without a GPU: generation returns a
# # tiny canned JSON and activations are random. Lets you validate probe_capture
# # end-to-end before spending H200 time.
# # =============================================================================
# from __future__ import annotations

# import os
# import re
# import json
# from typing import Any, Dict, List, Optional

# import numpy as np

# # ---- module state -----------------------------------------------------------
# _PROBE: "VLMProbe | None" = None
# RECORDS: List[Dict[str, Any]] = []      # one dict per probed generation call
# LAST_RECORD_IDX: int = -1               # index of the most recent record
# _CTX: Dict[str, Any] = {}               # context set by the caller (scene/region/iter)
# DET_SEED = None                         # if set, seed torch by DET_SEED + call_index


# def set_det_seed(s):
#     """Enable deterministic per-call seeding (None disables)."""
#     global DET_SEED
#     DET_SEED = s


# def reset_call_counter():
#     """Reset the per-call counter so both runs of a pair share the seed sequence."""
#     if _PROBE is not None:
#         _PROBE._call_counter = 0

# # Defaults mirror Heuristic_wo_dspy config (minus the litellm "openai/" prefix).
# # NOTE: the generator used MAX_TOKENS=12000 for the *served* endpoint, where the
# # server streams cheaply. LOCAL autoregressive decode pays per token, so 12000 is
# # the stall. Placement/plan JSON is small; 2048 is generous. Raise per-call only
# # if a JSON object actually gets truncated.
# DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"
# DEFAULT_TEMPERATURE = 0.5
# DEFAULT_MAX_TOKENS = 2048
# JSON_SYSTEM = (
#     "You are a strict JSON generator. Output ONLY a single valid JSON value "
#     "and nothing else. Do NOT include any reasoning, thinking, explanation, "
#     "preamble, markdown, or code fences. Your entire response must be parseable "
#     "by json.loads()."
# )


# def _select_layer_indices(num_hidden_layers: int, num_probe_layers: int) -> List[int]:
#     """Evenly spaced block outputs in [1..L] (skip the embedding layer 0)."""
#     L = num_hidden_layers
#     k = min(num_probe_layers, L)
#     idx = np.linspace(1, L, k).round().astype(int).tolist()
#     return sorted(set(int(i) for i in idx))


# def _parse_seed_from_prompt(prompt: str) -> Optional[int]:
#     """The pipeline prepends 'Random seed: N'; reuse it to seed local sampling
#     so best-of-N candidates diverge deterministically like the served run."""
#     m = re.search(r"Random seed:\s*(\d+)", prompt)
#     return int(m.group(1)) if m else None


# class VLMProbe:
#     def __init__(self, model_path: str, dtype: str = "bfloat16",
#                  device_map: str = "auto", num_probe_layers: int = 8,
#                  max_seq_len: int = 8192, random_init: bool = False):
#         self.model_path = model_path
#         self.num_probe_layers = num_probe_layers
#         self.max_seq_len = max_seq_len
#         self.random_init = random_init
#         self.mock = os.environ.get("PROBE_MOCK", "0") == "1"
#         self.model = None
#         self.tok = None
#         self.layers: List[int] = []
#         self.hidden_size: int = 0
#         self._call_counter = 0
#         if not self.mock:
#             self._load(dtype, device_map)
#         else:
#             # mock config so probe_capture can run without a GPU
#             self.layers = _select_layer_indices(48, num_probe_layers)
#             self.hidden_size = 64
#             print(f"[probe_backend] PROBE_MOCK=1 -> layers {self.layers}, H={self.hidden_size}")

#     # ---- model load (mirrors collect_probe_data.load_model) -----------------
#     def _load(self, dtype: str, device_map: str):
#         import torch
#         from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
#         td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
#               "float32": torch.float32}[dtype]
#         self.tok = AutoTokenizer.from_pretrained(
#             self.model_path, trust_remote_code=True, use_fast=True)
#         if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
#             self.tok.pad_token = self.tok.eos_token
#         if self.random_init:
#             # control model: same architecture, random weights. Heavy for 27B;
#             # for a cheap control point --model-path at a small checkpoint.
#             cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
#             self.model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(td)
#         else:
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path, trust_remote_code=True, torch_dtype=td,
#                 device_map=device_map)
#         self.model.config.output_hidden_states = True
#         self.model.config.use_cache = True   # cache on for generation
#         self.model.eval()
#         nL = self.model.config.num_hidden_layers
#         self.layers = _select_layer_indices(nL, self.num_probe_layers)
#         self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))
#         print(f"[probe_backend] loaded {self.model_path}: "
#               f"num_hidden_layers={nL}; probing {self.layers}; H={self.hidden_size}")

#     # ---- robust hidden-state forward (handles wrapper variants) -------------
#     def _forward_hidden(self, input_ids, attention_mask):
#         import torch
#         with torch.inference_mode():
#             # Try the top-level ForCausalLM forward first.
#             try:
#                 out = self.model(input_ids=input_ids, attention_mask=attention_mask,
#                                  output_hidden_states=True, use_cache=False,
#                                  return_dict=True)
#                 if getattr(out, "hidden_states", None) is not None:
#                     return out.hidden_states
#             except TypeError:
#                 pass
#             # Fall back to the text backbone submodule (VL wrappers).
#             for attr in ("model", "language_model"):
#                 sub = getattr(self.model, attr, None)
#                 if sub is None:
#                     continue
#                 inner = getattr(sub, "model", sub)
#                 try:
#                     out = inner(input_ids=input_ids, attention_mask=attention_mask,
#                                 output_hidden_states=True, use_cache=False,
#                                 return_dict=True)
#                     if getattr(out, "hidden_states", None) is not None:
#                         return out.hidden_states
#                 except TypeError:
#                     continue
#         raise RuntimeError("Could not obtain hidden_states from the model; "
#                            "check the wrapper's forward signature.")

#     # ---- the drop-in --------------------------------------------------------
#     def complete(self, prompt: str, images=None, system: Optional[str] = None,
#                  temperature: float = DEFAULT_TEMPERATURE,
#                  max_tokens: int = DEFAULT_MAX_TOKENS,
#                  force_json: bool = True) -> str:
#         global LAST_RECORD_IDX
#         if images:
#             raise NotImplementedError(
#                 "probe_backend is text-only; run the pipeline with orientation=False "
#                 "(the orientation refiner is the only image call).")

#         seed = _parse_seed_from_prompt(prompt)
#         self._call_counter += 1

#         if self.mock:
#             text = '[{"label": "sofa", "object_type": "bulk", "position": [0.0, 1.0], ' \
#                    '"bbox": [-1.0, 0.0, 2.0, 1.0], "rotation_angle": 0.0}]'
#             acts = {p: {li: np.random.randn(self.hidden_size).astype(np.float16)
#                         for li in self.layers} for p in ("prompt_end", "delta_end")}
#             RECORDS.append({"activations": acts, "output_text": text,
#                             "ctx": dict(_CTX), "seed": seed, "prompt": prompt,
#                             "system": system, "force_json": force_json,
#                             "call_index": self._call_counter, "labels": {}})
#             LAST_RECORD_IDX = len(RECORDS) - 1
#             return text

#         import torch
#         sys_msg = system or (JSON_SYSTEM if force_json else None)
#         messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
#                    [{"role": "user", "content": prompt}]

#         # Qwen3.x chat template: disable "thinking" so output is pure JSON.
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

#         # Determinism for matched pairs: a per-call seed keyed by call index makes
#         # every call reproducible AND identical across baseline/adversarial runs
#         # (so divergence is only from the token), without greedy over-elaboration.
#         if DET_SEED is not None:
#             torch.manual_seed(DET_SEED + self._call_counter)
#         elif seed is not None:
#             torch.manual_seed(seed)

#         # ---- generate (text out, drop-in) ----
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

#         # visibility: first few calls print throughput + flag a runaway/thinking leak
#         dt = time.time() - t0
#         n_new = int(new_ids.shape[0])
#         if self._call_counter <= 5 or n_new >= max_tokens:
#             tps = n_new / dt if dt > 0 else 0.0
#             note = ""
#             if n_new >= max_tokens:
#                 note = "  <-- hit max_new_tokens (truncated / no EOS; raise cap or check stop)"
#             if "<think>" in output_text or "</think>" in output_text:
#                 note += "  <-- OUTPUT CONTAINS <think>: thinking not suppressed, wasting tokens"
#             print(f"[probe_backend] call {self._call_counter}: {n_new} tok in {dt:.1f}s "
#                   f"({tps:.1f} tok/s){note}")

#         # ---- capture activations at prompt_end and delta_end ----
#         full_ids = full[:, : self.max_seq_len]
#         full_attn = torch.ones_like(full_ids)
#         hidden = self._forward_hidden(full_ids, full_attn)  # tuple len L+1, each [1,S,H]
#         S = full_ids.shape[1]
#         pos_prompt_end = prompt_len - 1
#         # last non-eos generated token; fall back to last token
#         eos = self.tok.eos_token_id
#         pos_delta_end = S - 1
#         while pos_delta_end > prompt_len and eos is not None and int(full_ids[0, pos_delta_end]) == eos:
#             pos_delta_end -= 1

#         acts: Dict[str, Dict[int, np.ndarray]] = {"prompt_end": {}, "delta_end": {}}
#         for li in self.layers:
#             h = hidden[li][0]  # [S, H]
#             acts["prompt_end"][li] = h[pos_prompt_end].to(torch.float16).cpu().numpy()
#             acts["delta_end"][li] = h[pos_delta_end].to(torch.float16).cpu().numpy()

#         RECORDS.append({"activations": acts, "output_text": output_text,
#                         "ctx": dict(_CTX), "seed": seed, "prompt": prompt,
#                         "system": system, "force_json": force_json,
#                         "call_index": self._call_counter, "labels": {}})
#         LAST_RECORD_IDX = len(RECORDS) - 1
#         return output_text


# # ---- module-level API (what probe_capture / the pipeline use) ---------------
# def init_probe(model_path: str = DEFAULT_MODEL_PATH, dtype: str = "bfloat16",
#                device_map: str = "auto", num_probe_layers: int = 8,
#                random_init: bool = False) -> VLMProbe:
#     global _PROBE
#     _PROBE = VLMProbe(model_path, dtype, device_map, num_probe_layers,
#                       random_init=random_init)
#     return _PROBE


# def probed_complete(prompt: str, images=None, system: Optional[str] = None,
#                     temperature: float = DEFAULT_TEMPERATURE,
#                     max_tokens: int = DEFAULT_MAX_TOKENS,
#                     force_json: bool = True) -> str:
#     """Exact drop-in for Heuristic_wo_dspy.llm_complete."""
#     if _PROBE is None:
#         raise RuntimeError("call probe_backend.init_probe(...) first")
#     return _PROBE.complete(prompt, images, system, temperature, max_tokens, force_json)


# def set_context(**kw):
#     """Caller sets scene_id/region/iteration/etc. before a probed call."""
#     _CTX.clear()
#     _CTX.update(kw)


# def attach_labels(idx: int, **labels):
#     """probe_capture calls this after computing geometric GT for record idx."""
#     if 0 <= idx < len(RECORDS):
#         RECORDS[idx]["labels"].update(labels)


# def reset_records():
#     RECORDS.clear()


# def get_probe() -> "VLMProbe | None":
#     return _PROBE

#!/usr/bin/env python3
# =============================================================================
# probe_backend.py
# -----------------------------------------------------------------------------
# A LOCAL HuggingFace backend that (a) generates like the vLLM endpoint did and
# (b) captures residual-stream activations during that generation. It is a
# drop-in for Heuristic_wo_dspy.llm_complete so the whole SAGE pipeline drives a
# locally-loaded copy of the model and we can read hidden states.
#
# Why this is needed
# ------------------
# The pipeline routes every model call through llm_complete -> litellm ->
# the vLLM OpenAI endpoint, which returns TEXT ONLY. Hidden states are not
# available on that path. Probing therefore requires a locally-loaded model,
# exactly like collect_probe_data.py already does for the read-back experiment.
#
# What we capture per call (the Othello analog)
# ---------------------------------------------
# For each placement-candidate generation we record the residual stream at TWO
# token positions and several depths:
#   * "prompt_end" : the last prompt token, BEFORE any coordinate is emitted.
#                    Use this for accumulated-state / plan-vs-execution / a
#                    forward-looking legality probe (the model has not yet
#                    written the placement it is about to commit).
#   * "delta_end"  : the last generated token, AFTER the candidate placement is
#                    written. This is the direct Othello "probe at the move
#                    token" position; use it for board-state and legality of the
#                    candidate the model just proposed.
# Ground-truth labels are attached later, by probe_capture.py, from the parsed
# candidate + the prior committed objects (external geometry = "the rules").
#
# Architecture caveats (read once)
# --------------------------------
#   * Qwen3.5-27B is a unified multimodal, hybrid (Gated-DeltaNet + MoE) model.
#     output_hidden_states still yields a per-layer tuple (embeddings + L
#     blocks), and we treat hidden_states[i] as the residual stream after block
#     i. The clean additive-stream picture that linear Othello probes rely on is
#     not guaranteed for hybrid/MoE blocks -- validate, do not assume.
#   * This backend is TEXT-ONLY on purpose (you chose to probe LLM capability).
#     If a call passes images (the orientation refiner does), we raise, so run
#     the pipeline with orientation=False. Placement + scale are text-only.
#
# Set PROBE_MOCK=1 to exercise the wiring without a GPU: generation returns a
# tiny canned JSON and activations are random. Lets you validate probe_capture
# end-to-end before spending H200 time.
# =============================================================================
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

import numpy as np

# ---- module state -----------------------------------------------------------
_PROBE: "VLMProbe | None" = None
RECORDS: List[Dict[str, Any]] = []      # one dict per probed generation call
LAST_RECORD_IDX: int = -1               # index of the most recent record
_CTX: Dict[str, Any] = {}               # context set by the caller (scene/region/iter)
DET_SEED = None                         # if set, seed torch by DET_SEED + call_index


def set_det_seed(s):
    """Enable deterministic per-call seeding (None disables)."""
    global DET_SEED
    DET_SEED = s


def reset_call_counter():
    """Reset the per-call counter so both runs of a pair share the seed sequence."""
    if _PROBE is not None:
        _PROBE._call_counter = 0

# Defaults mirror Heuristic_wo_dspy config (minus the litellm "openai/" prefix).
# NOTE: the generator used MAX_TOKENS=12000 for the *served* endpoint, where the
# server streams cheaply. LOCAL autoregressive decode pays per token, so 12000 is
# the stall. Placement/plan JSON is small; 2048 is generous. Raise per-call only
# if a JSON object actually gets truncated.
DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 2048
JSON_SYSTEM = (
    "You are a strict JSON generator. Output ONLY a single valid JSON value "
    "and nothing else. Do NOT include any reasoning, thinking, explanation, "
    "preamble, markdown, or code fences. Your entire response must be parseable "
    "by json.loads()."
)


def _select_layer_indices(num_hidden_layers: int, num_probe_layers: int) -> List[int]:
    """Evenly spaced block outputs in [1..L] (skip the embedding layer 0)."""
    L = num_hidden_layers
    k = min(num_probe_layers, L)
    idx = np.linspace(1, L, k).round().astype(int).tolist()
    return sorted(set(int(i) for i in idx))


def _parse_seed_from_prompt(prompt: str) -> Optional[int]:
    """The pipeline prepends 'Random seed: N'; reuse it to seed local sampling
    so best-of-N candidates diverge deterministically like the served run."""
    m = re.search(r"Random seed:\s*(\d+)", prompt)
    return int(m.group(1)) if m else None


class VLMProbe:
    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 device_map: str = "auto", num_probe_layers: int = 8,
                 max_seq_len: int = 8192, random_init: bool = False):
        self.model_path = model_path
        self.num_probe_layers = num_probe_layers
        self.max_seq_len = max_seq_len
        self.random_init = random_init
        self.mock = os.environ.get("PROBE_MOCK", "0") == "1"
        self.model = None
        self.tok = None
        self.layers: List[int] = []
        self.hidden_size: int = 0
        self._call_counter = 0
        if not self.mock:
            self._load(dtype, device_map)
        else:
            # mock config so probe_capture can run without a GPU
            self.layers = _select_layer_indices(48, num_probe_layers)
            self.hidden_size = 64
            print(f"[probe_backend] PROBE_MOCK=1 -> layers {self.layers}, H={self.hidden_size}")

    # ---- model load (mirrors collect_probe_data.load_model) -----------------
    def _load(self, dtype: str, device_map: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        if self.random_init:
            # control model: same architecture, random weights. Heavy for 27B;
            # for a cheap control point --model-path at a small checkpoint.
            cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(td)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=td,
                device_map=device_map)
        self.model.config.output_hidden_states = True
        self.model.config.use_cache = True   # cache on for generation
        self.model.eval()
        nL = self.model.config.num_hidden_layers
        self.layers = _select_layer_indices(nL, self.num_probe_layers)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        print(f"[probe_backend] loaded {self.model_path}: "
              f"num_hidden_layers={nL}; probing {self.layers}; H={self.hidden_size}")

    # ---- robust hidden-state forward (handles wrapper variants) -------------
    def _forward_hidden(self, input_ids, attention_mask):
        import torch
        with torch.inference_mode():
            # Try the top-level ForCausalLM forward first.
            try:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True, use_cache=False,
                                 return_dict=True)
                if getattr(out, "hidden_states", None) is not None:
                    return out.hidden_states
            except TypeError:
                pass
            # Fall back to the text backbone submodule (VL wrappers).
            for attr in ("model", "language_model"):
                sub = getattr(self.model, attr, None)
                if sub is None:
                    continue
                inner = getattr(sub, "model", sub)
                try:
                    out = inner(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, use_cache=False,
                                return_dict=True)
                    if getattr(out, "hidden_states", None) is not None:
                        return out.hidden_states
                except TypeError:
                    continue
        raise RuntimeError("Could not obtain hidden_states from the model; "
                           "check the wrapper's forward signature.")

    # ---- the drop-in --------------------------------------------------------
    def complete(self, prompt: str, images=None, system: Optional[str] = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 force_json: bool = True) -> str:
        global LAST_RECORD_IDX
        if images:
            raise NotImplementedError(
                "probe_backend is text-only; run the pipeline with orientation=False "
                "(the orientation refiner is the only image call).")

        seed = _parse_seed_from_prompt(prompt)
        self._call_counter += 1

        if self.mock:
            text = '[{"label": "sofa", "object_type": "bulk", "position": [0.0, 1.0], ' \
                   '"bbox": [-1.0, 0.0, 2.0, 1.0], "rotation_angle": 0.0}]'
            acts = {p: {li: np.random.randn(self.hidden_size).astype(np.float16)
                        for li in self.layers} for p in ("prompt_end", "delta_end")}
            RECORDS.append({"activations": acts, "output_text": text,
                            "ctx": dict(_CTX), "seed": seed, "prompt": prompt,
                            "system": system, "force_json": force_json,
                            "call_index": self._call_counter, "labels": {}})
            LAST_RECORD_IDX = len(RECORDS) - 1
            _CTX.clear()   # see note at the real-path clear below
            return text

        import torch
        sys_msg = system or (JSON_SYSTEM if force_json else None)
        messages = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + \
                   [{"role": "user", "content": prompt}]

        # Qwen3.x chat template: disable "thinking" so output is pure JSON.
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

        # Determinism for matched pairs: a per-call seed keyed by call index makes
        # every call reproducible AND identical across baseline/adversarial runs
        # (so divergence is only from the token), without greedy over-elaboration.
        if DET_SEED is not None:
            torch.manual_seed(DET_SEED + self._call_counter)
        elif seed is not None:
            torch.manual_seed(seed)

        # ---- generate (text out, drop-in) ----
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

        # visibility: first few calls print throughput + flag a runaway/thinking leak
        dt = time.time() - t0
        n_new = int(new_ids.shape[0])
        if self._call_counter <= 5 or n_new >= max_tokens:
            tps = n_new / dt if dt > 0 else 0.0
            note = ""
            if n_new >= max_tokens:
                note = "  <-- hit max_new_tokens (truncated / no EOS; raise cap or check stop)"
            if "<think>" in output_text or "</think>" in output_text:
                note += "  <-- OUTPUT CONTAINS <think>: thinking not suppressed, wasting tokens"
            print(f"[probe_backend] call {self._call_counter}: {n_new} tok in {dt:.1f}s "
                  f"({tps:.1f} tok/s){note}")

        # ---- capture activations at prompt_end and delta_end ----
        full_ids = full[:, : self.max_seq_len]
        full_attn = torch.ones_like(full_ids)
        hidden = self._forward_hidden(full_ids, full_attn)  # tuple len L+1, each [1,S,H]
        S = full_ids.shape[1]
        pos_prompt_end = prompt_len - 1
        # last non-eos generated token; fall back to last token
        eos = self.tok.eos_token_id
        pos_delta_end = S - 1
        while pos_delta_end > prompt_len and eos is not None and int(full_ids[0, pos_delta_end]) == eos:
            pos_delta_end -= 1

        acts: Dict[str, Dict[int, np.ndarray]] = {"prompt_end": {}, "delta_end": {}}
        for li in self.layers:
            h = hidden[li][0]  # [S, H]
            acts["prompt_end"][li] = h[pos_prompt_end].to(torch.float16).cpu().numpy()
            acts["delta_end"][li] = h[pos_delta_end].to(torch.float16).cpu().numpy()

        RECORDS.append({"activations": acts, "output_text": output_text,
                        "ctx": dict(_CTX), "seed": seed, "prompt": prompt,
                        "system": system, "force_json": force_json,
                        "call_index": self._call_counter, "labels": {}})
        LAST_RECORD_IDX = len(RECORDS) - 1
        # _CTX is set by the caller for the NEXT call only. Clear it once consumed, or
        # non-placement calls (description/plan) inherit stale scene/region/iteration.
        _CTX.clear()
        return output_text


# ---- module-level API (what probe_capture / the pipeline use) ---------------
def init_probe(model_path: str = DEFAULT_MODEL_PATH, dtype: str = "bfloat16",
               device_map: str = "auto", num_probe_layers: int = 8,
               random_init: bool = False) -> VLMProbe:
    global _PROBE
    _PROBE = VLMProbe(model_path, dtype, device_map, num_probe_layers,
                      random_init=random_init)
    return _PROBE


def probed_complete(prompt: str, images=None, system: Optional[str] = None,
                    temperature: float = DEFAULT_TEMPERATURE,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    force_json: bool = True) -> str:
    """Exact drop-in for Heuristic_wo_dspy.llm_complete."""
    if _PROBE is None:
        raise RuntimeError("call probe_backend.init_probe(...) first")
    return _PROBE.complete(prompt, images, system, temperature, max_tokens, force_json)


def set_context(**kw):
    """Caller sets scene_id/region/iteration/etc. before a probed call."""
    _CTX.clear()
    _CTX.update(kw)


def attach_labels(idx: int, **labels):
    """probe_capture calls this after computing geometric GT for record idx."""
    if 0 <= idx < len(RECORDS):
        RECORDS[idx]["labels"].update(labels)


def reset_records():
    RECORDS.clear()


def get_probe() -> "VLMProbe | None":
    return _PROBE