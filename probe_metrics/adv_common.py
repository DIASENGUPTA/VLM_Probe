# # # # # # #!/usr/bin/env python3
# # # # # # # =============================================================================
# # # # # # # adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# # # # # # # -----------------------------------------------------------------------------
# # # # # # # Everything the levels share:
# # # # # # #   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
# # # # # # #                          so BASELINE vs ADVERSARIAL differ ONLY by the token
# # # # # # #                          (no sampling confound). Wraps the patched llm_complete;
# # # # # # #                          no edit to probe_backend.
# # # # # # #   * run_once()         : drive the real SAGE pipeline in-process for a set of
# # # # # # #                          benchmarks, optionally injecting an adversarial note
# # # # # # #                          into the environment note.
# # # # # # #   * snapshot/save/load : persist the captured records so each level is a cheap
# # # # # # #                          offline re-analysis of the SAME generation.
# # # # # # #   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
# # # # # # #
# # # # # # # Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# # # # # # # many times as you like over the saved .pkl snapshots.
# # # # # # # =============================================================================
# # # # # # from __future__ import annotations

# # # # # # import copy
# # # # # # import glob
# # # # # # import json
# # # # # # import os
# # # # # # import pickle
# # # # # # import re

# # # # # # import numpy as np

# # # # # # import probe_backend
# # # # # # import probe_capture


# # # # # # # ---- benchmark parsing (same extraction as run_probed_generation) -----------
# # # # # # def parse_benchmark(path: str):
# # # # # #     with open(path) as f:
# # # # # #         data = json.load(f)
# # # # # #     task = data.get("task_description", "This is an indoor room.")
# # # # # #     room = data.get("room_dimension", {})
# # # # # #     dim_x = int(room.get("room_dimension_x", 10))
# # # # # #     dim_y = int(room.get("room_dimension_y", 10))
# # # # # #     assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
# # # # # #     name = os.path.splitext(os.path.basename(path))[0]
# # # # # #     rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
# # # # # #     return f"{rtype}_{name}", task, assets, dim_x, dim_y


# # # # # # def list_specs(benchmark_glob, max_benchmarks=None):
# # # # # #     specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
# # # # # #     if max_benchmarks:
# # # # # #         specs = specs[:max_benchmarks]
# # # # # #     if not specs:
# # # # # #         raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
# # # # # #     return specs


# # # # # # # ---- determinism: force greedy so pairs are comparable ----------------------
# # # # # # def force_greedy():
# # # # # #     import Heuristic_wo_dspy as H
# # # # # #     inner = H.llm_complete  # probe_backend.probed_complete after enable_probing

# # # # # #     def greedy(prompt, images=None, system=None, temperature=0.5,
# # # # # #                max_tokens=probe_backend.DEFAULT_MAX_TOKENS, force_json=True):
# # # # # #         return inner(prompt, images, system, 0.0, max_tokens, force_json)

# # # # # #     H.llm_complete = greedy
# # # # # #     print("[adv] forced greedy decoding (temperature=0) for deterministic pairs")


# # # # # # # ---- capture -> plain picklable snapshot ------------------------------------
# # # # # # def snapshot_records():
# # # # # #     out = []
# # # # # #     for r in probe_backend.RECORDS:
# # # # # #         out.append({
# # # # # #             "output_text": r.get("output_text", ""),
# # # # # #             "prompt": r.get("prompt", ""),
# # # # # #             "system": r.get("system"),
# # # # # #             "force_json": r.get("force_json", True),
# # # # # #             "ctx": dict(r.get("ctx", {})),
# # # # # #             "call_index": r.get("call_index"),
# # # # # #             "labels": copy.deepcopy(r.get("labels", {})),
# # # # # #             "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
# # # # # #                             for p, d in r.get("activations", {}).items()},
# # # # # #         })
# # # # # #     return out


# # # # # # def save_snapshot(records, path):
# # # # # #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# # # # # #     with open(path, "wb") as f:
# # # # # #         pickle.dump(records, f)
# # # # # #     print(f"[adv] saved {len(records)} records -> {path}")


# # # # # # def load_snapshot(path):
# # # # # #     with open(path, "rb") as f:
# # # # # #         return pickle.load(f)


# # # # # # # ---- run the real pipeline once ---------------------------------------------
# # # # # # def run_once(specs, adv_note, save_dir_root):
# # # # # #     """Drive SAGE in-process; inject adv_note into the environment note.
# # # # # #     Requires enable_probing() + force_greedy() to have been called already.

# # # # # #     IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
# # # # # #     so both runs share identical geometry and the ONLY difference is the token in
# # # # # #     the prompt text (env_note). This isolates the token's effect. env_note itself
# # # # # #     is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
# # # # # #     note reaches description, object-choice, plan, and placement prompts."""
# # # # # #     probe_backend.reset_records()
# # # # # #     import Heuristic_wo_dspy as H
# # # # # #     import utils_copy_img
# # # # # #     for name, task, assets, dim_x, dim_y in specs:
# # # # # #         instruction = task + (adv_note or "")            # only the prompt text carries the note
# # # # # #         save_dir = os.path.join(save_dir_root, name)
# # # # # #         os.makedirs(save_dir, exist_ok=True)
# # # # # #         probe_capture.set_scene(name)
# # # # # #         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
# # # # # #         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
# # # # # #         if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
# # # # # #             print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
# # # # # #             continue
# # # # # #         planner = H.ScenePlanner(save_dir=save_dir)
# # # # # #         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
# # # # # #                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# # # # # #                         orientation=False, scale=False)
# # # # # #     return snapshot_records()


# # # # # # # ---- alignment + parsing (used by L1 and L2) --------------------------------
# # # # # # def placement_key(rec):
# # # # # #     """Placement calls carry labels + full ctx; key them by scene/region/iter."""
# # # # # #     c = rec["ctx"]
# # # # # #     if rec["labels"]:
# # # # # #         return (str(c.get("scene_id")), int(c.get("region_id", -1)), int(c.get("iteration", -1)))
# # # # # #     return None


# # # # # # def align(rec_a, rec_b):
# # # # # #     """Return (call-index stream pairs, placement-key pairs, (len_a, len_b))."""
# # # # # #     n = min(len(rec_a), len(rec_b))
# # # # # #     stream = list(zip(rec_a[:n], rec_b[:n]))
# # # # # #     pa = {placement_key(r): r for r in rec_a if placement_key(r)}
# # # # # #     pb = {placement_key(r): r for r in rec_b if placement_key(r)}
# # # # # #     keys = sorted(k for k in pa if k in pb)
# # # # # #     placement_pairs = [(pa[k], pb[k]) for k in keys]
# # # # # #     return stream, placement_pairs, (len(rec_a), len(rec_b))


# # # # # # def parse_objs(text):
# # # # # #     """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
# # # # # #     try:
# # # # # #         data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
# # # # # #     except Exception:
# # # # # #         return {}
# # # # # #     out = {}
# # # # # #     for o in (data if isinstance(data, list) else []):
# # # # # #         if isinstance(o, dict) and "label" in o:
# # # # # #             p = o.get("position")
# # # # # #             if isinstance(p, (list, tuple)) and len(p) >= 2:
# # # # # #                 out[str(o["label"])] = (float(p[0]), float(p[1]))
# # # # # #     return out


# # # # # #!/usr/bin/env python3
# # # # # # =============================================================================
# # # # # # adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# # # # # # -----------------------------------------------------------------------------
# # # # # # Everything the levels share:
# # # # # #   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
# # # # # #                          so BASELINE vs ADVERSARIAL differ ONLY by the token
# # # # # #                          (no sampling confound). Wraps the patched llm_complete;
# # # # # #                          no edit to probe_backend.
# # # # # #   * run_once()         : drive the real SAGE pipeline in-process for a set of
# # # # # #                          benchmarks, optionally injecting an adversarial note
# # # # # #                          into the environment note.
# # # # # #   * snapshot/save/load : persist the captured records so each level is a cheap
# # # # # #                          offline re-analysis of the SAME generation.
# # # # # #   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
# # # # # #
# # # # # # Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# # # # # # many times as you like over the saved .pkl snapshots.
# # # # # # =============================================================================
# # # # # from __future__ import annotations

# # # # # import copy
# # # # # import glob
# # # # # import json
# # # # # import os
# # # # # import pickle
# # # # # import re

# # # # # import numpy as np

# # # # # import probe_backend
# # # # # import probe_capture


# # # # # # ---- benchmark parsing (same extraction as run_probed_generation) -----------
# # # # # def parse_benchmark(path: str):
# # # # #     with open(path) as f:
# # # # #         data = json.load(f)
# # # # #     task = data.get("task_description", "This is an indoor room.")
# # # # #     room = data.get("room_dimension", {})
# # # # #     dim_x = int(room.get("room_dimension_x", 10))
# # # # #     dim_y = int(room.get("room_dimension_y", 10))
# # # # #     assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
# # # # #     name = os.path.splitext(os.path.basename(path))[0]
# # # # #     rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
# # # # #     return f"{rtype}_{name}", task, assets, dim_x, dim_y


# # # # # def list_specs(benchmark_glob, max_benchmarks=None):
# # # # #     specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
# # # # #     if max_benchmarks:
# # # # #         specs = specs[:max_benchmarks]
# # # # #     if not specs:
# # # # #         raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
# # # # #     return specs


# # # # # # ---- determinism: make pairs comparable ------------------------------------
# # # # # def force_greedy(max_tokens=8192):
# # # # #     """Greedy (temperature=0) decoding, with a raised token cap so long structured
# # # # #     plans are not truncated. NOTE: greedy can OVER-ELABORATE (verbose plans that
# # # # #     blow past the cap) -- if you still see truncated-JSON crashes, prefer
# # # # #     force_deterministic_sampling() instead."""
# # # # #     import Heuristic_wo_dspy as H
# # # # #     inner = H.llm_complete
# # # # #     cap = max_tokens

# # # # #     def greedy(prompt, images=None, system=None, temperature=0.5,
# # # # #                max_tokens=None, force_json=True):
# # # # #         return inner(prompt, images, system, 0.0, cap, force_json)

# # # # #     H.llm_complete = greedy
# # # # #     probe_backend.set_det_seed(None)
# # # # #     print(f"[adv] forced GREEDY decoding (temp=0), max_tokens={cap}")


# # # # # def force_deterministic_sampling(temperature=0.5, max_tokens=8192, base_seed=1234):
# # # # #     """Keep the pipeline's natural sampling temperature but seed torch per call by
# # # # #     index, so both members of a pair see the SAME randomness and differ ONLY where
# # # # #     the token changes an input. Reproduces run_probed's well-sized plans (no greedy
# # # # #     over-elaboration) while staying deterministic. This is the recommended mode."""
# # # # #     import Heuristic_wo_dspy as H
# # # # #     inner = H.llm_complete
# # # # #     cap = max_tokens
# # # # #     temp = temperature

# # # # #     def det(prompt, images=None, system=None, temperature=0.5,
# # # # #             max_tokens=None, force_json=True):
# # # # #         return inner(prompt, images, system, temp, cap, force_json)

# # # # #     H.llm_complete = det
# # # # #     probe_backend.set_det_seed(base_seed)
# # # # #     print(f"[adv] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
# # # # #           f"max_tokens={cap}")


# # # # # # ---- capture -> plain picklable snapshot ------------------------------------
# # # # # def snapshot_records():
# # # # #     out = []
# # # # #     for r in probe_backend.RECORDS:
# # # # #         out.append({
# # # # #             "output_text": r.get("output_text", ""),
# # # # #             "prompt": r.get("prompt", ""),
# # # # #             "system": r.get("system"),
# # # # #             "force_json": r.get("force_json", True),
# # # # #             "ctx": dict(r.get("ctx", {})),
# # # # #             "call_index": r.get("call_index"),
# # # # #             "labels": copy.deepcopy(r.get("labels", {})),
# # # # #             "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
# # # # #                             for p, d in r.get("activations", {}).items()},
# # # # #         })
# # # # #     return out


# # # # # def save_snapshot(records, path):
# # # # #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# # # # #     with open(path, "wb") as f:
# # # # #         pickle.dump(records, f)
# # # # #     print(f"[adv] saved {len(records)} records -> {path}")


# # # # # def load_snapshot(path):
# # # # #     with open(path, "rb") as f:
# # # # #         return pickle.load(f)


# # # # # # ---- run the real pipeline once ---------------------------------------------
# # # # # def run_once(specs, adv_note, save_dir_root):
# # # # #     """Drive SAGE in-process; inject adv_note into the environment note.
# # # # #     Requires enable_probing() + force_greedy() to have been called already.

# # # # #     IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
# # # # #     so both runs share identical geometry and the ONLY difference is the token in
# # # # #     the prompt text (env_note). This isolates the token's effect. env_note itself
# # # # #     is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
# # # # #     note reaches description, object-choice, plan, and placement prompts."""
# # # # #     probe_backend.reset_records()
# # # # #     probe_backend.reset_call_counter()   # both runs share the per-call seed sequence
# # # # #     import Heuristic_wo_dspy as H
# # # # #     import utils_copy_img
# # # # #     for name, task, assets, dim_x, dim_y in specs:
# # # # #         instruction = task + (adv_note or "")            # only the prompt text carries the note
# # # # #         save_dir = os.path.join(save_dir_root, name)
# # # # #         os.makedirs(save_dir, exist_ok=True)
# # # # #         probe_capture.set_scene(name)
# # # # #         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
# # # # #         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
# # # # #         if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
# # # # #             print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
# # # # #             continue
# # # # #         planner = H.ScenePlanner(save_dir=save_dir)
# # # # #         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
# # # # #                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# # # # #                         orientation=False, scale=False)
# # # # #     return snapshot_records()


# # # # # # ---- alignment + parsing (used by L1 and L2) --------------------------------
# # # # # def placement_key(rec):
# # # # #     """Placement calls carry labels + full ctx; key them by scene/region/iter."""
# # # # #     c = rec["ctx"]
# # # # #     if rec["labels"]:
# # # # #         return (str(c.get("scene_id")), int(c.get("region_id", -1)), int(c.get("iteration", -1)))
# # # # #     return None


# # # # # def align(rec_a, rec_b):
# # # # #     """Return (call-index stream pairs, placement-key pairs, (len_a, len_b))."""
# # # # #     n = min(len(rec_a), len(rec_b))
# # # # #     stream = list(zip(rec_a[:n], rec_b[:n]))
# # # # #     pa = {placement_key(r): r for r in rec_a if placement_key(r)}
# # # # #     pb = {placement_key(r): r for r in rec_b if placement_key(r)}
# # # # #     keys = sorted(k for k in pa if k in pb)
# # # # #     placement_pairs = [(pa[k], pb[k]) for k in keys]
# # # # #     return stream, placement_pairs, (len(rec_a), len(rec_b))


# # # # # def parse_objs(text):
# # # # #     """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
# # # # #     try:
# # # # #         data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
# # # # #     except Exception:
# # # # #         return {}
# # # # #     out = {}
# # # # #     for o in (data if isinstance(data, list) else []):
# # # # #         if isinstance(o, dict) and "label" in o:
# # # # #             p = o.get("position")
# # # # #             if isinstance(p, (list, tuple)) and len(p) >= 2:
# # # # #                 out[str(o["label"])] = (float(p[0]), float(p[1]))
# # # # #     return out


# # # # #!/usr/bin/env python3
# # # # # =============================================================================
# # # # # adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# # # # # -----------------------------------------------------------------------------
# # # # # Everything the levels share:
# # # # #   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
# # # # #                          so BASELINE vs ADVERSARIAL differ ONLY by the token
# # # # #                          (no sampling confound). Wraps the patched llm_complete;
# # # # #                          no edit to probe_backend.
# # # # #   * run_once()         : drive the real SAGE pipeline in-process for a set of
# # # # #                          benchmarks, optionally injecting an adversarial note
# # # # #                          into the environment note.
# # # # #   * snapshot/save/load : persist the captured records so each level is a cheap
# # # # #                          offline re-analysis of the SAME generation.
# # # # #   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
# # # # #
# # # # # Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# # # # # many times as you like over the saved .pkl snapshots.
# # # # # =============================================================================
# # # # from __future__ import annotations

# # # # import copy
# # # # import glob
# # # # import json
# # # # import os
# # # # import pickle
# # # # import re

# # # # import numpy as np

# # # # import probe_backend
# # # # import probe_capture


# # # # # ---- benchmark parsing (same extraction as run_probed_generation) -----------
# # # # def parse_benchmark(path: str):
# # # #     with open(path) as f:
# # # #         data = json.load(f)
# # # #     task = data.get("task_description", "This is an indoor room.")
# # # #     room = data.get("room_dimension", {})
# # # #     dim_x = int(room.get("room_dimension_x", 10))
# # # #     dim_y = int(room.get("room_dimension_y", 10))
# # # #     assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
# # # #     name = os.path.splitext(os.path.basename(path))[0]
# # # #     rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
# # # #     return f"{rtype}_{name}", task, assets, dim_x, dim_y


# # # # def list_specs(benchmark_glob, max_benchmarks=None):
# # # #     specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
# # # #     if max_benchmarks:
# # # #         specs = specs[:max_benchmarks]
# # # #     if not specs:
# # # #         raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
# # # #     return specs


# # # # # ---- determinism: make pairs comparable ------------------------------------
# # # # def force_greedy(max_tokens=8192):
# # # #     """Greedy (temperature=0) decoding, with a raised token cap so long structured
# # # #     plans are not truncated. NOTE: greedy can OVER-ELABORATE (verbose plans that
# # # #     blow past the cap) -- if you still see truncated-JSON crashes, prefer
# # # #     force_deterministic_sampling() instead."""
# # # #     import Heuristic_wo_dspy as H
# # # #     inner = H.llm_complete
# # # #     cap = max_tokens

# # # #     def greedy(prompt, images=None, system=None, temperature=0.5,
# # # #                max_tokens=None, force_json=True):
# # # #         return inner(prompt, images, system, 0.0, cap, force_json)

# # # #     H.llm_complete = greedy
# # # #     probe_backend.set_det_seed(None)
# # # #     print(f"[adv] forced GREEDY decoding (temp=0), max_tokens={cap}")


# # # # def set_repeat_seed(s):
# # # #     """Change the per-call seed base between repeats (after force_* was called)."""
# # # #     probe_backend.set_det_seed(s)


# # # # def force_deterministic_sampling(temperature=0.5, max_tokens=8192, base_seed=1234):
# # # #     """Keep the pipeline's natural sampling temperature but seed torch per call by
# # # #     index, so both members of a pair see the SAME randomness and differ ONLY where
# # # #     the token changes an input. Reproduces run_probed's well-sized plans (no greedy
# # # #     over-elaboration) while staying deterministic. This is the recommended mode."""
# # # #     import Heuristic_wo_dspy as H
# # # #     inner = H.llm_complete
# # # #     cap = max_tokens
# # # #     temp = temperature

# # # #     def det(prompt, images=None, system=None, temperature=0.5,
# # # #             max_tokens=None, force_json=True):
# # # #         return inner(prompt, images, system, temp, cap, force_json)

# # # #     H.llm_complete = det
# # # #     probe_backend.set_det_seed(base_seed)
# # # #     print(f"[adv] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
# # # #           f"max_tokens={cap}")


# # # # # ---- capture -> plain picklable snapshot ------------------------------------
# # # # def snapshot_records():
# # # #     out = []
# # # #     for r in probe_backend.RECORDS:
# # # #         out.append({
# # # #             "output_text": r.get("output_text", ""),
# # # #             "prompt": r.get("prompt", ""),
# # # #             "system": r.get("system"),
# # # #             "force_json": r.get("force_json", True),
# # # #             "ctx": dict(r.get("ctx", {})),
# # # #             "call_index": r.get("call_index"),
# # # #             "labels": copy.deepcopy(r.get("labels", {})),
# # # #             "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
# # # #                             for p, d in r.get("activations", {}).items()},
# # # #         })
# # # #     return out


# # # # def save_snapshot(records, path):
# # # #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# # # #     with open(path, "wb") as f:
# # # #         pickle.dump(records, f)
# # # #     print(f"[adv] saved {len(records)} records -> {path}")


# # # # def load_snapshot(path):
# # # #     with open(path, "rb") as f:
# # # #         return pickle.load(f)


# # # # # ---- run the real pipeline once ---------------------------------------------
# # # # def run_once(specs, adv_note, save_dir_root):
# # # #     """Drive SAGE in-process; inject adv_note into the environment note.
# # # #     Requires enable_probing() + force_greedy() to have been called already.

# # # #     IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
# # # #     so both runs share identical geometry and the ONLY difference is the token in
# # # #     the prompt text (env_note). This isolates the token's effect. env_note itself
# # # #     is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
# # # #     note reaches description, object-choice, plan, and placement prompts."""
# # # #     probe_backend.reset_records()
# # # #     probe_backend.reset_call_counter()   # both runs share the per-call seed sequence
# # # #     import Heuristic_wo_dspy as H
# # # #     import utils_copy_img
# # # #     for name, task, assets, dim_x, dim_y in specs:
# # # #         instruction = task + (adv_note or "")            # only the prompt text carries the note
# # # #         save_dir = os.path.join(save_dir_root, name)
# # # #         os.makedirs(save_dir, exist_ok=True)
# # # #         probe_capture.set_scene(name)
# # # #         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
# # # #         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
# # # #         if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
# # # #             print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
# # # #             continue
# # # #         planner = H.ScenePlanner(save_dir=save_dir)
# # # #         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
# # # #                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# # # #                         orientation=False, scale=False)
# # # #     return snapshot_records()


# # # # # ---- alignment + parsing (used by L1 and L2) --------------------------------
# # # # def placement_key(rec):
# # # #     """Placement calls carry labels + full ctx; key them by scene/region/iter."""
# # # #     c = rec["ctx"]
# # # #     if rec["labels"]:
# # # #         return (str(c.get("scene_id")), int(c.get("region_id", -1)), int(c.get("iteration", -1)))
# # # #     return None


# # # # def align(rec_a, rec_b):
# # # #     """Return (call-index stream pairs, placement-key pairs, (len_a, len_b))."""
# # # #     n = min(len(rec_a), len(rec_b))
# # # #     stream = list(zip(rec_a[:n], rec_b[:n]))
# # # #     pa = {placement_key(r): r for r in rec_a if placement_key(r)}
# # # #     pb = {placement_key(r): r for r in rec_b if placement_key(r)}
# # # #     keys = sorted(k for k in pa if k in pb)
# # # #     placement_pairs = [(pa[k], pb[k]) for k in keys]
# # # #     return stream, placement_pairs, (len(rec_a), len(rec_b))


# # # # def parse_objs(text):
# # # #     """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
# # # #     try:
# # # #         data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
# # # #     except Exception:
# # # #         return {}
# # # #     out = {}
# # # #     for o in (data if isinstance(data, list) else []):
# # # #         if isinstance(o, dict) and "label" in o:
# # # #             p = o.get("position")
# # # #             if isinstance(p, (list, tuple)) and len(p) >= 2:
# # # #                 out[str(o["label"])] = (float(p[0]), float(p[1]))
# # # #     return out


# # # #!/usr/bin/env python3
# # # # =============================================================================
# # # # adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# # # # -----------------------------------------------------------------------------
# # # # Everything the levels share:
# # # #   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
# # # #                          so BASELINE vs ADVERSARIAL differ ONLY by the token
# # # #                          (no sampling confound). Wraps the patched llm_complete;
# # # #                          no edit to probe_backend.
# # # #   * run_once()         : drive the real SAGE pipeline in-process for a set of
# # # #                          benchmarks, optionally injecting an adversarial note
# # # #                          into the environment note.
# # # #   * snapshot/save/load : persist the captured records so each level is a cheap
# # # #                          offline re-analysis of the SAME generation.
# # # #   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
# # # #
# # # # Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# # # # many times as you like over the saved .pkl snapshots.
# # # # =============================================================================
# # # from __future__ import annotations

# # # import copy
# # # import glob
# # # import json
# # # import os
# # # import pickle
# # # import re

# # # import numpy as np

# # # import probe_backend
# # # import probe_capture


# # # # ---- benchmark parsing (same extraction as run_probed_generation) -----------
# # # def parse_benchmark(path: str):
# # #     with open(path) as f:
# # #         data = json.load(f)
# # #     task = data.get("task_description", "This is an indoor room.")
# # #     room = data.get("room_dimension", {})
# # #     dim_x = int(room.get("room_dimension_x", 10))
# # #     dim_y = int(room.get("room_dimension_y", 10))
# # #     assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
# # #     name = os.path.splitext(os.path.basename(path))[0]
# # #     rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
# # #     return f"{rtype}_{name}", task, assets, dim_x, dim_y


# # # def list_specs(benchmark_glob, max_benchmarks=None):
# # #     specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
# # #     if max_benchmarks:
# # #         specs = specs[:max_benchmarks]
# # #     if not specs:
# # #         raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
# # #     return specs


# # # # ---- determinism: make pairs comparable ------------------------------------
# # # def force_greedy(max_tokens=8192):
# # #     """Greedy (temperature=0) decoding, with a raised token cap so long structured
# # #     plans are not truncated. NOTE: greedy can OVER-ELABORATE (verbose plans that
# # #     blow past the cap) -- if you still see truncated-JSON crashes, prefer
# # #     force_deterministic_sampling() instead."""
# # #     import Heuristic_wo_dspy as H
# # #     inner = H.llm_complete
# # #     cap = max_tokens

# # #     def greedy(prompt, images=None, system=None, temperature=0.5,
# # #                max_tokens=None, force_json=True):
# # #         return inner(prompt, images, system, 0.0, cap, force_json)

# # #     H.llm_complete = greedy
# # #     probe_backend.set_det_seed(None)
# # #     print(f"[adv] forced GREEDY decoding (temp=0), max_tokens={cap}")


# # # _RUN_SEED = None


# # # def seed_everything(s):
# # #     """Seed EVERY RNG the pipeline touches.

# # #     CRITICAL: Heuristic_wo_dspy calls utils_copy_img.generate_random_seed() at each
# # #     stage and interpolates the result into the PROMPT TEXT ("Random seed: {seed}").
# # #     That RNG is Python's global `random` (and possibly numpy) -- NOT torch. If it is
# # #     unseeded, every run builds DIFFERENT PROMPTS, so torch seeding alone cannot make
# # #     runs reproducible. Seeding here makes generate_random_seed() emit the same
# # #     sequence in both members of a pair, so the prompts differ ONLY by the note."""
# # #     import random
# # #     random.seed(s)
# # #     np.random.seed(s % (2 ** 32))
# # #     try:
# # #         import torch
# # #         torch.manual_seed(s)
# # #         if torch.cuda.is_available():
# # #             torch.cuda.manual_seed_all(s)
# # #     except Exception:
# # #         pass


# # # def set_repeat_seed(s):
# # #     """Change the seed base between repeats (after force_* was called)."""
# # #     global _RUN_SEED
# # #     _RUN_SEED = s
# # #     probe_backend.set_det_seed(s)


# # # def force_deterministic_sampling(temperature=0.5, max_tokens=8192, base_seed=1234):
# # #     """Keep the pipeline's natural sampling temperature but seed torch per call by
# # #     index, so both members of a pair see the SAME randomness and differ ONLY where
# # #     the token changes an input. Reproduces run_probed's well-sized plans (no greedy
# # #     over-elaboration) while staying deterministic. This is the recommended mode."""
# # #     import Heuristic_wo_dspy as H
# # #     inner = H.llm_complete
# # #     cap = max_tokens
# # #     temp = temperature

# # #     def det(prompt, images=None, system=None, temperature=0.5,
# # #             max_tokens=None, force_json=True):
# # #         return inner(prompt, images, system, temp, cap, force_json)

# # #     H.llm_complete = det
# # #     probe_backend.set_det_seed(base_seed)
# # #     print(f"[adv] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
# # #           f"max_tokens={cap}")


# # # # ---- capture -> plain picklable snapshot ------------------------------------
# # # def snapshot_records():
# # #     out = []
# # #     for r in probe_backend.RECORDS:
# # #         out.append({
# # #             "output_text": r.get("output_text", ""),
# # #             "prompt": r.get("prompt", ""),
# # #             "system": r.get("system"),
# # #             "force_json": r.get("force_json", True),
# # #             "ctx": dict(r.get("ctx", {})),
# # #             "call_index": r.get("call_index"),
# # #             "labels": copy.deepcopy(r.get("labels", {})),
# # #             "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
# # #                             for p, d in r.get("activations", {}).items()},
# # #         })
# # #     return out


# # # def save_snapshot(records, path):
# # #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# # #     with open(path, "wb") as f:
# # #         pickle.dump(records, f)
# # #     print(f"[adv] saved {len(records)} records -> {path}")


# # # def load_snapshot(path):
# # #     with open(path, "rb") as f:
# # #         return pickle.load(f)


# # # # ---- run the real pipeline once ---------------------------------------------
# # # def run_once(specs, adv_note, save_dir_root):
# # #     """Drive SAGE in-process; inject adv_note into the environment note.
# # #     Requires enable_probing() + force_greedy() to have been called already.

# # #     IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
# # #     so both runs share identical geometry and the ONLY difference is the token in
# # #     the prompt text (env_note). This isolates the token's effect. env_note itself
# # #     is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
# # #     note reaches description, object-choice, plan, and placement prompts."""
# # #     probe_backend.reset_records()
# # #     probe_backend.reset_call_counter()   # both runs share the per-call seed sequence
# # #     if _RUN_SEED is not None:
# # #         seed_everything(_RUN_SEED)       # makes generate_random_seed() reproducible
# # #     import Heuristic_wo_dspy as H
# # #     import utils_copy_img
# # #     for name, task, assets, dim_x, dim_y in specs:
# # #         instruction = task + (adv_note or "")            # only the prompt text carries the note
# # #         save_dir = os.path.join(save_dir_root, name)
# # #         os.makedirs(save_dir, exist_ok=True)
# # #         probe_capture.set_scene(name)
# # #         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
# # #         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
# # #         if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
# # #             print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
# # #             continue
# # #         planner = H.ScenePlanner(save_dir=save_dir)
# # #         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
# # #                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# # #                         orientation=False, scale=False)
# # #     return snapshot_records()


# # # # ---- alignment + parsing (used by L1 and L2) --------------------------------
# # # def placement_key(rec):
# # #     """Placement calls carry labels + full ctx; key them by scene/region/iter."""
# # #     c = rec["ctx"]
# # #     if rec["labels"]:
# # #         return (str(c.get("scene_id")), int(c.get("region_id", -1)), int(c.get("iteration", -1)))
# # #     return None


# # # def align(rec_a, rec_b):
# # #     """Return (call-index stream pairs, placement-key pairs, (len_a, len_b))."""
# # #     n = min(len(rec_a), len(rec_b))
# # #     stream = list(zip(rec_a[:n], rec_b[:n]))
# # #     pa = {placement_key(r): r for r in rec_a if placement_key(r)}
# # #     pb = {placement_key(r): r for r in rec_b if placement_key(r)}
# # #     keys = sorted(k for k in pa if k in pb)
# # #     placement_pairs = [(pa[k], pb[k]) for k in keys]
# # #     return stream, placement_pairs, (len(rec_a), len(rec_b))


# # # def parse_objs(text):
# # #     """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
# # #     try:
# # #         data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
# # #     except Exception:
# # #         return {}
# # #     out = {}
# # #     for o in (data if isinstance(data, list) else []):
# # #         if isinstance(o, dict) and "label" in o:
# # #             p = o.get("position")
# # #             if isinstance(p, (list, tuple)) and len(p) >= 2:
# # #                 out[str(o["label"])] = (float(p[0]), float(p[1]))
# # #     return out


# # #!/usr/bin/env python3
# # # =============================================================================
# # # adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# # # -----------------------------------------------------------------------------
# # # Everything the levels share:
# # #   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
# # #                          so BASELINE vs ADVERSARIAL differ ONLY by the token
# # #                          (no sampling confound). Wraps the patched llm_complete;
# # #                          no edit to probe_backend.
# # #   * run_once()         : drive the real SAGE pipeline in-process for a set of
# # #                          benchmarks, optionally injecting an adversarial note
# # #                          into the environment note.
# # #   * snapshot/save/load : persist the captured records so each level is a cheap
# # #                          offline re-analysis of the SAME generation.
# # #   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
# # #
# # # Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# # # many times as you like over the saved .pkl snapshots.
# # # =============================================================================
# # from __future__ import annotations

# # import copy
# # import glob
# # import json
# # import os
# # import pickle
# # import re

# # import numpy as np

# # import probe_backend
# # import probe_capture


# # # ---- benchmark parsing (same extraction as run_probed_generation) -----------
# # def parse_benchmark(path: str):
# #     with open(path) as f:
# #         data = json.load(f)
# #     task = data.get("task_description", "This is an indoor room.")
# #     room = data.get("room_dimension", {})
# #     dim_x = int(room.get("room_dimension_x", 10))
# #     dim_y = int(room.get("room_dimension_y", 10))
# #     assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
# #     name = os.path.splitext(os.path.basename(path))[0]
# #     rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
# #     return f"{rtype}_{name}", task, assets, dim_x, dim_y


# # def list_specs(benchmark_glob, max_benchmarks=None):
# #     specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
# #     if max_benchmarks:
# #         specs = specs[:max_benchmarks]
# #     if not specs:
# #         raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
# #     return specs


# # # ---- determinism: make pairs comparable ------------------------------------
# # def force_greedy(max_tokens=8192):
# #     """Greedy (temperature=0) decoding, with a raised token cap so long structured
# #     plans are not truncated. NOTE: greedy can OVER-ELABORATE (verbose plans that
# #     blow past the cap) -- if you still see truncated-JSON crashes, prefer
# #     force_deterministic_sampling() instead."""
# #     import Heuristic_wo_dspy as H
# #     inner = H.llm_complete
# #     cap = max_tokens

# #     def greedy(prompt, images=None, system=None, temperature=0.5,
# #                max_tokens=None, force_json=True):
# #         return inner(prompt, images, system, 0.0, cap, force_json)

# #     H.llm_complete = greedy
# #     probe_backend.set_det_seed(None)
# #     print(f"[adv] forced GREEDY decoding (temp=0), max_tokens={cap}")


# # _RUN_SEED = None
# # _SEEDCTR = {"n": 0}


# # def force_deterministic_seed_text():
# #     """Make the 'Random seed: N' text in prompts immune to RNG-stream desync.

# #     utils_copy_img.generate_random_seed() is random.randint() on the GLOBAL python
# #     RNG, and its value goes into the prompt. Anything else that draws from that same
# #     global stream (e.g. plot_utils allocating random label colours on the FIRST
# #     visualize() call, then caching them) shifts the stream for that run only -- so
# #     run 0 gets a different seed sequence than runs 1+, even with identical seeding.
# #     That is exactly the 'run 0 diverges at call 4, runs 1-4 identical' pattern.

# #     Fix: replace it with a COUNTER-based generator derived from (run_seed, call#).
# #     Diversity across best-of-N candidates is preserved (the counter increments), but
# #     the sequence no longer depends on the stream position, so every run is identical."""
# #     import utils_copy_img
# #     if getattr(utils_copy_img.generate_random_seed, "_det_patched", False):
# #         return

# #     def det_seed():
# #         _SEEDCTR["n"] += 1
# #         base = _RUN_SEED if _RUN_SEED is not None else 0
# #         return (base * 7919 + _SEEDCTR["n"] * 104729) % 10000001

# #     det_seed._det_patched = True
# #     utils_copy_img.generate_random_seed = det_seed
# #     print("[adv] patched generate_random_seed -> deterministic counter-based "
# #           "(immune to global-RNG stream desync)")


# # def seed_everything(s):
# #     """Seed EVERY RNG the pipeline touches.

# #     CRITICAL: Heuristic_wo_dspy calls utils_copy_img.generate_random_seed() at each
# #     stage and interpolates the result into the PROMPT TEXT ("Random seed: {seed}").
# #     That RNG is Python's global `random` (and possibly numpy) -- NOT torch. If it is
# #     unseeded, every run builds DIFFERENT PROMPTS, so torch seeding alone cannot make
# #     runs reproducible. Seeding here makes generate_random_seed() emit the same
# #     sequence in both members of a pair, so the prompts differ ONLY by the note."""
# #     import random
# #     random.seed(s)
# #     np.random.seed(s % (2 ** 32))
# #     try:
# #         import torch
# #         torch.manual_seed(s)
# #         if torch.cuda.is_available():
# #             torch.cuda.manual_seed_all(s)
# #     except Exception:
# #         pass


# # def set_repeat_seed(s):
# #     """Change the seed base between repeats (after force_* was called)."""
# #     global _RUN_SEED
# #     _RUN_SEED = s
# #     probe_backend.set_det_seed(s)


# # def force_deterministic_sampling(temperature=0.5, max_tokens=8192, base_seed=1234):
# #     """Keep the pipeline's natural sampling temperature but seed torch per call by
# #     index, so both members of a pair see the SAME randomness and differ ONLY where
# #     the token changes an input. Reproduces run_probed's well-sized plans (no greedy
# #     over-elaboration) while staying deterministic. This is the recommended mode."""
# #     import Heuristic_wo_dspy as H
# #     inner = H.llm_complete
# #     cap = max_tokens
# #     temp = temperature

# #     def det(prompt, images=None, system=None, temperature=0.5,
# #             max_tokens=None, force_json=True):
# #         return inner(prompt, images, system, temp, cap, force_json)

# #     H.llm_complete = det
# #     probe_backend.set_det_seed(base_seed)
# #     print(f"[adv] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
# #           f"max_tokens={cap}")


# # # ---- capture -> plain picklable snapshot ------------------------------------
# # def snapshot_records():
# #     out = []
# #     for r in probe_backend.RECORDS:
# #         out.append({
# #             "output_text": r.get("output_text", ""),
# #             "prompt": r.get("prompt", ""),
# #             "system": r.get("system"),
# #             "force_json": r.get("force_json", True),
# #             "ctx": dict(r.get("ctx", {})),
# #             "call_index": r.get("call_index"),
# #             "labels": copy.deepcopy(r.get("labels", {})),
# #             "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
# #                             for p, d in r.get("activations", {}).items()},
# #         })
# #     return out


# # def save_snapshot(records, path):
# #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# #     with open(path, "wb") as f:
# #         pickle.dump(records, f)
# #     print(f"[adv] saved {len(records)} records -> {path}")


# # def load_snapshot(path):
# #     with open(path, "rb") as f:
# #         return pickle.load(f)


# # # ---- run the real pipeline once ---------------------------------------------
# # def run_once(specs, adv_note, save_dir_root):
# #     """Drive SAGE in-process; inject adv_note into the environment note.
# #     Requires enable_probing() + force_greedy() to have been called already.

# #     IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
# #     so both runs share identical geometry and the ONLY difference is the token in
# #     the prompt text (env_note). This isolates the token's effect. env_note itself
# #     is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
# #     note reaches description, object-choice, plan, and placement prompts."""
# #     probe_backend.reset_records()
# #     probe_backend.reset_call_counter()   # both runs share the per-call seed sequence
# #     force_deterministic_seed_text()      # immune to RNG-stream desync (run-0 effect)
# #     _SEEDCTR["n"] = 0                    # identical seed-text sequence every run
# #     if _RUN_SEED is not None:
# #         seed_everything(_RUN_SEED)       # makes generate_random_seed() reproducible
# #     import Heuristic_wo_dspy as H
# #     import utils_copy_img
# #     for name, task, assets, dim_x, dim_y in specs:
# #         instruction = task + (adv_note or "")            # only the prompt text carries the note
# #         save_dir = os.path.join(save_dir_root, name)
# #         os.makedirs(save_dir, exist_ok=True)
# #         probe_capture.set_scene(name)
# #         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
# #         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
# #         if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
# #             print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
# #             continue
# #         planner = H.ScenePlanner(save_dir=save_dir)
# #         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
# #                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
# #                         orientation=False, scale=False)
# #     return snapshot_records()


# # # ---- alignment + parsing (used by L1 and L2) --------------------------------
# # def placement_key(rec):
# #     """Placement calls carry labels + full ctx; key them by scene/region/iter."""
# #     c = rec["ctx"]
# #     if rec["labels"]:
# #         return (str(c.get("scene_id")), int(c.get("region_id", -1)), int(c.get("iteration", -1)))
# #     return None


# # def align(rec_a, rec_b):
# #     """Return (call-index stream pairs, placement-key pairs, (len_a, len_b))."""
# #     n = min(len(rec_a), len(rec_b))
# #     stream = list(zip(rec_a[:n], rec_b[:n]))
# #     pa = {placement_key(r): r for r in rec_a if placement_key(r)}
# #     pb = {placement_key(r): r for r in rec_b if placement_key(r)}
# #     keys = sorted(k for k in pa if k in pb)
# #     placement_pairs = [(pa[k], pb[k]) for k in keys]
# #     return stream, placement_pairs, (len(rec_a), len(rec_b))


# # def parse_objs(text):
# #     """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
# #     try:
# #         data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
# #     except Exception:
# #         return {}
# #     out = {}
# #     for o in (data if isinstance(data, list) else []):
# #         if isinstance(o, dict) and "label" in o:
# #             p = o.get("position")
# #             if isinstance(p, (list, tuple)) and len(p) >= 2:
# #                 out[str(o["label"])] = (float(p[0]), float(p[1]))
# #     return out


# #!/usr/bin/env python3
# # =============================================================================
# # adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# # -----------------------------------------------------------------------------
# # Everything the levels share:
# #   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
# #                          so BASELINE vs ADVERSARIAL differ ONLY by the token
# #                          (no sampling confound). Wraps the patched llm_complete;
# #                          no edit to probe_backend.
# #   * run_once()         : drive the real SAGE pipeline in-process for a set of
# #                          benchmarks, optionally injecting an adversarial note
# #                          into the environment note.
# #   * snapshot/save/load : persist the captured records so each level is a cheap
# #                          offline re-analysis of the SAME generation.
# #   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
# #
# # Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# # many times as you like over the saved .pkl snapshots.
# # =============================================================================
# from __future__ import annotations

# import copy
# import glob
# import json
# import os
# import pickle
# import re

# import numpy as np

# import probe_backend
# import probe_capture


# # ---- benchmark parsing (same extraction as run_probed_generation) -----------
# def parse_benchmark(path: str):
#     with open(path) as f:
#         data = json.load(f)
#     task = data.get("task_description", "This is an indoor room.")
#     room = data.get("room_dimension", {})
#     dim_x = int(room.get("room_dimension_x", 10))
#     dim_y = int(room.get("room_dimension_y", 10))
#     assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
#     name = os.path.splitext(os.path.basename(path))[0]
#     rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
#     return f"{rtype}_{name}", task, assets, dim_x, dim_y


# def list_specs(benchmark_glob, max_benchmarks=None):
#     specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
#     if max_benchmarks:
#         specs = specs[:max_benchmarks]
#     if not specs:
#         raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
#     return specs


# # ---- determinism: make pairs comparable ------------------------------------
# def force_greedy(max_tokens=8192):
#     """Greedy (temperature=0) decoding, with a raised token cap so long structured
#     plans are not truncated. NOTE: greedy can OVER-ELABORATE (verbose plans that
#     blow past the cap) -- if you still see truncated-JSON crashes, prefer
#     force_deterministic_sampling() instead."""
#     import Heuristic_wo_dspy as H
#     inner = H.llm_complete
#     cap = max_tokens

#     def greedy(prompt, images=None, system=None, temperature=0.5,
#                max_tokens=None, force_json=True):
#         return inner(prompt, images, system, 0.0, cap, force_json)

#     H.llm_complete = greedy
#     probe_backend.set_det_seed(None)
#     print(f"[adv] forced GREEDY decoding (temp=0), max_tokens={cap}")


# _RUN_SEED = None
# _SEEDCTR = {"n": 0}


# def force_deterministic_seed_text():
#     """Make the 'Random seed: N' text in prompts immune to RNG-stream desync.

#     utils_copy_img.generate_random_seed() is random.randint() on the GLOBAL python
#     RNG, and its value goes into the prompt. Anything else that draws from that same
#     global stream (e.g. plot_utils allocating random label colours on the FIRST
#     visualize() call, then caching them) shifts the stream for that run only -- so
#     run 0 gets a different seed sequence than runs 1+, even with identical seeding.
#     That is exactly the 'run 0 diverges at call 4, runs 1-4 identical' pattern.

#     Fix: replace it with a COUNTER-based generator derived from (run_seed, call#).
#     Diversity across best-of-N candidates is preserved (the counter increments), but
#     the sequence no longer depends on the stream position, so every run is identical."""
#     import utils_copy_img
#     if getattr(utils_copy_img.generate_random_seed, "_det_patched", False):
#         return

#     def det_seed():
#         _SEEDCTR["n"] += 1
#         base = _RUN_SEED if _RUN_SEED is not None else 0
#         return (base * 7919 + _SEEDCTR["n"] * 104729) % 10000001

#     det_seed._det_patched = True
#     utils_copy_img.generate_random_seed = det_seed
#     print("[adv] patched generate_random_seed -> deterministic counter-based "
#           "(immune to global-RNG stream desync)")


# def seed_everything(s):
#     """Seed EVERY RNG the pipeline touches.

#     CRITICAL: Heuristic_wo_dspy calls utils_copy_img.generate_random_seed() at each
#     stage and interpolates the result into the PROMPT TEXT ("Random seed: {seed}").
#     That RNG is Python's global `random` (and possibly numpy) -- NOT torch. If it is
#     unseeded, every run builds DIFFERENT PROMPTS, so torch seeding alone cannot make
#     runs reproducible. Seeding here makes generate_random_seed() emit the same
#     sequence in both members of a pair, so the prompts differ ONLY by the note."""
#     import random
#     random.seed(s)
#     np.random.seed(s % (2 ** 32))
#     try:
#         import torch
#         torch.manual_seed(s)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(s)
#     except Exception:
#         pass


# def set_repeat_seed(s):
#     """Change the seed base between repeats (after force_* was called)."""
#     global _RUN_SEED
#     _RUN_SEED = s
#     probe_backend.set_det_seed(s)


# def force_deterministic_sampling(temperature=0.5, max_tokens=8192, base_seed=1234):
#     """Keep the pipeline's natural sampling temperature but seed torch per call by
#     index, so both members of a pair see the SAME randomness and differ ONLY where
#     the token changes an input. Reproduces run_probed's well-sized plans (no greedy
#     over-elaboration) while staying deterministic. This is the recommended mode."""
#     import Heuristic_wo_dspy as H
#     inner = H.llm_complete
#     cap = max_tokens
#     temp = temperature

#     def det(prompt, images=None, system=None, temperature=0.5,
#             max_tokens=None, force_json=True):
#         return inner(prompt, images, system, temp, cap, force_json)

#     H.llm_complete = det
#     probe_backend.set_det_seed(base_seed)
#     print(f"[adv] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
#           f"max_tokens={cap}")


# # ---- capture -> plain picklable snapshot ------------------------------------
# def snapshot_records():
#     out = []
#     for r in probe_backend.RECORDS:
#         out.append({
#             "output_text": r.get("output_text", ""),
#             "prompt": r.get("prompt", ""),
#             "system": r.get("system"),
#             "force_json": r.get("force_json", True),
#             "ctx": dict(r.get("ctx", {})),
#             "call_index": r.get("call_index"),
#             "labels": copy.deepcopy(r.get("labels", {})),
#             "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
#                             for p, d in r.get("activations", {}).items()},
#         })
#     return out


# def save_snapshot(records, path):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     with open(path, "wb") as f:
#         pickle.dump(records, f)
#     print(f"[adv] saved {len(records)} records -> {path}")


# def load_snapshot(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# # ---- run the real pipeline once ---------------------------------------------
# def run_once(specs, adv_note, save_dir_root):
#     """Drive SAGE in-process; inject adv_note into the environment note.
#     Requires enable_probing() + force_greedy() to have been called already.

#     IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
#     so both runs share identical geometry and the ONLY difference is the token in
#     the prompt text (env_note). This isolates the token's effect. env_note itself
#     is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
#     note reaches description, object-choice, plan, and placement prompts."""
#     probe_backend.reset_records()
#     probe_backend.reset_call_counter()   # both runs share the per-call seed sequence
#     force_deterministic_seed_text()      # immune to RNG-stream desync (run-0 effect)
#     _SEEDCTR["n"] = 0                    # identical seed-text sequence every run
#     if _RUN_SEED is not None:
#         seed_everything(_RUN_SEED)       # makes generate_random_seed() reproducible
#     import Heuristic_wo_dspy as H
#     import utils_copy_img
#     for name, task, assets, dim_x, dim_y in specs:
#         instruction = task + (adv_note or "")            # only the prompt text carries the note
#         save_dir = os.path.join(save_dir_root, name)
#         os.makedirs(save_dir, exist_ok=True)
#         probe_capture.set_scene(name)
#         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
#         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
#         if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
#             print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
#             continue
#         planner = H.ScenePlanner(save_dir=save_dir)
#         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
#                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
#                         orientation=False, scale=False)
#     return snapshot_records()


# # ---- alignment + parsing (used by L1 and L2) --------------------------------
# def placement_key(rec):
#     """Placement calls carry labels + full ctx; key them by scene/region/iter."""
#     c = rec["ctx"]
#     if rec["labels"]:
#         return (str(c.get("scene_id")), int(c.get("region_id", -1)), int(c.get("iteration", -1)))
#     return None


# def align(rec_a, rec_b):
#     """Return (call-index stream pairs, placement-key pairs, (len_a, len_b))."""
#     n = min(len(rec_a), len(rec_b))
#     stream = list(zip(rec_a[:n], rec_b[:n]))
#     pa = {placement_key(r): r for r in rec_a if placement_key(r)}
#     pb = {placement_key(r): r for r in rec_b if placement_key(r)}
#     keys = sorted(k for k in pa if k in pb)
#     placement_pairs = [(pa[k], pb[k]) for k in keys]
#     return stream, placement_pairs, (len(rec_a), len(rec_b))


# def parse_objs(text):
#     """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
#     try:
#         data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
#     except Exception:
#         return {}
#     out = {}
#     for o in (data if isinstance(data, list) else []):
#         if isinstance(o, dict) and "label" in o:
#             p = o.get("position")
#             if isinstance(p, (list, tuple)) and len(p) >= 2:
#                 out[str(o["label"])] = (float(p[0]), float(p[1]))
#     return out

#!/usr/bin/env python3
# =============================================================================
# adv_common.py  --  shared foundation for the level-by-level adversarial analysis
# -----------------------------------------------------------------------------
# Everything the levels share:
#   * force_greedy()     : make every pipeline LLM call decode at temperature 0,
#                          so BASELINE vs ADVERSARIAL differ ONLY by the token
#                          (no sampling confound). Wraps the patched llm_complete;
#                          no edit to probe_backend.
#   * run_once()         : drive the real SAGE pipeline in-process for a set of
#                          benchmarks, optionally injecting an adversarial note
#                          into the environment note.
#   * snapshot/save/load : persist the captured records so each level is a cheap
#                          offline re-analysis of the SAME generation.
#   * align/_parse_objs  : pair BASELINE and ADVERSARIAL records for diffing.
#
# Generation is expensive; run it ONCE (adv_generate_pairs.py) then run L1/L2 as
# many times as you like over the saved .pkl snapshots.
# =============================================================================
from __future__ import annotations

import copy
import glob
import json
import os
import pickle
import re

import numpy as np

import probe_backend
import probe_capture


# ---- benchmark parsing (same extraction as run_probed_generation) -----------
def parse_benchmark(path: str):
    with open(path) as f:
        data = json.load(f)
    task = data.get("task_description", "This is an indoor room.")
    room = data.get("room_dimension", {})
    dim_x = int(room.get("room_dimension_x", 10))
    dim_y = int(room.get("room_dimension_y", 10))
    assets = ", ".join(f"{v} {k}" for k, v in data.get("assets", {}).items())
    name = os.path.splitext(os.path.basename(path))[0]
    rtype = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return f"{rtype}_{name}", task, assets, dim_x, dim_y


def list_specs(benchmark_glob, max_benchmarks=None):
    specs = [parse_benchmark(p) for p in sorted(glob.glob(benchmark_glob, recursive=True))]
    if max_benchmarks:
        specs = specs[:max_benchmarks]
    if not specs:
        raise SystemExit(f"no benchmarks matched {benchmark_glob!r}")
    return specs


# ---- determinism: make pairs comparable ------------------------------------
def force_greedy(max_tokens=8192):
    """Greedy (temperature=0) decoding, with a raised token cap so long structured
    plans are not truncated. NOTE: greedy can OVER-ELABORATE (verbose plans that
    blow past the cap) -- if you still see truncated-JSON crashes, prefer
    force_deterministic_sampling() instead."""
    import Heuristic_wo_dspy as H
    inner = H.llm_complete
    cap = max_tokens

    def greedy(prompt, images=None, system=None, temperature=0.5,
               max_tokens=None, force_json=True):
        return inner(prompt, images, system, 0.0, cap, force_json)

    H.llm_complete = greedy
    probe_backend.set_det_seed(None)
    print(f"[adv] forced GREEDY decoding (temp=0), max_tokens={cap}")


_RUN_SEED = None
_SEEDCTR = {"n": 0}


def force_deterministic_seed_text():
    """Make the 'Random seed: N' text in prompts immune to RNG-stream desync.

    utils_copy_img.generate_random_seed() is random.randint() on the GLOBAL python
    RNG, and its value goes into the prompt. Anything else that draws from that same
    global stream (e.g. plot_utils allocating random label colours on the FIRST
    visualize() call, then caching them) shifts the stream for that run only -- so
    run 0 gets a different seed sequence than runs 1+, even with identical seeding.
    That is exactly the 'run 0 diverges at call 4, runs 1-4 identical' pattern.

    Fix: replace it with a COUNTER-based generator derived from (run_seed, call#).
    Diversity across best-of-N candidates is preserved (the counter increments), but
    the sequence no longer depends on the stream position, so every run is identical."""
    import utils_copy_img
    if getattr(utils_copy_img.generate_random_seed, "_det_patched", False):
        return

    def det_seed():
        _SEEDCTR["n"] += 1
        base = _RUN_SEED if _RUN_SEED is not None else 0
        return (base * 7919 + _SEEDCTR["n"] * 104729) % 10000001

    det_seed._det_patched = True
    utils_copy_img.generate_random_seed = det_seed
    print("[adv] patched generate_random_seed -> deterministic counter-based "
          "(immune to global-RNG stream desync)")


def seed_everything(s):
    """Seed EVERY RNG the pipeline touches.

    CRITICAL: Heuristic_wo_dspy calls utils_copy_img.generate_random_seed() at each
    stage and interpolates the result into the PROMPT TEXT ("Random seed: {seed}").
    That RNG is Python's global `random` (and possibly numpy) -- NOT torch. If it is
    unseeded, every run builds DIFFERENT PROMPTS, so torch seeding alone cannot make
    runs reproducible. Seeding here makes generate_random_seed() emit the same
    sequence in both members of a pair, so the prompts differ ONLY by the note."""
    import random
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def set_repeat_seed(s):
    """Change the seed base between repeats (after force_* was called)."""
    global _RUN_SEED
    _RUN_SEED = s
    probe_backend.set_det_seed(s)


def force_deterministic_sampling(temperature=0.5, max_tokens=8192, base_seed=1234):
    """Keep the pipeline's natural sampling temperature but seed torch per call by
    index, so both members of a pair see the SAME randomness and differ ONLY where
    the token changes an input. Reproduces run_probed's well-sized plans (no greedy
    over-elaboration) while staying deterministic. This is the recommended mode."""
    import Heuristic_wo_dspy as H
    inner = H.llm_complete
    cap = max_tokens
    temp = temperature

    def det(prompt, images=None, system=None, temperature=0.5,
            max_tokens=None, force_json=True):
        return inner(prompt, images, system, temp, cap, force_json)

    H.llm_complete = det
    probe_backend.set_det_seed(base_seed)
    print(f"[adv] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
          f"max_tokens={cap}")


# ---- capture -> plain picklable snapshot ------------------------------------
def snapshot_records():
    out = []
    for r in probe_backend.RECORDS:
        out.append({
            "output_text": r.get("output_text", ""),
            "prompt": r.get("prompt", ""),
            "system": r.get("system"),
            "force_json": r.get("force_json", True),
            "ctx": dict(r.get("ctx", {})),
            "call_index": r.get("call_index"),
            "labels": copy.deepcopy(r.get("labels", {})),
            "activations": {p: {int(li): np.asarray(v) for li, v in d.items()}
                            for p, d in r.get("activations", {}).items()},
        })
    return out


def save_snapshot(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f)
    print(f"[adv] saved {len(records)} records -> {path}")


def load_snapshot(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- run the real pipeline once ---------------------------------------------
def run_once(specs, adv_note, save_dir_root):
    """Drive SAGE in-process; inject adv_note into the environment note.
    Requires enable_probing() + force_greedy() to have been called already.

    IMPORTANT: position/dx/dy are derived from the BASELINE task (NOT the note),
    so both runs share identical geometry and the ONLY difference is the token in
    the prompt text (env_note). This isolates the token's effect. env_note itself
    is built inside ScenePlanner.forward as f"Instruction: {instruction}", so the
    note reaches description, object-choice, plan, and placement prompts."""
    probe_backend.reset_records()
    probe_backend.reset_call_counter()   # both runs share the per-call seed sequence
    force_deterministic_seed_text()      # immune to RNG-stream desync (run-0 effect)
    _SEEDCTR["n"] = 0                    # identical seed-text sequence every run
    if _RUN_SEED is not None:
        seed_everything(_RUN_SEED)       # makes generate_random_seed() reproducible
    import Heuristic_wo_dspy as H
    import utils_copy_img
    for name, task, assets, dim_x, dim_y in specs:
        instruction = task + (adv_note or "")            # only the prompt text carries the note
        save_dir = os.path.join(save_dir_root, name)
        os.makedirs(save_dir, exist_ok=True)
        probe_capture.set_scene(name)
        dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
        pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
        if utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
            print(f"[skip] {name}: scenegraph exists (use a fresh --save-root)")
            continue
        planner = H.ScenePlanner(save_dir=save_dir)
        planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
                        objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
                        orientation=False, scale=False)
    return snapshot_records()


# ---- alignment + parsing (used by L1 and L2) --------------------------------
def placement_key(rec, ignore_region=False):
    """Placement calls carry labels + full ctx; key them by scene/region/iter.

    ignore_region=True drops region_id. Needed because probe_capture's region
    counter was (in earlier runs) a process-global that increments across runs, so
    the SAME region of the SAME scene gets a different region_id in baseline vs
    adversarial. For the single-region baseline, (scene, iteration) is unique and
    is a safe key."""
    c = rec["ctx"]
    if rec["labels"]:
        sid = str(c.get("scene_id"))
        it = int(c.get("iteration") or -1)
        if ignore_region:
            return (sid, it)
        return (sid, int(c.get("region_id") if c.get("region_id") is not None else -1), it)
    return None


def align(rec_a, rec_b, ignore_region=None):
    """Return (call-index stream pairs, placement-key pairs, (len_a, len_b)).

    ignore_region=None -> try strict keys, and if NOTHING overlaps, automatically
    retry without region_id and warn. That rescues snapshots affected by the
    non-resetting region counter without re-generating."""
    n = min(len(rec_a), len(rec_b))
    stream = list(zip(rec_a[:n], rec_b[:n]))

    def build(ig):
        pa = {placement_key(r, ig): r for r in rec_a if placement_key(r, ig)}
        pb = {placement_key(r, ig): r for r in rec_b if placement_key(r, ig)}
        keys = sorted(k for k in pa if k in pb)
        return pa, pb, keys

    if ignore_region is None:
        pa, pb, keys = build(False)
        if not keys:
            pa2, pb2, keys2 = build(True)
            if keys2:
                print("[align] WARNING: no overlap on (scene, region, iteration) -- the "
                      "region counter differs between runs (known probe_capture bug: the "
                      "counter is process-global and never resets). Falling back to "
                      f"(scene, iteration); {len(keys2)} pair(s) aligned.")
                pa, pb, keys = pa2, pb2, keys2
    else:
        pa, pb, keys = build(ignore_region)

    placement_pairs = [(pa[k], pb[k]) for k in keys]
    return stream, placement_pairs, (len(rec_a), len(rec_b))


def parse_objs(text):
    """Best-effort parse of a delta list -> {label: (x, y) centroid}."""
    try:
        data = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    except Exception:
        return {}
    out = {}
    for o in (data if isinstance(data, list) else []):
        if isinstance(o, dict) and "label" in o:
            p = o.get("position")
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out[str(o["label"])] = (float(p[0]), float(p[1]))
    return out