# #!/usr/bin/env python3
# # =============================================================================
# # probe_capture.py
# # -----------------------------------------------------------------------------
# # Wires probe_backend into the SAGE generator IN-PROCESS and builds Othello-style
# # ground truth from the generation loop itself. Two monkeypatches:
# #   1. Heuristic_wo_dspy.llm_complete   -> probe_backend.probed_complete
# #      (every model call now runs locally and records activations)
# #   2. Heuristic_wo_dspy.ObjectPlacementManager -> ProbedObjectPlacementManager
# #      (each best-of-N placement candidate is tagged with scene/region/iteration
# #       and paired with geometric GT computed here)
# #
# # Ground truth per placement candidate (external geometry = "the rules"):
# #   * occupancy       : R x C occupancy grid of (prior committed + this candidate)
# #   * occupancy_prior : R x C grid of prior committed only (accumulated state)
# #   * contested       : per-cell "covered by >=2 objects" (overlap footprint)
# #   * legality        : 1 if NO new object overlaps a PREVIOUS placement, else 0.
# #                       "Previous" = objects committed before this call, in output
# #                       order union earlier objects in the same candidate.
# #   * out_of_bounds   : any new object leaves the room extent
# # Two probe token positions are stored (prompt_end, delta_end) -- see backend.
# #
# # Run your generation exactly as usual, but IN THE SAME PROCESS, e.g.:
# #     import probe_capture
# #     probe_capture.enable_probing(model_path="Qwen/Qwen3.5-27B",
# #                                  grid_rows=5, grid_cols=5, num_probe_layers=8)
# #     import Heuristic_wo_dspy as H          # import AFTER enable_probing
# #     for bench in benchmarks:
# #         probe_capture.set_scene(bench.name)
# #         planner = H.ScenePlanner(save_dir)               # picks up probed manager
# #         planner.forward(position, prev_scenegraphs=[], instruction=bench.task_desc,
# #                         objects_list=..., dim_x=bench.dim_x, dim_y=bench.dim_y,
# #                         dx=..., dy=..., orientation=False, scale=False)
# #     probe_capture.save_probe_data("probe_out/run1")
# #
# # Subprocess will NOT work: the monkeypatch and in-memory activations live in
# # this process only.
# # =============================================================================
# from __future__ import annotations

# import os
# import re
# import json
# import copy
# from typing import Any, Dict, List, Tuple

# import numpy as np

# import probe_backend

# EMPTY_SYMBOL = "E"
# EGO_LABEL = "ego_person"

# # ---- run config (set by enable_probing) -------------------------------------
# _CFG = {"grid_rows": 5, "grid_cols": 5, "min_area_frac": 0.10,
#         "label_mode": "category", "scene_id": "scene_0"}


# # =============================================================================
# # Geometry (self-contained; bbox = [x, y, w, h], (x,y) = bottom-left corner)
# # =============================================================================
# def _corners(bbox) -> Tuple[float, float, float, float]:
#     x, y, w, h = [float(v) for v in bbox]
#     return x, y, x + w, y + h


# def _inter_area(a, b) -> float:
#     ax0, ay0, ax1, ay1 = a
#     bx0, by0, bx1, by1 = b
#     ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
#     iy = max(0.0, min(ay1, by1) - max(ay0, by0))
#     return ix * iy


# def room_extent(dim_x: float, dim_y: float) -> Tuple[float, float, float, float]:
#     return -dim_x / 2.0, 0.0, dim_x / 2.0, dim_y


# def _real_objs(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     out = []
#     for o in objs:
#         if not isinstance(o, dict) or o.get("label") == EGO_LABEL:
#             continue
#         bbox = o.get("bbox")
#         if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
#             continue
#         _, _, w, h = [float(v) for v in bbox]
#         if w <= 0 or h <= 0:
#             continue
#         out.append(o)
#     return out


# # ---- label canonicalisation (compact copy of the collector's category mode) --
# _CANON = [
#     ("tv stand", ["tv stand"]), ("potted plant", ["potted plant"]),
#     ("coffee table", ["coffee table"]), ("conference table", ["conference table"]),
#     ("dining table", ["dining table"]), ("bedside table", ["bedside table"]),
#     ("side table", ["side table"]), ("reception desk", ["reception desk"]),
#     ("bean bag chair", ["bean bag chair", "bean bag"]), ("office chair", ["office chair"]),
#     ("gaming chair", ["gaming chair"]), ("armchair", ["armchair"]),
#     ("bookshelf", ["bookshelf"]), ("floor lamp", ["floor lamp", "lamp"]),
#     ("grand piano", ["piano"]), ("wardrobe", ["wardrobe"]), ("dresser", ["dresser"]),
#     ("cabinet", ["cabinet"]), ("mirror", ["mirror"]), ("shelf", ["shelf"]),
#     ("sofa", ["sofa", "couch"]), ("bed", ["bed"]), ("desk", ["desk"]),
#     ("trash can", ["trash can"]), ("chair", ["chair"]), ("table", ["table"]),
# ]
# _ORD = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh",
#         "eighth", "ninth", "tenth", "left", "right"}


# def _norm_label(lab: str, mode: str) -> str:
#     if mode == "identity":
#         return lab
#     s = str(lab).lower().replace("_", " ")
#     s = re.sub(r"\s+\d+\s*$", " ", s)
#     s = " ".join(w for w in s.split() if w not in _ORD)
#     s = re.sub(r"\s+", " ", s).strip()
#     for canon, keys in _CANON:
#         if any(k in s for k in keys):
#             return canon
#     return s or str(lab)


# def build_grid_strings(objs: List[Dict[str, Any]], rows: int, cols: int,
#                        extent, min_area_frac: float, mode: str) -> Tuple[List[str], List[bool]]:
#     """Row-major R*C grid of label STRINGS (argmax intersection) + contested mask."""
#     x_min, y_min, x_max, y_max = extent
#     cw = (x_max - x_min) / cols
#     ch = (y_max - y_min) / rows
#     cell_area = max(cw * ch, 1e-9)
#     n = rows * cols
#     best = [0.0] * n
#     labels = [EMPTY_SYMBOL] * n
#     ncov = [0] * n
#     for o in _real_objs(objs):
#         lab = _norm_label(o.get("label", "obj"), mode)
#         obb = _corners(o["bbox"])
#         for r in range(rows):
#             cy0 = y_min + r * ch
#             for c in range(cols):
#                 cx0 = x_min + c * cw
#                 area = _inter_area(obb, (cx0, cy0, cx0 + cw, cy0 + ch))
#                 if area <= min_area_frac * cell_area:
#                     continue
#                 k = r * cols + c
#                 ncov[k] += 1
#                 if area > best[k]:
#                     best[k] = area
#                     labels[k] = lab
#     contested = [c >= 2 for c in ncov]
#     return labels, contested


# def legality_overlap(prior_objs: List[Dict[str, Any]],
#                      delta_objs: List[Dict[str, Any]]) -> Tuple[bool, bool, List[bool]]:
#     """
#     legality per your definition: a new placement must not overlap a PREVIOUS one.
#     "Previous" for object j = all prior committed objects + delta objects placed
#     before j (output order). Returns (overlap_any, any_within_delta, per_new_overlap).
#     """
#     prior = [_corners(o["bbox"]) for o in _real_objs(prior_objs)]
#     dels = [_corners(o["bbox"]) for o in _real_objs(delta_objs)]
#     per = []
#     within = False
#     for j, dj in enumerate(dels):
#         hit = any(_inter_area(dj, p) > 1e-9 for p in prior)
#         for k in range(j):
#             if _inter_area(dj, dels[k]) > 1e-9:
#                 hit = True
#                 within = True
#         per.append(hit)
#     return (any(per), within, per)


# def out_of_bounds(delta_objs, extent) -> bool:
#     x_min, y_min, x_max, y_max = extent
#     for o in _real_objs(delta_objs):
#         x0, y0, x1, y1 = _corners(o["bbox"])
#         if x0 < x_min - 1e-6 or y0 < y_min - 1e-6 or x1 > x_max + 1e-6 or y1 > y_max + 1e-6:
#             return True
#     return False


# # =============================================================================
# # Probed placement manager (mirrors ObjectPlacementManager.forward + hooks)
# # =============================================================================
# def make_probed_manager(base_cls):
#     """Factory so we subclass the ACTUAL class from the generator module."""
#     import utils_copy_img  # same import the generator uses
#     from Heuristic_wo_dspy import build_object_placement_prompt

#     class ProbedObjectPlacementManager(base_cls):
#         _region_counter = 0

#         def forward(self, position, plan, new_area_range, allowed_objs, dim_x, dim_y,
#                     consistency_block, movement_block, base_scene, dx, dy, env_note="No note"):
#             region_id = ProbedObjectPlacementManager._region_counter
#             ProbedObjectPlacementManager._region_counter += 1
#             rows, cols = _CFG["grid_rows"], _CFG["grid_cols"]
#             extent = room_extent(dim_x, dim_y)
#             prior_objs = list(base_scene.get("objects", []))   # committed before this call

#             plan_text = "\n".join(
#                 t.get("execution_prompt", "") for t in plan.get("atomic_tasks", [])
#                 if t.get("execution_prompt"))
#             instruction_prompt = build_object_placement_prompt(
#                 plan_text, position, dim_x, dim_y, consistency_block, movement_block,
#                 env_note, new_area_range)
#             new_area_range_parsed = utils_copy_img._parse_new_area_range(new_area_range)

#             best_score = float("-inf")
#             best_delta: List[Dict[str, Any]] = []
#             best_iter = -1
#             feedback = ""
#             self._refine_logs = []
#             iter_record_idx: List[int] = []
#             prev_occ = None            # occupancy strings of the previous candidate
#             prev_overlap = False       # overlap_any of the previous candidate

#             for i in range(self.N):
#                 seed = utils_copy_img.generate_random_seed()
#                 # tag the upcoming probed call
#                 probe_backend.set_context(
#                     scene_id=_CFG["scene_id"], region_id=region_id, iteration=i + 1,
#                     call_type="placement", n_prior=len(_real_objs(prior_objs)))
#                 delta = self._generate_placement(instruction_prompt, feedback, seed)
#                 rec_idx = probe_backend.LAST_RECORD_IDX
#                 iter_record_idx.append(rec_idx)

#                 delta_norm = [utils_copy_img.normalize_obj(o) for o in delta]

#                 # ---- geometric ground truth for THIS candidate ----
#                 occ, contested = build_grid_strings(
#                     prior_objs + delta_norm, rows, cols, extent,
#                     _CFG["min_area_frac"], _CFG["label_mode"])
#                 occ_prior, _ = build_grid_strings(
#                     prior_objs, rows, cols, extent, _CFG["min_area_frac"], _CFG["label_mode"])
#                 overlap_any, within, per_new = legality_overlap(prior_objs, delta_norm)
#                 oob = out_of_bounds(delta_norm, extent)

#                 # ---- refinement dynamics vs the PREVIOUS candidate (Othello "flips") ----
#                 # changed[k] = the board cell flipped between candidate i-1 and i, i.e.
#                 # the model's edit in response to the feedback. Only defined for i>0.
#                 # overlap_resolved = the illegal predecessor became legal here.
#                 has_transition = prev_occ is not None
#                 if has_transition:
#                     changed = [occ[k] != prev_occ[k] for k in range(len(occ))]
#                     overlap_resolved = int(prev_overlap and not overlap_any)
#                     resolvable = bool(prev_overlap)
#                 else:
#                     changed = [False] * (rows * cols)
#                     overlap_resolved = 0
#                     resolvable = False

#                 probe_backend.attach_labels(
#                     rec_idx,
#                     occupancy_str=occ, occupancy_prior_str=occ_prior,
#                     contested=contested, overlap_any=bool(overlap_any),
#                     within_overlap=bool(within), oob=bool(oob),
#                     legality=int(not overlap_any),
#                     changed=changed, has_transition=bool(has_transition),
#                     prev_overlap=bool(prev_overlap), overlap_resolved=overlap_resolved,
#                     resolvable=resolvable,
#                     n_new=len(delta_norm), region_id=region_id,
#                     scene_id=_CFG["scene_id"], iteration=i + 1, committed=False)
#                 prev_occ, prev_overlap = occ, bool(overlap_any)

#                 # ---- original behaviour: visualize, score, feedback, keep best ----
#                 tmp_scene = copy.deepcopy(base_scene)
#                 tmp_scene["objects"].extend(delta_norm)
#                 utils_copy_img.visualize(tmp_scene, position=position, save_dir=self.save_dir,
#                                          save_name=f"check_{i+1}", sem_name=f"sem_check_{i+1}")
#                 score, diagnostics = utils_copy_img.score_candidate_delta(
#                     delta, base_scene, allowed_objs, new_area_range_parsed, position, dx, dy)
#                 feedback = ("Placement Diagnostics / Issues:\n"
#                             + "\n".join(f"- {d}" for d in diagnostics)
#                             + "\n\nAction Required:\nUpdate the positions or bounding box "
#                             "dimensions of the objects having placement problems to address "
#                             "the above issues. Ensure that any changes do not cause bounding "
#                             "box overlaps with other objects in the scene.")
#                 self._refine_logs.append({"iteration": i + 1, "delta": delta_norm,
#                                           "score": score, "diagnostics": diagnostics,
#                                           "feedback_text": feedback})
#                 log_path = os.path.join(self.save_dir,
#                                         f"refine_logs__{position[0]}_{position[1]}.json")
#                 with open(log_path, "w") as f:
#                     json.dump(self._refine_logs, f, indent=4, default=str)

#                 if score > best_score:
#                     best_score = score
#                     best_delta = delta
#                     best_iter = i
#                 if score >= self.threshold:
#                     break

#             # mark the committed candidate's record
#             if 0 <= best_iter < len(iter_record_idx):
#                 probe_backend.attach_labels(iter_record_idx[best_iter], committed=True)

#             delta_objects = [utils_copy_img.normalize_obj(o) for o in best_delta]
#             if delta_objects:
#                 delta_objects = utils_copy_img.assign_progressive_ids(
#                     delta_objects, scene_graph_dir=self.save_dir)
#             base_scene["objects"].extend(delta_objects)
#             return base_scene

#     return ProbedObjectPlacementManager


# # =============================================================================
# # Enable / configure / export
# # =============================================================================
# def enable_probing(model_path: str = "Qwen/Qwen3.5-27B", dtype: str = "bfloat16",
#                    device_map: str = "auto", num_probe_layers: int = 8,
#                    grid_rows: int = 5, grid_cols: int = 5, min_area_frac: float = 0.10,
#                    label_mode: str = "category", random_init: bool = False):
#     _CFG.update(grid_rows=grid_rows, grid_cols=grid_cols,
#                 min_area_frac=min_area_frac, label_mode=label_mode)
#     probe_backend.init_probe(model_path=model_path, dtype=dtype, device_map=device_map,
#                              num_probe_layers=num_probe_layers, random_init=random_init)
#     import Heuristic_wo_dspy as H
#     H.llm_complete = probe_backend.probed_complete
#     H.ObjectPlacementManager = make_probed_manager(H.ObjectPlacementManager)
#     print("[probe_capture] patched llm_complete + ObjectPlacementManager; "
#           f"grid {grid_rows}x{grid_cols}, label_mode={label_mode}")


# def set_scene(scene_id: str):
#     _CFG["scene_id"] = str(scene_id)


# def save_probe_data(out_dir: str):
#     os.makedirs(out_dir, exist_ok=True)
#     recs = [r for r in probe_backend.RECORDS if r.get("labels")]  # placement calls only
#     if not recs:
#         raise SystemExit("No labelled placement records. Did generation run in-process "
#                          "AFTER enable_probing(), with orientation=False?")

#     rows, cols = _CFG["grid_rows"], _CFG["grid_cols"]
#     n_cells = rows * cols
#     layers = probe_backend.get_probe().layers
#     positions = ["prompt_end", "delta_end"]

#     # vocab over all occupancy strings
#     vocab = {EMPTY_SYMBOL: 0}
#     for r in recs:
#         for grid_key in ("occupancy_str", "occupancy_prior_str"):
#             for lab in r["labels"].get(grid_key, []):
#                 if lab not in vocab:
#                     vocab[lab] = len(vocab)

#     def enc_grid(strs): return np.array([vocab.get(s, 0) for s in strs], dtype=np.int64)

#     N = len(recs)
#     Y_occ = np.zeros((N, n_cells), np.int64)
#     Y_occ_prior = np.zeros((N, n_cells), np.int64)
#     contested = np.zeros((N, n_cells), bool)
#     legality = np.zeros(N, np.int64)
#     overlap_any = np.zeros(N, bool)
#     within_overlap = np.zeros(N, bool)
#     oob = np.zeros(N, bool)
#     committed = np.zeros(N, bool)
#     iteration = np.zeros(N, np.int64)
#     region_ids = np.zeros(N, np.int64)
#     scene_ids = np.zeros(N, np.int64)
#     # refinement-dynamics targets
#     Y_changed = np.zeros((N, n_cells), np.int64)     # board-diff vs previous candidate
#     has_transition = np.zeros(N, bool)               # True for iterations 2..N
#     prev_overlap = np.zeros(N, bool)
#     overlap_resolved = np.zeros(N, np.int64)
#     resolvable = np.zeros(N, bool)                   # has_transition & prev candidate illegal
#     scene_key = {}
#     X = {p: {li: np.zeros((N, probe_backend.get_probe().hidden_size), np.float16)
#              for li in layers} for p in positions}

#     for i, r in enumerate(recs):
#         lab = r["labels"]
#         Y_occ[i] = enc_grid(lab["occupancy_str"])
#         Y_occ_prior[i] = enc_grid(lab["occupancy_prior_str"])
#         contested[i] = np.array(lab["contested"], bool)
#         legality[i] = lab["legality"]
#         overlap_any[i] = lab["overlap_any"]
#         within_overlap[i] = lab.get("within_overlap", False)
#         oob[i] = lab["oob"]
#         committed[i] = lab.get("committed", False)
#         iteration[i] = lab["iteration"]
#         region_ids[i] = lab["region_id"]
#         Y_changed[i] = np.array(lab.get("changed", [False] * n_cells), np.int64)
#         has_transition[i] = lab.get("has_transition", False)
#         prev_overlap[i] = lab.get("prev_overlap", False)
#         overlap_resolved[i] = lab.get("overlap_resolved", 0)
#         resolvable[i] = lab.get("resolvable", False)
#         sk = lab["scene_id"]
#         scene_ids[i] = scene_key.setdefault(sk, len(scene_key))
#         for p in positions:
#             for li in layers:
#                 X[p][li][i] = r["activations"][p][li]

#     npz = dict(Y_occupancy=Y_occ, Y_occupancy_prior=Y_occ_prior, contested=contested,
#                in_room=np.ones((N, n_cells), bool), legality=legality,
#                overlap_any=overlap_any, within_overlap=within_overlap, oob=oob,
#                committed=committed, iteration=iteration, region_ids=region_ids,
#                scene_ids=scene_ids, layer_indices=np.array(layers, np.int64),
#                Y_changed=Y_changed, has_transition=has_transition,
#                prev_overlap=prev_overlap, overlap_resolved=overlap_resolved,
#                resolvable=resolvable)
#     for p in positions:
#         for li in layers:
#             npz[f"X_{p}_layer_{li}"] = X[p][li]
#     np.savez_compressed(os.path.join(out_dir, "activations.npz"), **npz)

#     meta = {"grid_rows": rows, "grid_cols": cols, "n_cells": n_cells,
#             "symbol_vocab": vocab, "n_classes": len(vocab), "empty_class_id": 0,
#             "layer_indices": layers, "probe_positions": positions,
#             "targets": ["occupancy", "occupancy_prior", "contested", "legality",
#                         "movement", "overlap_resolved"],
#             "label_mode": _CFG["label_mode"], "n_samples": N,
#             "n_scenes": len(scene_key), "hidden_size": probe_backend.get_probe().hidden_size,
#             "legality_convention": "1=legal (no new object overlaps a previous placement)"}
#     with open(os.path.join(out_dir, "meta.json"), "w") as f:
#         json.dump(meta, f, indent=2)
#     print(f"[probe_capture] wrote {N} samples ({len(scene_key)} scenes), "
#           f"{len(vocab)} classes -> {out_dir}/activations.npz + meta.json")


# # ---- offline self-test of the geometry (no torch/model) ---------------------
# if __name__ == "__main__":
#     ext = room_extent(10, 10)
#     A = {"label": "sofa", "bbox": [-1, 0, 2, 1]}
#     B = {"label": "table", "bbox": [0.5, 0.5, 2, 1]}    # overlaps A
#     C = {"label": "lamp", "bbox": [3, 6, 0.5, 0.5]}     # isolated
#     print("grid:", build_grid_strings([A, C], 5, 5, ext, 0.10, "category")[0])
#     print("overlap A,B (expect True):", legality_overlap([], [A, B])[0])
#     print("overlap A,C (expect False):", legality_overlap([], [A, C])[0])
#     print("oob far object (expect True):",
#           out_of_bounds([{"label": "x", "bbox": [40, 40, 1, 1]}], ext))


#!/usr/bin/env python3
# =============================================================================
# probe_capture.py
# -----------------------------------------------------------------------------
# Wires probe_backend into the SAGE generator IN-PROCESS and builds Othello-style
# ground truth from the generation loop itself. Two monkeypatches:
#   1. Heuristic_wo_dspy.llm_complete   -> probe_backend.probed_complete
#      (every model call now runs locally and records activations)
#   2. Heuristic_wo_dspy.ObjectPlacementManager -> ProbedObjectPlacementManager
#      (each best-of-N placement candidate is tagged with scene/region/iteration
#       and paired with geometric GT computed here)
#
# Ground truth per placement candidate (external geometry = "the rules"):
#   * occupancy       : R x C occupancy grid of (prior committed + this candidate)
#   * occupancy_prior : R x C grid of prior committed only (accumulated state)
#   * contested       : per-cell "covered by >=2 objects" (overlap footprint)
#   * legality        : 1 if NO new object overlaps a PREVIOUS placement, else 0.
#                       "Previous" = objects committed before this call, in output
#                       order union earlier objects in the same candidate.
#   * out_of_bounds   : any new object leaves the room extent
# Two probe token positions are stored (prompt_end, delta_end) -- see backend.
#
# Run your generation exactly as usual, but IN THE SAME PROCESS, e.g.:
#     import probe_capture
#     probe_capture.enable_probing(model_path="Qwen/Qwen3.5-27B",
#                                  grid_rows=5, grid_cols=5, num_probe_layers=8)
#     import Heuristic_wo_dspy as H          # import AFTER enable_probing
#     for bench in benchmarks:
#         probe_capture.set_scene(bench.name)
#         planner = H.ScenePlanner(save_dir)               # picks up probed manager
#         planner.forward(position, prev_scenegraphs=[], instruction=bench.task_desc,
#                         objects_list=..., dim_x=bench.dim_x, dim_y=bench.dim_y,
#                         dx=..., dy=..., orientation=False, scale=False)
#     probe_capture.save_probe_data("probe_out/run1")
#
# Subprocess will NOT work: the monkeypatch and in-memory activations live in
# this process only.
# =============================================================================
from __future__ import annotations

import os
import re
import json
import copy
from typing import Any, Dict, List, Tuple

import numpy as np

import probe_backend

EMPTY_SYMBOL = "E"
EGO_LABEL = "ego_person"

# ---- run config (set by enable_probing) -------------------------------------
_CFG = {"grid_rows": 5, "grid_cols": 5, "min_area_frac": 0.10,
        "label_mode": "category", "scene_id": "scene_0"}
_REGION = {"n": 0}   # region index WITHIN the current scene; reset by set_scene()


# =============================================================================
# Geometry (self-contained; bbox = [x, y, w, h], (x,y) = bottom-left corner)
# =============================================================================
def _corners(bbox) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x, y, x + w, y + h


def _inter_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return ix * iy


def room_extent(dim_x: float, dim_y: float) -> Tuple[float, float, float, float]:
    return -dim_x / 2.0, 0.0, dim_x / 2.0, dim_y


def _real_objs(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for o in objs:
        if not isinstance(o, dict) or o.get("label") == EGO_LABEL:
            continue
        bbox = o.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        _, _, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        out.append(o)
    return out


# ---- label canonicalisation (compact copy of the collector's category mode) --
_CANON = [
    ("tv stand", ["tv stand"]), ("potted plant", ["potted plant"]),
    ("coffee table", ["coffee table"]), ("conference table", ["conference table"]),
    ("dining table", ["dining table"]), ("bedside table", ["bedside table"]),
    ("side table", ["side table"]), ("reception desk", ["reception desk"]),
    ("bean bag chair", ["bean bag chair", "bean bag"]), ("office chair", ["office chair"]),
    ("gaming chair", ["gaming chair"]), ("armchair", ["armchair"]),
    ("bookshelf", ["bookshelf"]), ("floor lamp", ["floor lamp", "lamp"]),
    ("grand piano", ["piano"]), ("wardrobe", ["wardrobe"]), ("dresser", ["dresser"]),
    ("cabinet", ["cabinet"]), ("mirror", ["mirror"]), ("shelf", ["shelf"]),
    ("sofa", ["sofa", "couch"]), ("bed", ["bed"]), ("desk", ["desk"]),
    ("trash can", ["trash can"]), ("chair", ["chair"]), ("table", ["table"]),
]
_ORD = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth", "tenth", "left", "right"}


def _norm_label(lab: str, mode: str) -> str:
    if mode == "identity":
        return lab
    s = str(lab).lower().replace("_", " ")
    s = re.sub(r"\s+\d+\s*$", " ", s)
    s = " ".join(w for w in s.split() if w not in _ORD)
    s = re.sub(r"\s+", " ", s).strip()
    for canon, keys in _CANON:
        if any(k in s for k in keys):
            return canon
    return s or str(lab)


def build_grid_strings(objs: List[Dict[str, Any]], rows: int, cols: int,
                       extent, min_area_frac: float, mode: str) -> Tuple[List[str], List[bool]]:
    """Row-major R*C grid of label STRINGS (argmax intersection) + contested mask."""
    x_min, y_min, x_max, y_max = extent
    cw = (x_max - x_min) / cols
    ch = (y_max - y_min) / rows
    cell_area = max(cw * ch, 1e-9)
    n = rows * cols
    best = [0.0] * n
    labels = [EMPTY_SYMBOL] * n
    ncov = [0] * n
    for o in _real_objs(objs):
        lab = _norm_label(o.get("label", "obj"), mode)
        obb = _corners(o["bbox"])
        for r in range(rows):
            cy0 = y_min + r * ch
            for c in range(cols):
                cx0 = x_min + c * cw
                area = _inter_area(obb, (cx0, cy0, cx0 + cw, cy0 + ch))
                if area <= min_area_frac * cell_area:
                    continue
                k = r * cols + c
                ncov[k] += 1
                if area > best[k]:
                    best[k] = area
                    labels[k] = lab
    contested = [c >= 2 for c in ncov]
    return labels, contested


def legality_overlap(prior_objs: List[Dict[str, Any]],
                     delta_objs: List[Dict[str, Any]]) -> Tuple[bool, bool, List[bool]]:
    """
    legality per your definition: a new placement must not overlap a PREVIOUS one.
    "Previous" for object j = all prior committed objects + delta objects placed
    before j (output order). Returns (overlap_any, any_within_delta, per_new_overlap).
    """
    prior = [_corners(o["bbox"]) for o in _real_objs(prior_objs)]
    dels = [_corners(o["bbox"]) for o in _real_objs(delta_objs)]
    per = []
    within = False
    for j, dj in enumerate(dels):
        hit = any(_inter_area(dj, p) > 1e-9 for p in prior)
        for k in range(j):
            if _inter_area(dj, dels[k]) > 1e-9:
                hit = True
                within = True
        per.append(hit)
    return (any(per), within, per)


def out_of_bounds(delta_objs, extent) -> bool:
    x_min, y_min, x_max, y_max = extent
    for o in _real_objs(delta_objs):
        x0, y0, x1, y1 = _corners(o["bbox"])
        if x0 < x_min - 1e-6 or y0 < y_min - 1e-6 or x1 > x_max + 1e-6 or y1 > y_max + 1e-6:
            return True
    return False


# =============================================================================
# Probed placement manager (mirrors ObjectPlacementManager.forward + hooks)
# =============================================================================
def make_probed_manager(base_cls):
    """Factory so we subclass the ACTUAL class from the generator module."""
    import utils_copy_img  # same import the generator uses
    from Heuristic_wo_dspy import build_object_placement_prompt

    class ProbedObjectPlacementManager(base_cls):
        def forward(self, position, plan, new_area_range, allowed_objs, dim_x, dim_y,
                    consistency_block, movement_block, base_scene, dx, dy, env_note="No note"):
            # region_id must identify the region WITHIN A SCENE, not a process-global
            # counter: a global counter makes the same region get different ids in two
            # runs, so baseline/adversarial records never align. _REGION is reset by
            # set_scene().
            region_id = _REGION["n"]
            _REGION["n"] += 1
            rows, cols = _CFG["grid_rows"], _CFG["grid_cols"]
            extent = room_extent(dim_x, dim_y)
            prior_objs = list(base_scene.get("objects", []))   # committed before this call

            plan_text = "\n".join(
                t.get("execution_prompt", "") for t in plan.get("atomic_tasks", [])
                if t.get("execution_prompt"))
            instruction_prompt = build_object_placement_prompt(
                plan_text, position, dim_x, dim_y, consistency_block, movement_block,
                env_note, new_area_range)
            new_area_range_parsed = utils_copy_img._parse_new_area_range(new_area_range)

            best_score = float("-inf")
            best_delta: List[Dict[str, Any]] = []
            best_iter = -1
            feedback = ""
            self._refine_logs = []
            iter_record_idx: List[int] = []
            prev_occ = None            # occupancy strings of the previous candidate
            prev_overlap = False       # overlap_any of the previous candidate

            for i in range(self.N):
                seed = utils_copy_img.generate_random_seed()
                # tag the upcoming probed call
                probe_backend.set_context(
                    scene_id=_CFG["scene_id"], region_id=region_id, iteration=i + 1,
                    call_type="placement", n_prior=len(_real_objs(prior_objs)))
                delta = self._generate_placement(instruction_prompt, feedback, seed)
                rec_idx = probe_backend.LAST_RECORD_IDX
                iter_record_idx.append(rec_idx)

                delta_norm = [utils_copy_img.normalize_obj(o) for o in delta]

                # ---- geometric ground truth for THIS candidate ----
                occ, contested = build_grid_strings(
                    prior_objs + delta_norm, rows, cols, extent,
                    _CFG["min_area_frac"], _CFG["label_mode"])
                occ_prior, _ = build_grid_strings(
                    prior_objs, rows, cols, extent, _CFG["min_area_frac"], _CFG["label_mode"])
                overlap_any, within, per_new = legality_overlap(prior_objs, delta_norm)
                oob = out_of_bounds(delta_norm, extent)

                # ---- refinement dynamics vs the PREVIOUS candidate (Othello "flips") ----
                # changed[k] = the board cell flipped between candidate i-1 and i, i.e.
                # the model's edit in response to the feedback. Only defined for i>0.
                # overlap_resolved = the illegal predecessor became legal here.
                has_transition = prev_occ is not None
                if has_transition:
                    changed = [occ[k] != prev_occ[k] for k in range(len(occ))]
                    overlap_resolved = int(prev_overlap and not overlap_any)
                    resolvable = bool(prev_overlap)
                else:
                    changed = [False] * (rows * cols)
                    overlap_resolved = 0
                    resolvable = False

                probe_backend.attach_labels(
                    rec_idx,
                    occupancy_str=occ, occupancy_prior_str=occ_prior,
                    contested=contested, overlap_any=bool(overlap_any),
                    within_overlap=bool(within), oob=bool(oob),
                    legality=int(not overlap_any),
                    changed=changed, has_transition=bool(has_transition),
                    prev_overlap=bool(prev_overlap), overlap_resolved=overlap_resolved,
                    resolvable=resolvable,
                    n_new=len(delta_norm), region_id=region_id,
                    scene_id=_CFG["scene_id"], iteration=i + 1, committed=False)
                prev_occ, prev_overlap = occ, bool(overlap_any)

                # ---- original behaviour: visualize, score, feedback, keep best ----
                tmp_scene = copy.deepcopy(base_scene)
                tmp_scene["objects"].extend(delta_norm)
                utils_copy_img.visualize(tmp_scene, position=position, save_dir=self.save_dir,
                                         save_name=f"check_{i+1}", sem_name=f"sem_check_{i+1}")
                score, diagnostics = utils_copy_img.score_candidate_delta(
                    delta, base_scene, allowed_objs, new_area_range_parsed, position, dx, dy)
                feedback = ("Placement Diagnostics / Issues:\n"
                            + "\n".join(f"- {d}" for d in diagnostics)
                            + "\n\nAction Required:\nUpdate the positions or bounding box "
                            "dimensions of the objects having placement problems to address "
                            "the above issues. Ensure that any changes do not cause bounding "
                            "box overlaps with other objects in the scene.")
                self._refine_logs.append({"iteration": i + 1, "delta": delta_norm,
                                          "score": score, "diagnostics": diagnostics,
                                          "feedback_text": feedback})
                log_path = os.path.join(self.save_dir,
                                        f"refine_logs__{position[0]}_{position[1]}.json")
                with open(log_path, "w") as f:
                    json.dump(self._refine_logs, f, indent=4, default=str)

                if score > best_score:
                    best_score = score
                    best_delta = delta
                    best_iter = i
                if score >= self.threshold:
                    break

            # mark the committed candidate's record
            if 0 <= best_iter < len(iter_record_idx):
                probe_backend.attach_labels(iter_record_idx[best_iter], committed=True)

            delta_objects = [utils_copy_img.normalize_obj(o) for o in best_delta]
            if delta_objects:
                delta_objects = utils_copy_img.assign_progressive_ids(
                    delta_objects, scene_graph_dir=self.save_dir)
            base_scene["objects"].extend(delta_objects)
            return base_scene

    return ProbedObjectPlacementManager


# =============================================================================
# Enable / configure / export
# =============================================================================
def enable_probing(model_path: str = "Qwen/Qwen3.5-27B", dtype: str = "bfloat16",
                   device_map: str = "auto", num_probe_layers: int = 8,
                   grid_rows: int = 5, grid_cols: int = 5, min_area_frac: float = 0.10,
                   label_mode: str = "category", random_init: bool = False):
    _CFG.update(grid_rows=grid_rows, grid_cols=grid_cols,
                min_area_frac=min_area_frac, label_mode=label_mode)
    probe_backend.init_probe(model_path=model_path, dtype=dtype, device_map=device_map,
                             num_probe_layers=num_probe_layers, random_init=random_init)
    import Heuristic_wo_dspy as H
    H.llm_complete = probe_backend.probed_complete
    H.ObjectPlacementManager = make_probed_manager(H.ObjectPlacementManager)
    print("[probe_capture] patched llm_complete + ObjectPlacementManager; "
          f"grid {grid_rows}x{grid_cols}, label_mode={label_mode}")


def set_scene(scene_id: str):
    _CFG["scene_id"] = str(scene_id)
    _REGION["n"] = 0   # region ids restart per scene -> comparable across runs


def save_probe_data(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    recs = [r for r in probe_backend.RECORDS if r.get("labels")]  # placement calls only
    if not recs:
        raise SystemExit("No labelled placement records. Did generation run in-process "
                         "AFTER enable_probing(), with orientation=False?")

    rows, cols = _CFG["grid_rows"], _CFG["grid_cols"]
    n_cells = rows * cols
    layers = probe_backend.get_probe().layers
    positions = ["prompt_end", "delta_end"]

    # vocab over all occupancy strings
    vocab = {EMPTY_SYMBOL: 0}
    for r in recs:
        for grid_key in ("occupancy_str", "occupancy_prior_str"):
            for lab in r["labels"].get(grid_key, []):
                if lab not in vocab:
                    vocab[lab] = len(vocab)

    def enc_grid(strs): return np.array([vocab.get(s, 0) for s in strs], dtype=np.int64)

    N = len(recs)
    Y_occ = np.zeros((N, n_cells), np.int64)
    Y_occ_prior = np.zeros((N, n_cells), np.int64)
    contested = np.zeros((N, n_cells), bool)
    legality = np.zeros(N, np.int64)
    overlap_any = np.zeros(N, bool)
    within_overlap = np.zeros(N, bool)
    oob = np.zeros(N, bool)
    committed = np.zeros(N, bool)
    iteration = np.zeros(N, np.int64)
    region_ids = np.zeros(N, np.int64)
    scene_ids = np.zeros(N, np.int64)
    # refinement-dynamics targets
    Y_changed = np.zeros((N, n_cells), np.int64)     # board-diff vs previous candidate
    has_transition = np.zeros(N, bool)               # True for iterations 2..N
    prev_overlap = np.zeros(N, bool)
    overlap_resolved = np.zeros(N, np.int64)
    resolvable = np.zeros(N, bool)                   # has_transition & prev candidate illegal
    scene_key = {}
    X = {p: {li: np.zeros((N, probe_backend.get_probe().hidden_size), np.float16)
             for li in layers} for p in positions}

    for i, r in enumerate(recs):
        lab = r["labels"]
        Y_occ[i] = enc_grid(lab["occupancy_str"])
        Y_occ_prior[i] = enc_grid(lab["occupancy_prior_str"])
        contested[i] = np.array(lab["contested"], bool)
        legality[i] = lab["legality"]
        overlap_any[i] = lab["overlap_any"]
        within_overlap[i] = lab.get("within_overlap", False)
        oob[i] = lab["oob"]
        committed[i] = lab.get("committed", False)
        iteration[i] = lab["iteration"]
        region_ids[i] = lab["region_id"]
        Y_changed[i] = np.array(lab.get("changed", [False] * n_cells), np.int64)
        has_transition[i] = lab.get("has_transition", False)
        prev_overlap[i] = lab.get("prev_overlap", False)
        overlap_resolved[i] = lab.get("overlap_resolved", 0)
        resolvable[i] = lab.get("resolvable", False)
        sk = lab["scene_id"]
        scene_ids[i] = scene_key.setdefault(sk, len(scene_key))
        for p in positions:
            for li in layers:
                X[p][li][i] = r["activations"][p][li]

    npz = dict(Y_occupancy=Y_occ, Y_occupancy_prior=Y_occ_prior, contested=contested,
               in_room=np.ones((N, n_cells), bool), legality=legality,
               overlap_any=overlap_any, within_overlap=within_overlap, oob=oob,
               committed=committed, iteration=iteration, region_ids=region_ids,
               scene_ids=scene_ids, layer_indices=np.array(layers, np.int64),
               Y_changed=Y_changed, has_transition=has_transition,
               prev_overlap=prev_overlap, overlap_resolved=overlap_resolved,
               resolvable=resolvable)
    for p in positions:
        for li in layers:
            npz[f"X_{p}_layer_{li}"] = X[p][li]
    np.savez_compressed(os.path.join(out_dir, "activations.npz"), **npz)

    meta = {"grid_rows": rows, "grid_cols": cols, "n_cells": n_cells,
            "symbol_vocab": vocab, "n_classes": len(vocab), "empty_class_id": 0,
            "layer_indices": layers, "probe_positions": positions,
            "targets": ["occupancy", "occupancy_prior", "contested", "legality",
                        "movement", "overlap_resolved"],
            "label_mode": _CFG["label_mode"], "n_samples": N,
            "n_scenes": len(scene_key), "hidden_size": probe_backend.get_probe().hidden_size,
            "legality_convention": "1=legal (no new object overlaps a previous placement)"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[probe_capture] wrote {N} samples ({len(scene_key)} scenes), "
          f"{len(vocab)} classes -> {out_dir}/activations.npz + meta.json")


# ---- offline self-test of the geometry (no torch/model) ---------------------
if __name__ == "__main__":
    ext = room_extent(10, 10)
    A = {"label": "sofa", "bbox": [-1, 0, 2, 1]}
    B = {"label": "table", "bbox": [0.5, 0.5, 2, 1]}    # overlaps A
    C = {"label": "lamp", "bbox": [3, 6, 0.5, 0.5]}     # isolated
    print("grid:", build_grid_strings([A, C], 5, 5, ext, 0.10, "category")[0])
    print("overlap A,B (expect True):", legality_overlap([], [A, B])[0])
    print("overlap A,C (expect False):", legality_overlap([], [A, C])[0])
    print("oob far object (expect True):",
          out_of_bounds([{"label": "x", "bbox": [40, 40, 1, 1]}], ext))