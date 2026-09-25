# from __future__ import annotations
# # =============================================================================
# # generate_scenes.py  --  generate SAGE scenes exactly like the probed pairs,
# # but with NO activation capture and NO refinement (first placement only).
# #
# # Structure produced:
# #   save_root/<adjective_tag>/seed_<i>/<rtype>_<name>/scene_pos_*.json
# #
# #   for each benchmark (baseline)
# #     for each adjective (appended to the environment note)
# #       for each of N fixed seeds (SAME seed list reused for every setup)
# #         -> one generated scene
# #
# # The model is loaded ONCE; all adjectives x seeds run in this single process.
# # Generation faithfully mirrors probe_backend (same enable_thinking=False chat
# # template, same DET_SEED+call_counter torch seeding, same do_sample/temp/top_p),
# # so the scenes match what you were probing -- minus the hidden-state pass.
# #
# # Heavy imports (Heuristic_wo_dspy, gen_backend, torch) are deferred so --dry-run
# # can print the full plan without a model or server.
# # =============================================================================

# import argparse
# import glob
# import json
# import os
# import re


# # ----------------------------------------------------------------------------
# # benchmark parsing (verbatim behaviour from adv_common)
# # ----------------------------------------------------------------------------
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


# def sanitize_tag(s: str) -> str:
#     return re.sub(r"[^A-Za-z0-9_-]+", "", s.strip().replace(" ", "_")) or "x"


# # ----------------------------------------------------------------------------
# # decode config + seeding (mirrors adv_common force_* / seed_everything, but
# # backed by gen_backend instead of probe_backend)
# # ----------------------------------------------------------------------------
# _RUN_SEED = None
# _SEEDCTR = {"n": 0}


# def configure_decode(decode, temperature, max_tokens, base_seed):
#     import Heuristic_wo_dspy as H
#     import gen_backend as G
#     inner = G.gen_complete
#     cap = max_tokens
#     if decode == "greedy":
#         def wrap(prompt, images=None, system=None, temperature=0.5,
#                  max_tokens=None, force_json=True):
#             return inner(prompt, images, system, 0.0, cap, force_json)
#         G.set_det_seed(None)
#         print(f"[gen] GREEDY decoding (temp=0), max_tokens={cap}")
#     else:
#         temp = temperature
#         def wrap(prompt, images=None, system=None, temperature=0.5,
#                  max_tokens=None, force_json=True):
#             return inner(prompt, images, system, temp, cap, force_json)
#         G.set_det_seed(base_seed)
#         print(f"[gen] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
#               f"max_tokens={cap}")
#     H.llm_complete = wrap


# def force_deterministic_seed_text():
#     """Make the 'Random seed: N' prompt text immune to global-RNG stream desync
#     (the run-0 divergence). Counter-based, derived from (_RUN_SEED, call#)."""
#     import utils_copy_img
#     if getattr(utils_copy_img.generate_random_seed, "_det_patched", False):
#         return

#     def det_seed():
#         _SEEDCTR["n"] += 1
#         base = _RUN_SEED if _RUN_SEED is not None else 0
#         return (base * 7919 + _SEEDCTR["n"] * 104729) % 10000001

#     det_seed._det_patched = True
#     utils_copy_img.generate_random_seed = det_seed


# def seed_everything(s):
#     import random
#     import numpy as np
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
#     global _RUN_SEED
#     _RUN_SEED = s
#     import gen_backend as G
#     G.set_det_seed(s)


# # ----------------------------------------------------------------------------
# # one pass over all specs for the current (adjective, seed)
# # ----------------------------------------------------------------------------
# def run_once(specs, adv_note, save_dir_root, resume=True, reset_seed_per_scene=False):
#     import Heuristic_wo_dspy as H
#     import utils_copy_img
#     import gen_backend as G

#     G.reset_call_counter()
#     force_deterministic_seed_text()
#     _SEEDCTR["n"] = 0
#     if _RUN_SEED is not None:
#         seed_everything(_RUN_SEED)

#     for name, task, assets, dim_x, dim_y in specs:
#         if reset_seed_per_scene:
#             # per-scene isolation: scene k under seed i gets identical initial RNG
#             # regardless of adjective or other scenes (cleaner per-scene counterfactual,
#             # but NOT how probe_backend behaved -- it reset once per run).
#             G.reset_call_counter()
#             _SEEDCTR["n"] = 0
#             if _RUN_SEED is not None:
#                 seed_everything(_RUN_SEED)

#         instruction = task + (adv_note or "")   # adjective appended to the env note
#         save_dir = os.path.join(save_dir_root, name)
#         os.makedirs(save_dir, exist_ok=True)

#         dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
#         pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
#         if resume and utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
#             print(f"    [skip] {name}: scene exists")
#             continue

#         planner = H.ScenePlanner(save_dir=save_dir)
#         planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
#                         objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
#                         orientation=False, scale=False)   # first placement only


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--benchmark-glob", required=True)
#     ap.add_argument("--save-root", required=True)
#     ap.add_argument("--adjectives", required=True,
#                     help="comma-separated, e.g. 'messy,cluttered,minimalist,cozy,spacious'")
#     ap.add_argument("--adj-template", default=" The room is {adj}.",
#                     help="how each adjective is appended to the env note; must contain {adj}. "
#                          "For a raw append use ' {adj}'.")
#     ap.add_argument("--include-baseline", action="store_true",
#                     help="also generate a no-adjective run (tag 'baseline')")
#     ap.add_argument("--num-seeds", type=int, default=50)
#     ap.add_argument("--base-seed", type=int, default=1234)
#     ap.add_argument("--seed-stride", type=int, default=10000,
#                     help="seed i = base_seed + i*stride; the SAME list is reused for every adjective")
#     ap.add_argument("--decode", choices=["sample", "greedy"], default="sample",
#                     help="sample = seeded sampling (needed for seed diversity); greedy = temp 0 "
#                          "(WARNING: all seeds collapse to one scene under greedy)")
#     ap.add_argument("--temperature", type=float, default=0.5)
#     ap.add_argument("--max-tokens", type=int, default=8192)
#     ap.add_argument("--model-path", default="Qwen/Qwen3.5-27B")
#     ap.add_argument("--reset-seed-per-scene", action="store_true",
#                     help="reset the per-call seed counter at every scene (per-scene RNG isolation). "
#                          "Default off = faithful to probe_backend (reset once per seed-run).")
#     ap.add_argument("--max-benchmarks", type=int, default=None)
#     ap.add_argument("--no-resume", action="store_true",
#                     help="regenerate even if a scene_pos_*.json already exists")
#     ap.add_argument("--dry-run", action="store_true",
#                     help="print the full plan (no model, no server) and exit")
#     args = ap.parse_args()

#     specs = list_specs(args.benchmark_glob, args.max_benchmarks)
#     if "{adj}" not in args.adj_template:
#         raise SystemExit("--adj-template must contain {adj}")

#     adjectives = [a.strip() for a in args.adjectives.split(",") if a.strip()]
#     conditions = ([("", "baseline")] if args.include_baseline else []) + \
#                  [(a, sanitize_tag(a)) for a in adjectives]

#     seeds = [args.base_seed + i * args.seed_stride for i in range(args.num_seeds)]

#     total = len(specs) * len(conditions) * len(seeds)
#     print(f"[gen] benchmarks={len(specs)}  conditions={len(conditions)}  "
#           f"seeds={len(seeds)}  -> {total} scenes")
#     print(f"[gen] seeds (same for every condition): {seeds if len(seeds) <= 10 else seeds[:5] + ['...'] + seeds[-2:]}")
#     if args.decode == "greedy" and len(seeds) > 1:
#         print("[gen] WARNING: --decode greedy is argmax -> all seeds produce the SAME scene. "
#               "Use --decode sample for a real 50-seed sweep.")

#     if args.dry_run:
#         for adj, tag in conditions:
#             note = args.adj_template.format(adj=adj) if adj else ""
#             print(f"\n=== condition '{tag}'  adj_note={note!r} ===")
#             for i, s in enumerate(seeds):
#                 root = os.path.join(args.save_root, tag, f"seed_{i:03d}")
#                 for name, task, assets, dim_x, dim_y in specs:
#                     instr = task + note
#                     print(f"  seed_{i:03d}(={s}) {os.path.join(root, name)}")
#                     print(f"      instruction: {instr!r}")
#                 if len(seeds) > 3 and i == 2:
#                     print(f"  ... ({len(seeds) - 3} more seeds, same layout)")
#                     break
#         print("\n[gen] dry-run only; no scenes generated.")
#         return

#     # ---- real run: load model once, then loop ----
#     import gen_backend as G
#     G.init_gen(model_path=args.model_path)
#     configure_decode(args.decode, args.temperature, args.max_tokens, args.base_seed)

#     os.makedirs(args.save_root, exist_ok=True)
#     with open(os.path.join(args.save_root, "seeds.json"), "w") as f:
#         json.dump({"seeds": seeds, "base_seed": args.base_seed,
#                    "seed_stride": args.seed_stride}, f, indent=2)

#     for adj, tag in conditions:
#         note = args.adj_template.format(adj=adj) if adj else ""
#         print(f"\n[gen] ################ CONDITION '{tag}'  adj_note={note!r} ################")
#         for i, s in enumerate(seeds):
#             set_repeat_seed(s)
#             root = os.path.join(args.save_root, tag, f"seed_{i:03d}")
#             print(f"[gen] --- {tag} seed_{i:03d} (seed={s}) ---")
#             run_once(specs, adv_note=note, save_dir_root=root,
#                      resume=not args.no_resume,
#                      reset_seed_per_scene=args.reset_seed_per_scene)

#     print(f"\n[gen] done -> {args.save_root}/<adjective>/seed_*/<scene>/scene_pos_*.json")


# if __name__ == "__main__":
#     main()

from __future__ import annotations
# =============================================================================
# generate_scenes.py  --  generate SAGE scenes exactly like the probed pairs,
# but with NO activation capture and NO refinement (first placement only).
#
# Structure produced:
#   save_root/<adjective_tag>/seed_<i>/<rtype>_<name>/scene_pos_*.json
#
#   for each benchmark (baseline)
#     for each adjective (appended to the environment note)
#       for each of N fixed seeds (SAME seed list reused for every setup)
#         -> one generated scene
#
# The model is loaded ONCE; all adjectives x seeds run in this single process.
# Generation faithfully mirrors probe_backend (same enable_thinking=False chat
# template, same DET_SEED+call_counter torch seeding, same do_sample/temp/top_p),
# so the scenes match what you were probing -- minus the hidden-state pass.
#
# Heavy imports (Heuristic_wo_dspy, gen_backend, torch) are deferred so --dry-run
# can print the full plan without a model or server.
# =============================================================================

import argparse
import glob
import json
import os
import re


# ----------------------------------------------------------------------------
# benchmark parsing (verbatim behaviour from adv_common)
# ----------------------------------------------------------------------------
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


def sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "", s.strip().replace(" ", "_")) or "x"


# ----------------------------------------------------------------------------
# decode config + seeding (mirrors adv_common force_* / seed_everything, but
# backed by gen_backend instead of probe_backend)
# ----------------------------------------------------------------------------
_RUN_SEED = None
_SEEDCTR = {"n": 0}


def configure_decode(decode, temperature, max_tokens, base_seed):
    import Heuristic_wo_dspy as H
    import gen_backend as G
    inner = G.gen_complete
    cap = max_tokens
    if decode == "greedy":
        def wrap(prompt, images=None, system=None, temperature=0.5,
                 max_tokens=None, force_json=True):
            return inner(prompt, images, system, 0.0, cap, force_json)
        G.set_det_seed(None)
        print(f"[gen] GREEDY decoding (temp=0), max_tokens={cap}")
    else:
        temp = temperature
        def wrap(prompt, images=None, system=None, temperature=0.5,
                 max_tokens=None, force_json=True):
            return inner(prompt, images, system, temp, cap, force_json)
        G.set_det_seed(base_seed)
        print(f"[gen] deterministic SAMPLING (temp={temp}, per-call seed base={base_seed}), "
              f"max_tokens={cap}")
    H.llm_complete = wrap


def force_deterministic_seed_text():
    """Make the 'Random seed: N' prompt text immune to global-RNG stream desync
    (the run-0 divergence). Counter-based, derived from (_RUN_SEED, call#)."""
    import utils_copy_img
    if getattr(utils_copy_img.generate_random_seed, "_det_patched", False):
        return

    def det_seed():
        _SEEDCTR["n"] += 1
        base = _RUN_SEED if _RUN_SEED is not None else 0
        return (base * 7919 + _SEEDCTR["n"] * 104729) % 10000001

    det_seed._det_patched = True
    utils_copy_img.generate_random_seed = det_seed


def seed_everything(s):
    import random
    import numpy as np
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
    global _RUN_SEED
    _RUN_SEED = s
    import gen_backend as G
    G.set_det_seed(s)


# ----------------------------------------------------------------------------
# one pass over all specs for the current (adjective, seed)
# ----------------------------------------------------------------------------
def run_once(specs, adv_note, save_dir_root, resume=True, reset_seed_per_scene=False,
             placement_n=5):
    import Heuristic_wo_dspy as H
    import utils_copy_img
    import gen_backend as G

    G.reset_call_counter()
    force_deterministic_seed_text()
    _SEEDCTR["n"] = 0
    if _RUN_SEED is not None:
        seed_everything(_RUN_SEED)

    for name, task, assets, dim_x, dim_y in specs:
        if reset_seed_per_scene:
            # per-scene isolation: scene k under seed i gets identical initial RNG
            # regardless of adjective or other scenes (cleaner per-scene counterfactual,
            # but NOT how probe_backend behaved -- it reset once per run).
            G.reset_call_counter()
            _SEEDCTR["n"] = 0
            if _RUN_SEED is not None:
                seed_everything(_RUN_SEED)

        instruction = task + (adv_note or "")   # adjective appended to the env note
        save_dir = os.path.join(save_dir_root, name)
        os.makedirs(save_dir, exist_ok=True)

        dx, dy = utils_copy_img.extract_movement_from_note(task)   # geometry from baseline task
        pos = (0 + dx * utils_copy_img.STEP_SIZE, 0 + dy * utils_copy_img.STEP_SIZE)
        if resume and utils_copy_img.load_existing_scenegraph(pos, save_dir=save_dir):
            print(f"    [skip] {name}: scene exists")
            continue

        planner = H.ScenePlanner(save_dir=save_dir)
        planner.manager.N = placement_n   # best-of-N candidates for first placement
        planner.forward(position=pos, prev_scenegraphs=[], instruction=instruction,
                        objects_list=assets, dim_x=dim_x, dim_y=dim_y, dx=dx, dy=dy,
                        orientation=False, scale=False)   # first placement only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-glob", required=True)
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--adjectives", required=True,
                    help="comma-separated, e.g. 'messy,cluttered,minimalist,cozy,spacious'")
    ap.add_argument("--adj-template", default=" The room is {adj}.",
                    help="how each adjective is appended to the env note; must contain {adj}. "
                         "For a raw append use ' {adj}'.")
    ap.add_argument("--include-baseline", action="store_true",
                    help="also generate a no-adjective run (tag 'baseline')")
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--base-seed", type=int, default=1234)
    ap.add_argument("--seed-stride", type=int, default=10000,
                    help="seed i = base_seed + i*stride; the SAME list is reused for every adjective")
    ap.add_argument("--decode", choices=["sample", "greedy"], default="sample",
                    help="sample = seeded sampling (needed for seed diversity); greedy = temp 0 "
                         "(WARNING: all seeds collapse to one scene under greedy)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--model-path", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--reset-seed-per-scene", action="store_true",
                    help="reset the per-call seed counter at every scene (per-scene RNG isolation). "
                         "Default off = faithful to probe_backend (reset once per seed-run).")
    ap.add_argument("--placement-n", type=int, default=5,
                    help="ObjectPlacementManager best-of-N candidates per scene. Default 5 = "
                         "faithful to the probed runs (feedback re-prompt loop, keep best, early "
                         "stop at threshold). Set 1 for a single-shot placement: one LLM call, no "
                         "feedback loop, no scoring/early-stop. N<5 makes scenes diverge from what "
                         "you probed.")
    ap.add_argument("--max-benchmarks", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="regenerate even if a scene_pos_*.json already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full plan (no model, no server) and exit")
    args = ap.parse_args()

    specs = list_specs(args.benchmark_glob, args.max_benchmarks)
    if "{adj}" not in args.adj_template:
        raise SystemExit("--adj-template must contain {adj}")

    adjectives = [a.strip() for a in args.adjectives.split(",") if a.strip()]
    conditions = ([("", "baseline")] if args.include_baseline else []) + \
                 [(a, sanitize_tag(a)) for a in adjectives]

    seeds = [args.base_seed + i * args.seed_stride for i in range(args.num_seeds)]

    total = len(specs) * len(conditions) * len(seeds)
    print(f"[gen] benchmarks={len(specs)}  conditions={len(conditions)}  "
          f"seeds={len(seeds)}  placement_n={args.placement_n}  -> {total} scenes")
    print(f"[gen] seeds (same for every condition): {seeds if len(seeds) <= 10 else seeds[:5] + ['...'] + seeds[-2:]}")
    if args.decode == "greedy" and len(seeds) > 1:
        print("[gen] WARNING: --decode greedy is argmax -> all seeds produce the SAME scene. "
              "Use --decode sample for a real 50-seed sweep.")
    if args.placement_n != 5:
        print(f"[gen] NOTE: placement_n={args.placement_n} (probed runs used 5); scenes will "
              f"differ from what you probed.")

    if args.dry_run:
        for adj, tag in conditions:
            note = args.adj_template.format(adj=adj) if adj else ""
            print(f"\n=== condition '{tag}'  adj_note={note!r} ===")
            for i, s in enumerate(seeds):
                root = os.path.join(args.save_root, tag, f"seed_{i:03d}")
                for name, task, assets, dim_x, dim_y in specs:
                    instr = task + note
                    print(f"  seed_{i:03d}(={s}) {os.path.join(root, name)}")
                    print(f"      instruction: {instr!r}")
                if len(seeds) > 3 and i == 2:
                    print(f"  ... ({len(seeds) - 3} more seeds, same layout)")
                    break
        print("\n[gen] dry-run only; no scenes generated.")
        return

    # ---- real run: load model once, then loop ----
    import gen_backend as G
    G.init_gen(model_path=args.model_path)
    configure_decode(args.decode, args.temperature, args.max_tokens, args.base_seed)

    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, "seeds.json"), "w") as f:
        json.dump({"seeds": seeds, "base_seed": args.base_seed,
                   "seed_stride": args.seed_stride}, f, indent=2)

    for adj, tag in conditions:
        note = args.adj_template.format(adj=adj) if adj else ""
        print(f"\n[gen] ################ CONDITION '{tag}'  adj_note={note!r} ################")
        for i, s in enumerate(seeds):
            set_repeat_seed(s)
            root = os.path.join(args.save_root, tag, f"seed_{i:03d}")
            print(f"[gen] --- {tag} seed_{i:03d} (seed={s}) ---")
            run_once(specs, adv_note=note, save_dir_root=root,
                     resume=not args.no_resume,
                     reset_seed_per_scene=args.reset_seed_per_scene,
                     placement_n=args.placement_n)

    print(f"\n[gen] done -> {args.save_root}/<adjective>/seed_*/<scene>/scene_pos_*.json")


if __name__ == "__main__":
    main()