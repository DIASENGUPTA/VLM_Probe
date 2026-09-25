# #!/usr/bin/env python3
# # =============================================================================
# # adv_inspect.py -- WHAT actually changed? (labels, not metrics)
# # -----------------------------------------------------------------------------
# # adv_l1_variance tells you HOW MUCH changed. This tells you WHAT changed, so you
# # can tell a real effect from a naming artifact.
# #
# # obj_set_diff = 1.000 ("completely disjoint object sets") has two very different
# # explanations:
# #   (a) REAL: the note genuinely swapped the furniture.
# #   (b) ARTIFACT: same furniture, different instance numbering or wording
# #       ('wooden_table_1' vs 'table_1'; 'armchair_1..4' vs 'armchair_5..8').
# # Only looking at the labels distinguishes them.
# #
# #   python adv_inspect.py --dir adv_india2
# #   python adv_inspect.py --dir adv_india2 --repeat 0
# # =============================================================================
# from __future__ import annotations

# import argparse
# import glob
# import os
# import re

# import adv_common as C
# from adv_l1_variance import final_scene_state, _canon, load_runs


# def show(base, adv, repeat, scene_filter=None):
#     sa, sb = final_scene_state(base), final_scene_state(adv)
#     scenes = sorted(set(sa) & set(sb))
#     for s in scenes:
#         if scene_filter and scene_filter not in s:
#             continue
#         A, B = sa[s], sb[s]
#         la, lb = sorted(A["objects"]), sorted(B["objects"])
#         ta = sorted({_canon(x) for x in la})
#         tb = sorted({_canon(x) for x in lb})
#         print(f"\n=== repeat {repeat} | scene {s} ===")
#         print(f"  BASELINE    ({len(la):>2} objs): {', '.join(la) or '(none)'}")
#         print(f"  ADVERSARIAL ({len(lb):>2} objs): {', '.join(lb) or '(none)'}")
#         print(f"  -- instance-level: shared={len(set(la)&set(lb))}  "
#               f"only-baseline={sorted(set(la)-set(lb))}  only-adv={sorted(set(lb)-set(la))}")
#         print(f"  TYPES base: {', '.join(ta)}")
#         print(f"  TYPES adv : {', '.join(tb)}")
#         shared_t = set(ta) & set(tb)
#         print(f"  -- type-level: shared={len(shared_t)}  "
#               f"only-baseline={sorted(set(ta)-set(tb))}  only-adv={sorted(set(tb)-set(ta))}")
#         if set(la) != set(lb) and set(ta) == set(tb):
#             print("  >>> VERDICT: labels differ but TYPES ARE IDENTICAL -> obj_set_diff is a "
#                   "NAMING ARTIFACT for this scene, not a furniture change. Use --by-type.")
#         elif set(ta) != set(tb):
#             print("  >>> VERDICT: the TYPES themselves differ -> a REAL change in what was placed.")
#         else:
#             print("  >>> VERDICT: identical object sets; any effect is positional only.")
#         # positions of shared labels
#         shared = sorted(set(la) & set(lb))
#         if shared:
#             print("  shared-object positions (baseline -> adversarial):")
#             for l in shared[:12]:
#                 x0, y0 = A["objects"][l]
#                 x1, y1 = B["objects"][l]
#                 d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
#                 print(f"    {l:<22} ({x0:6.2f},{y0:6.2f}) -> ({x1:6.2f},{y1:6.2f})   |d|={d:.2f}")
#         print(f"  legality: baseline={A['legality']}  adversarial={B['legality']}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dir", required=True)
#     ap.add_argument("--repeat", type=int, default=None, help="default: all repeats")
#     ap.add_argument("--scene", default=None, help="substring filter on scene id")
#     args = ap.parse_args()

#     base, adv = load_runs(args.dir)
#     n = min(len(base), len(adv))
#     reps = [args.repeat] if args.repeat is not None else range(n)
#     for r in reps:
#         if r >= n:
#             raise SystemExit(f"repeat {r} not found (have {n})")
#         show(base[r], adv[r], r, args.scene)
#     print()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# =============================================================================
# adv_inspect.py -- WHAT actually changed? (labels, not metrics)
# -----------------------------------------------------------------------------
# adv_l1_variance tells you HOW MUCH changed. This tells you WHAT changed, so you
# can tell a real effect from a naming artifact.
#
# obj_set_diff = 1.000 ("completely disjoint object sets") has two very different
# explanations:
#   (a) REAL: the note genuinely swapped the furniture.
#   (b) ARTIFACT: same furniture, different instance numbering or wording
#       ('wooden_table_1' vs 'table_1'; 'armchair_1..4' vs 'armchair_5..8').
# Only looking at the labels distinguishes them.
#
#   python adv_inspect.py --dir adv_india2
#   python adv_inspect.py --dir adv_india2 --repeat 0
# =============================================================================
from __future__ import annotations

import argparse
import glob
import os
import re

import adv_common as C
from adv_l1_variance import final_scene_state, _canon, load_runs


def show(base, adv, repeat, scene_filter=None):
    from collections import Counter
    sa, sb = final_scene_state(base), final_scene_state(adv)
    scenes = sorted(set(sa) & set(sb))
    for s in scenes:
        if scene_filter and scene_filter not in s:
            continue
        A, B = sa[s], sb[s]
        la, lb = sorted(A["objects"]), sorted(B["objects"])
        ca, cb = Counter(_canon(x) for x in la), Counter(_canon(x) for x in lb)
        ta, tb = sorted(ca), sorted(cb)
        print(f"\n=== repeat {repeat} | scene {s} ===")
        print(f"  BASELINE    ({len(la):>2} objs): {', '.join(la) or '(none)'}")
        print(f"  ADVERSARIAL ({len(lb):>2} objs): {', '.join(lb) or '(none)'}")
        print(f"\n  canonical TYPE COUNTS (naming normalised):")
        for t in sorted(set(ta) | set(tb)):
            mark = "  <-- differs" if ca.get(t, 0) != cb.get(t, 0) else ""
            print(f"    {t:<20} baseline={ca.get(t,0):>2}   adversarial={cb.get(t,0):>2}{mark}")
        print(f"    {'TOTAL':<20} baseline={len(la):>2}   adversarial={len(lb):>2}")
        print("\n  >>> ", end="")
        if set(ta) != set(tb):
            print("the TYPES differ -> a REAL change in WHAT furniture was placed.")
        elif ca != cb:
            print("same types, different COUNTS -> the note changed HOW MANY objects were "
                  "placed (compliance with the requested asset counts), not what.")
        elif set(la) != set(lb):
            print("identical types AND counts; only the LABEL STRINGS differ -> naming churn, "
                  "not a scene change. Any real effect is positional.")
        else:
            print("identical object sets; any effect is positional only.")
        shared = sorted(set(la) & set(lb))
        if shared:
            print("\n  shared-object positions (baseline -> adversarial):")
            for l in shared[:12]:
                x0, y0 = A["objects"][l]
                x1, y1 = B["objects"][l]
                d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
                print(f"    {l:<22} ({x0:6.2f},{y0:6.2f}) -> ({x1:6.2f},{y1:6.2f})   |d|={d:.2f}")
        print(f"  legality: baseline={A['legality']}  adversarial={B['legality']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repeat", type=int, default=None, help="default: all repeats")
    ap.add_argument("--scene", default=None, help="substring filter on scene id")
    args = ap.parse_args()

    base, adv = load_runs(args.dir)
    n = min(len(base), len(adv))
    reps = [args.repeat] if args.repeat is not None else range(n)
    for r in reps:
        if r >= n:
            raise SystemExit(f"repeat {r} not found (have {n})")
        show(base[r], adv[r], r, args.scene)
    print()


if __name__ == "__main__":
    main()