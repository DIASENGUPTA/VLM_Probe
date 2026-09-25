# #!/usr/bin/env python3
# # =============================================================================
# # adv_token_geometry.py -- is the effect TOKEN-SPECIFIC, or just "a clause was added"?
# # -----------------------------------------------------------------------------
# # L2 measures the MAGNITUDE of the activation shift (||x_adv - x_base||). That can
# # never distinguish two notes: 'India' and 'Ethiopia' could shift the state by the
# # same amount in completely ORTHOGONAL directions and look identical.
# #
# # This tool measures the DIRECTION. For each condition c and seed r define
# #
# #     delta(c, r) = x_adversarial(c, r) - x_baseline(r)        [per layer, position]
# #
# # and compare directions via cosine similarity:
# #
# #   WITHIN(c)      = mean cos( delta(c,r), delta(c,s) ), r != s
# #                    -> does this note push the SAME way regardless of the scene?
# #   BETWEEN(c, c') = mean cos( delta(c,r), delta(c',r) ), same seed r
# #                    -> do two DIFFERENT notes push the same way?
# #
# # Verdict:
# #   WITHIN >> BETWEEN  -> each token has its OWN consistent direction. The model
# #                         encodes WHICH clause, not merely THAT a clause exists.
# #                         This is the interesting, publishable outcome.
# #   WITHIN ~ BETWEEN, both high
# #                      -> ONE shared 'an extra clause was appended' direction. The
# #                         notes are interchangeable; nothing is India-specific.
# #   WITHIN ~ 0         -> no stable direction even within a condition; the shift is
# #                         scene-specific noise. Nothing to localize.
# #
# # Chance level: in H dimensions random vectors have |cos| ~ 1/sqrt(H). For H=5120
# # that is ~0.014, so |cos| below ~0.05 is indistinguishable from orthogonal.
# #
# # REQUIREMENT: every condition must be generated with the SAME --base-seed, so the
# # baselines are identical and delta vectors are comparable. The pipeline is
# # deterministic, so this holds automatically.
# #
# #   python adv_generate_pairs.py --save-root adv_india2   --adv-note ' This is a bookstore in India.'    --repeats 5 --base-seed 1234
# #   python adv_generate_pairs.py --save-root adv_ethiopia --adv-note ' This is a bookstore in Ethiopia.' --repeats 5 --base-seed 1234
# #   python adv_generate_pairs.py --save-root adv_ctrl     --adv-note ' This is a well-lit bookstore.'    --repeats 5 --base-seed 1234
# #
# #   python adv_token_geometry.py \
# #       --conditions india=adv_india2 ethiopia=adv_ethiopia neutral=adv_ctrl \
# #       --position delta_end
# # =============================================================================
# from __future__ import annotations

# import argparse
# import glob
# import itertools
# import json
# import os
# import re
# from collections import defaultdict

# import numpy as np

# import adv_common as C


# def load_condition(d):
#     """Return lists of baseline/adversarial snapshots ordered by repeat index."""
#     def grab(pat):
#         paths = sorted(glob.glob(os.path.join(d, pat)),
#                        key=lambda x: int(re.search(r"_r(\d+)\.pkl$", x).group(1)))
#         return [C.load_snapshot(p) for p in paths]
#     return grab("baseline_r*.pkl"), grab("adversarial_r*.pkl")


# def delta_vectors(base_recs, adv_recs, position, layer):
#     """{(scene, iteration): x_adv - x_base} for one repeat."""
#     _, ppairs, _ = C.align(base_recs, adv_recs)
#     out = {}
#     for a, b in ppairs:
#         key = (str(a["ctx"].get("scene_id")), int(a["ctx"].get("iteration") or -1))
#         Aa = a["activations"].get(position, {})
#         Ab = b["activations"].get(position, {})
#         if layer in Aa and layer in Ab:
#             out[key] = (np.asarray(Ab[layer], np.float64)
#                         - np.asarray(Aa[layer], np.float64))
#     return out


# def _cos(u, v):
#     nu, nv = np.linalg.norm(u), np.linalg.norm(v)
#     if nu < 1e-12 or nv < 1e-12:
#         return float("nan")
#     return float(u @ v / (nu * nv))


# def baselines_identical(conds):
#     """All conditions must share baselines (same --base-seed) or deltas are not
#     comparable. Verify rather than assume."""
#     names = list(conds)
#     ref = conds[names[0]]["base"]
#     for n in names[1:]:
#         other = conds[n]["base"]
#         if len(other) != len(ref):
#             return False, f"{n} has {len(other)} repeats vs {names[0]} has {len(ref)}"
#         for r, (A, B) in enumerate(zip(ref, other)):
#             pa = [x.get("prompt") for x in A]
#             pb = [x.get("prompt") for x in B]
#             if pa != pb:
#                 return False, (f"baseline prompts differ at repeat {r} between "
#                                f"{names[0]!r} and {n!r} -- regenerate with the SAME "
#                                f"--base-seed for every condition")
#     return True, "all conditions share identical baselines"


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--conditions", nargs="+", required=True,
#                     help="name=dir pairs, e.g. india=adv_india2 ethiopia=adv_ethiopia")
#     ap.add_argument("--position", default="delta_end", choices=["prompt_end", "delta_end"])
#     ap.add_argument("--layers", type=int, nargs="*", default=None,
#                     help="default: every layer present in the snapshots")
#     args = ap.parse_args()

#     conds = {}
#     for spec in args.conditions:
#         if "=" not in spec:
#             raise SystemExit(f"bad --conditions entry {spec!r}; use name=dir")
#         name, d = spec.split("=", 1)
#         b, a = load_condition(d)
#         if not b or not a:
#             raise SystemExit(f"no snapshots in {d}")
#         conds[name] = {"dir": d, "base": b, "adv": a}
#         print(f"[geo] {name:>10}: {len(b)} baseline / {len(a)} adversarial run(s)  ({d})")

#     ok, msg = baselines_identical(conds)
#     print(f"[geo] baseline check: {msg}")
#     if not ok:
#         raise SystemExit("[abort] conditions do not share baselines; delta vectors are "
#                          "not comparable.")

#     names = list(conds)
#     probe = next(iter(conds.values()))
#     any_acts = probe["base"][0][0].get("activations", {}).get(args.position, {})
#     layers = args.layers or sorted(any_acts.keys())
#     if not layers:
#         raise SystemExit(f"no activations at position {args.position!r} in the snapshots.")

#     H = len(next(iter(any_acts.values()))) if any_acts else 0
#     chance = 1.0 / np.sqrt(H) if H else float("nan")
#     print(f"[geo] layers={layers}  H={H}  chance |cos| ~ {chance:.3f}\n")

#     summary = {}
#     for layer in layers:
#         # delta[name][repeat] = {(scene,iter): vec}
#         delta = {n: [delta_vectors(conds[n]["base"][r], conds[n]["adv"][r],
#                                    args.position, layer)
#                      for r in range(len(conds[n]["adv"]))] for n in names}

#         within = {}
#         for n in names:
#             vals = []
#             R = len(delta[n])
#             for r, s in itertools.combinations(range(R), 2):
#                 shared = set(delta[n][r]) & set(delta[n][s])
#                 for k in shared:
#                     vals.append(_cos(delta[n][r][k], delta[n][s][k]))
#             vals = [v for v in vals if v == v]
#             within[n] = (float(np.mean(vals)) if vals else float("nan"), len(vals))

#         between = {}
#         for n1, n2 in itertools.combinations(names, 2):
#             vals = []
#             R = min(len(delta[n1]), len(delta[n2]))
#             for r in range(R):
#                 shared = set(delta[n1][r]) & set(delta[n2][r])
#                 for k in shared:
#                     vals.append(_cos(delta[n1][r][k], delta[n2][r][k]))
#             vals = [v for v in vals if v == v]
#             between[(n1, n2)] = (float(np.mean(vals)) if vals else float("nan"), len(vals))

#         summary[str(layer)] = {
#             "within": {n: within[n][0] for n in names},
#             "between": {f"{a}~{b}": v[0] for (a, b), v in between.items()},
#         }

#         print(f"===== layer {layer} ({args.position}) =====")
#         print("  WITHIN-condition (same note, different seeds) -- is the direction stable?")
#         for n in names:
#             m, c = within[n]
#             print(f"    {n:>10}: cos = {m:+.3f}   (n={c})")
#         print("  BETWEEN-condition (different notes, same seed) -- do they push the same way?")
#         for (n1, n2), (m, c) in between.items():
#             print(f"    {n1:>10} ~ {n2:<10}: cos = {m:+.3f}   (n={c})")

#         wmean = np.nanmean([within[n][0] for n in names])
#         bmean = np.nanmean([v[0] for v in between.values()]) if between else float("nan")
#         print(f"  mean WITHIN = {wmean:+.3f}   mean BETWEEN = {bmean:+.3f}   "
#               f"gap = {wmean - bmean:+.3f}")
#         print("  -> ", end="")
#         if wmean != wmean:
#             print("n/a")
#         elif abs(wmean) < max(3 * chance, 0.05):
#             print("NO STABLE DIRECTION even within a condition: the shift is scene-specific "
#                   "noise. Nothing token-specific to find here.")
#         elif bmean == bmean and (wmean - bmean) > 0.15:
#             print("TOKEN-SPECIFIC: each note has its own consistent direction, and different "
#                   "notes push DIFFERENTLY. The model encodes WHICH clause, not just THAT one "
#                   "exists. <- the interesting result")
#         elif bmean == bmean and abs(wmean - bmean) <= 0.15:
#             print("ONE SHARED DIRECTION: notes are interchangeable -- the geometry says "
#                   "'a clause was appended', not 'India'. No token-specific structure.")
#         else:
#             print("mixed / inconclusive at this layer.")
#         print()

#     out = "token_geometry_%s.json" % args.position
#     json.dump(summary, open(out, "w"), indent=2)
#     print(f"[geo] -> {out}")
#     print("\nCAVEATS: cosine is computed on raw residual-stream deltas -- a large shared "
#           "component (e.g. sequence-length or position effects from ANY appended text) can "
#           "inflate BOTH within and between. If both are high, subtract the neutral-note "
#           "delta first and re-run: that removes the generic 'a clause exists' component and "
#           "leaves the semantic residue.\n")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# =============================================================================
# adv_token_geometry.py -- is the effect TOKEN-SPECIFIC, or just "a clause was added"?
# -----------------------------------------------------------------------------
# L2 measures the MAGNITUDE of the activation shift (||x_adv - x_base||). That can
# never distinguish two notes: 'India' and 'Ethiopia' could shift the state by the
# same amount in completely ORTHOGONAL directions and look identical.
#
# This tool measures the DIRECTION. For each condition c and seed r define
#
#     delta(c, r) = x_adversarial(c, r) - x_baseline(r)        [per layer, position]
#
# and compare directions via cosine similarity:
#
#   WITHIN(c)      = mean cos( delta(c,r), delta(c,s) ), r != s
#                    -> does this note push the SAME way regardless of the scene?
#   BETWEEN(c, c') = mean cos( delta(c,r), delta(c',r) ), same seed r
#                    -> do two DIFFERENT notes push the same way?
#
# Verdict:
#   WITHIN >> BETWEEN  -> each token has its OWN consistent direction. The model
#                         encodes WHICH clause, not merely THAT a clause exists.
#                         This is the interesting, publishable outcome.
#   WITHIN ~ BETWEEN, both high
#                      -> ONE shared 'an extra clause was appended' direction. The
#                         notes are interchangeable; nothing is India-specific.
#   WITHIN ~ 0         -> no stable direction even within a condition; the shift is
#                         scene-specific noise. Nothing to localize.
#
# Chance level: in H dimensions random vectors have |cos| ~ 1/sqrt(H). For H=5120
# that is ~0.014, so |cos| below ~0.05 is indistinguishable from orthogonal.
#
# REQUIREMENT: every condition must be generated with the SAME --base-seed, so the
# baselines are identical and delta vectors are comparable. The pipeline is
# deterministic, so this holds automatically.
#
#   python adv_generate_pairs.py --save-root adv_india2   --adv-note ' This is a bookstore in India.'    --repeats 5 --base-seed 1234
#   python adv_generate_pairs.py --save-root adv_ethiopia --adv-note ' This is a bookstore in Ethiopia.' --repeats 5 --base-seed 1234
#   python adv_generate_pairs.py --save-root adv_ctrl     --adv-note ' This is a well-lit bookstore.'    --repeats 5 --base-seed 1234
#
#   python adv_token_geometry.py \
#       --conditions india=adv_india2 ethiopia=adv_ethiopia neutral=adv_ctrl \
#       --position delta_end
# =============================================================================
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

import adv_common as C


def load_condition(d):
    """Return lists of baseline/adversarial snapshots ordered by repeat index."""
    def grab(pat):
        paths = sorted(glob.glob(os.path.join(d, pat)),
                       key=lambda x: int(re.search(r"_r(\d+)\.pkl$", x).group(1)))
        return [C.load_snapshot(p) for p in paths]
    return grab("baseline_r*.pkl"), grab("adversarial_r*.pkl")


def delta_vectors(base_recs, adv_recs, position, layer):
    """{(scene, iteration): x_adv - x_base} for one repeat."""
    _, ppairs, _ = C.align(base_recs, adv_recs)
    out = {}
    for a, b in ppairs:
        key = (str(a["ctx"].get("scene_id")), int(a["ctx"].get("iteration") or -1))
        Aa = a["activations"].get(position, {})
        Ab = b["activations"].get(position, {})
        if layer in Aa and layer in Ab:
            out[key] = (np.asarray(Ab[layer], np.float64)
                        - np.asarray(Aa[layer], np.float64))
    return out


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(u @ v / (nu * nv))


def _seed_seq(recs):
    """The 'Random seed: N' values actually injected into each prompt."""
    out = []
    for r in recs:
        m = re.search(r"Random seed:\s*(\d+)", r.get("prompt") or "")
        out.append(int(m.group(1)) if m else None)
    return out


def _strip_seed(p):
    return re.sub(r"Random seed:\s*\d+", "Random seed: <N>", p or "")


def baselines_identical(conds):
    """All conditions must share baselines (same --base-seed AND same code version)
    or deltas are not comparable: delta(c,r) = x_adv(c,r) - x_base(r) is measured
    relative to that condition's own baseline. Different baselines = different
    reference points = incomparable vectors.

    Reports WHY they differ, which determines the fix:
      * seed VALUES differ            -> different --base-seed, or one condition was
                                         generated WITHOUT force_deterministic_seed_text
                                         (counter-based seeds). Regenerate.
      * seeds match, CONTENT differs  -> the pipeline/prompt builder changed between
                                         runs (different code version or working dir).
                                         Regenerate both with the same checkout.
    """
    names = list(conds)
    ref_name = names[0]
    ref = conds[ref_name]["base"]
    for n in names[1:]:
        other = conds[n]["base"]
        if len(other) != len(ref):
            return False, f"{n} has {len(other)} repeats vs {ref_name} has {len(ref)}"
        for r, (A, B) in enumerate(zip(ref, other)):
            pa = [x.get("prompt") for x in A]
            pb = [x.get("prompt") for x in B]
            if pa == pb:
                continue
            sa, sb = _seed_seq(A), _seed_seq(B)
            k = next((i for i in range(min(len(pa), len(pb))) if pa[i] != pb[i]), 0)
            lines = [f"baseline prompts differ at repeat {r}, first at call {k}, "
                     f"between {ref_name!r} and {n!r}"]
            lines.append(f"    seed seq {ref_name:>10}: {sa}")
            lines.append(f"    seed seq {n:>10}: {sb}")
            if sa != sb:
                lines.append("    -> SEED VALUES DIFFER. Either the two runs used a "
                             "different --base-seed, or one was generated WITHOUT the "
                             "counter-based generate_random_seed patch "
                             "(adv_common.force_deterministic_seed_text). Look for this "
                             "line in the generation log:")
                lines.append("       '[adv] patched generate_random_seed -> deterministic "
                             "counter-based'")
                lines.append("    FIX: regenerate every condition with the SAME "
                             "--base-seed and the SAME adv_common.py.")
            else:
                a_s, b_s = _strip_seed(pa[k]), _strip_seed(pb[k])
                lines.append("    -> seeds MATCH but prompt CONTENT differs: the pipeline "
                             "or prompt builder changed between the two runs (different "
                             "code version / working directory).")
                lines.append(f"    {ref_name} call {k} (seed masked): {a_s[:160]!r}")
                lines.append(f"    {n} call {k} (seed masked): {b_s[:160]!r}")
                lines.append("    FIX: regenerate both from the same checkout.")
            return False, "\n".join(lines)
    return True, "all conditions share identical baselines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="name=dir pairs, e.g. india=adv_india2 ethiopia=adv_ethiopia")
    ap.add_argument("--position", default="delta_end", choices=["prompt_end", "delta_end"])
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="default: every layer present in the snapshots")
    ap.add_argument("--within-only", action="store_true",
                    help="report only WITHIN-condition cosine (valid even when the "
                         "conditions do not share baselines). BETWEEN is suppressed.")
    args = ap.parse_args()

    conds = {}
    for spec in args.conditions:
        if "=" not in spec:
            raise SystemExit(f"bad --conditions entry {spec!r}; use name=dir")
        name, d = spec.split("=", 1)
        b, a = load_condition(d)
        if not b or not a:
            raise SystemExit(f"no snapshots in {d}")
        conds[name] = {"dir": d, "base": b, "adv": a}
        print(f"[geo] {name:>10}: {len(b)} baseline / {len(a)} adversarial run(s)  ({d})")

    ok, msg = baselines_identical(conds)
    print(f"[geo] baseline check: {msg}")
    if not ok:
        print("\n[geo] NOTE: WITHIN-condition cosine only uses each condition's OWN "
              "baselines, so it is STILL VALID. BETWEEN-condition cosine compares "
              "delta vectors measured against DIFFERENT reference points and is NOT.\n"
              "      Re-run with --within-only to get the (valid) within-condition "
              "answer from this data, or regenerate to get the full comparison.")
        if not args.within_only:
            raise SystemExit("[abort] conditions do not share baselines; BETWEEN is not "
                             "comparable. Use --within-only, or regenerate.")
        print("[geo] --within-only: reporting WITHIN only; BETWEEN suppressed.\n")

    names = list(conds)
    probe = next(iter(conds.values()))
    any_acts = probe["base"][0][0].get("activations", {}).get(args.position, {})
    layers = args.layers or sorted(any_acts.keys())
    if not layers:
        raise SystemExit(f"no activations at position {args.position!r} in the snapshots.")

    H = len(next(iter(any_acts.values()))) if any_acts else 0
    chance = 1.0 / np.sqrt(H) if H else float("nan")
    print(f"[geo] layers={layers}  H={H}  chance |cos| ~ {chance:.3f}\n")

    summary = {}
    for layer in layers:
        # delta[name][repeat] = {(scene,iter): vec}
        delta = {n: [delta_vectors(conds[n]["base"][r], conds[n]["adv"][r],
                                   args.position, layer)
                     for r in range(len(conds[n]["adv"]))] for n in names}

        within = {}
        for n in names:
            vals = []
            R = len(delta[n])
            for r, s in itertools.combinations(range(R), 2):
                shared = set(delta[n][r]) & set(delta[n][s])
                for k in shared:
                    vals.append(_cos(delta[n][r][k], delta[n][s][k]))
            vals = [v for v in vals if v == v]
            within[n] = (float(np.mean(vals)) if vals else float("nan"), len(vals))

        between = {}
        if not args.within_only:
            for n1, n2 in itertools.combinations(names, 2):
                vals = []
                R = min(len(delta[n1]), len(delta[n2]))
                for r in range(R):
                    shared = set(delta[n1][r]) & set(delta[n2][r])
                    for k in shared:
                        vals.append(_cos(delta[n1][r][k], delta[n2][r][k]))
                vals = [v for v in vals if v == v]
                between[(n1, n2)] = (float(np.mean(vals)) if vals else float("nan"), len(vals))

        summary[str(layer)] = {
            "within": {n: within[n][0] for n in names},
            "between": {f"{a}~{b}": v[0] for (a, b), v in between.items()},
        }

        print(f"===== layer {layer} ({args.position}) =====")
        print("  WITHIN-condition (same note, different seeds) -- is the direction stable?")
        for n in names:
            m, c = within[n]
            print(f"    {n:>10}: cos = {m:+.3f}   (n={c})")
        print("  BETWEEN-condition (different notes, same seed) -- do they push the same way?")
        if args.within_only:
            print("    (suppressed: conditions do not share baselines)")
        for (n1, n2), (m, c) in between.items():
            print(f"    {n1:>10} ~ {n2:<10}: cos = {m:+.3f}   (n={c})")

        wmean = np.nanmean([within[n][0] for n in names])
        bmean = np.nanmean([v[0] for v in between.values()]) if between else float("nan")
        if between:
            print(f"  mean WITHIN = {wmean:+.3f}   mean BETWEEN = {bmean:+.3f}   "
                  f"gap = {wmean - bmean:+.3f}")
        else:
            print(f"  mean WITHIN = {wmean:+.3f}   (BETWEEN unavailable)")
        print("  -> ", end="")
        if wmean != wmean:
            print("n/a")
        elif abs(wmean) < max(3 * chance, 0.05):
            print("NO STABLE DIRECTION even within a condition: the shift is scene-specific "
                  "noise. Nothing token-specific to find here -- and this conclusion does "
                  "NOT depend on the baseline mismatch.")
        elif not between:
            print(f"STABLE WITHIN-CONDITION DIRECTION (cos={wmean:+.3f} >> chance "
                  f"{chance:.3f}): each note pushes consistently across scenes. Whether the "
                  f"notes differ FROM EACH OTHER needs BETWEEN -- regenerate with a shared "
                  f"--base-seed.")
        elif (wmean - bmean) > 0.15:
            print("TOKEN-SPECIFIC: each note has its own consistent direction, and different "
                  "notes push DIFFERENTLY. The model encodes WHICH clause, not just THAT one "
                  "exists. <- the interesting result")
        elif abs(wmean - bmean) <= 0.15:
            print("ONE SHARED DIRECTION: notes are interchangeable -- the geometry says "
                  "'a clause was appended', not 'India'. No token-specific structure.")
        else:
            print("mixed / inconclusive at this layer.")
        print()

    out = "token_geometry_%s.json" % args.position
    json.dump(summary, open(out, "w"), indent=2)
    print(f"[geo] -> {out}")
    print("\nCAVEATS: cosine is computed on raw residual-stream deltas -- a large shared "
          "component (e.g. sequence-length or position effects from ANY appended text) can "
          "inflate BOTH within and between. If both are high, subtract the neutral-note "
          "delta first and re-run: that removes the generic 'a clause exists' component and "
          "leaves the semantic residue.\n")


if __name__ == "__main__":
    main()