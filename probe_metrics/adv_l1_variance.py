# # # #!/usr/bin/env python3
# # # # =============================================================================
# # # # adv_l1_variance.py -- does the token move the output MORE than the pipeline's
# # # #                       own run-to-run noise?
# # # # -----------------------------------------------------------------------------
# # # # WHY THIS EXISTS
# # # #   A single baseline-vs-adversarial pair cannot support any claim: reruns of the
# # # #   SAME baseline already differ (different object sets, different layouts). So we
# # # #   compare two DISTRIBUTIONS of differences:
# # # #
# # # #     WITHIN-baseline    : diff(baseline_r, baseline_s)     r<s   -> pure noise
# # # #     WITHIN-adversarial : diff(adv_r, adv_s)               r<s   -> pure noise
# # # #     BETWEEN            : diff(baseline_r, adv_s)          all   -> noise + token
# # # #
# # # #   If the token does nothing, BETWEEN is drawn from the same distribution as the
# # # #   WITHINs. A token effect = BETWEEN shifted ABOVE the within envelope.
# # # #   This is the distance-based (PERMANOVA-style) logic, and it is the only honest
# # # #   test when the pipeline is itself stochastic.
# # # #
# # # #   All metrics are DIFFERENCE measures on the FINAL COMMITTED scene (i.e. after
# # # #   the best-of-N refinement loop) -- higher = more different:
# # # #     obj_set_diff    1 - Jaccard of the final object-label sets
# # # #     n_obj_diff      |#objects_a - #objects_b|
# # # #     occ_hamming     fraction of occupancy-grid cells that differ
# # # #     centroid_shift  mean euclidean move of objects present in BOTH (metres)
# # # #     legality_flip   final legality differs (0/1)
# # # #
# # # # USAGE
# # # #   python adv_l1_variance.py --dir adv_out
# # # #   python adv_l1_variance.py --dir adv_out --metric obj_set_diff
# # # #
# # # # INTERPRETATION
# # # #   * within ~ 0  and between > 0  -> deterministic pipeline, real token effect.
# # # #   * within ~ between            -> token effect indistinguishable from noise.
# # # #   * within large                -> pipeline is stochastic; you need many repeats
# # # #                                    AND many scenes before any claim is possible.
# # # # =============================================================================
# # # from __future__ import annotations

# # # import argparse
# # # import glob
# # # import itertools
# # # import json
# # # import math
# # # import os
# # # import re

# # # import numpy as np

# # # import adv_common as C

# # # METRICS = ["obj_set_diff", "n_obj_diff", "occ_hamming", "centroid_shift", "legality_flip"]


# # # # ---- final (post-refinement) committed state per scene ----------------------
# # # def final_scene_state(recs):
# # #     """scene_id -> {objects:{label:(x,y)}, grid:[...], legality:int}
# # #     Uses ONLY records flagged committed=True, i.e. the best-of-N winner that the
# # #     pipeline actually commits and renders as 'initial'."""
# # #     out = {}
# # #     for r in recs:
# # #         lab = r.get("labels", {})
# # #         if not lab.get("committed"):
# # #             continue
# # #         sid = str(r.get("ctx", {}).get("scene_id"))
# # #         e = out.setdefault(sid, {"objects": {}, "grid": None, "legality": None})
# # #         e["objects"].update(C.parse_objs(r.get("output_text", "")))
# # #         e["grid"] = lab.get("occupancy_str")
# # #         e["legality"] = lab.get("legality")
# # #     return out


# # # def _canon(label: str) -> str:
# # #     """Drop instance suffixes so 'wooden_table_1' vs 'wooden_table_2' both count
# # #     as the requested asset type when comparing SETS of object types."""
# # #     s = str(label).lower().strip()
# # #     s = re.sub(r"_\d+$", "", s)
# # #     return s


# # # def pair_metrics(recs_a, recs_b, by_type=False):
# # #     """Mean over shared scenes of each difference metric. Higher = more different."""
# # #     sa, sb = final_scene_state(recs_a), final_scene_state(recs_b)
# # #     scenes = sorted(set(sa) & set(sb))
# # #     if not scenes:
# # #         return None
# # #     acc = {m: [] for m in METRICS}
# # #     for s in scenes:
# # #         A, B = sa[s], sb[s]
# # #         ka = {(_canon(k) if by_type else k) for k in A["objects"]}
# # #         kb = {(_canon(k) if by_type else k) for k in B["objects"]}
# # #         union = ka | kb
# # #         jac = (len(ka & kb) / len(union)) if union else 1.0
# # #         acc["obj_set_diff"].append(1.0 - jac)
# # #         acc["n_obj_diff"].append(abs(len(A["objects"]) - len(B["objects"])))

# # #         ga, gb = A["grid"], B["grid"]
# # #         if ga and gb and len(ga) == len(gb):
# # #             acc["occ_hamming"].append(sum(x != y for x, y in zip(ga, gb)) / len(ga))

# # #         shared = set(A["objects"]) & set(B["objects"])
# # #         if shared:
# # #             acc["centroid_shift"].append(float(np.mean([
# # #                 math.hypot(A["objects"][l][0] - B["objects"][l][0],
# # #                            A["objects"][l][1] - B["objects"][l][1]) for l in shared])))

# # #         if A["legality"] is not None and B["legality"] is not None:
# # #             acc["legality_flip"].append(float(A["legality"] != B["legality"]))

# # #     return {m: (float(np.mean(v)) if v else float("nan")) for m, v in acc.items()}


# # # def _mean_std(xs):
# # #     xs = [x for x in xs if x == x]
# # #     if not xs:
# # #         return float("nan"), float("nan"), 0
# # #     m = float(np.mean(xs))
# # #     return m, float(np.std(xs)), len(xs)


# # # def load_runs(d):
# # #     def grab(pat):
# # #         out = []
# # #         for p in sorted(glob.glob(os.path.join(d, pat)),
# # #                         key=lambda x: int(re.search(r"_r(\d+)\.pkl$", x).group(1))):
# # #             out.append(C.load_snapshot(p))
# # #         return out
# # #     return grab("baseline_r*.pkl"), grab("adversarial_r*.pkl")


# # # def main():
# # #     ap = argparse.ArgumentParser()
# # #     ap.add_argument("--dir", required=True)
# # #     ap.add_argument("--by-type", action="store_true",
# # #                     help="compare object TYPES (strip _1/_2 suffixes) instead of instances")
# # #     ap.add_argument("--metric", default=None, choices=METRICS)
# # #     args = ap.parse_args()

# # #     base, adv = load_runs(args.dir)
# # #     print(f"[var] loaded {len(base)} baseline run(s), {len(adv)} adversarial run(s)")
# # #     if len(base) < 2:
# # #         raise SystemExit(
# # #             "[abort] need >=2 baseline repeats to estimate the noise envelope. "
# # #             "Re-run adv_generate_pairs.py with --repeats 5 (or more).")

# # #     within_b, within_a, between = [], [], []
# # #     for i, j in itertools.combinations(range(len(base)), 2):
# # #         m = pair_metrics(base[i], base[j], args.by_type)
# # #         if m: within_b.append(m)
# # #     for i, j in itertools.combinations(range(len(adv)), 2):
# # #         m = pair_metrics(adv[i], adv[j], args.by_type)
# # #         if m: within_a.append(m)
# # #     for i in range(len(base)):
# # #         for j in range(len(adv)):
# # #             m = pair_metrics(base[i], adv[j], args.by_type)
# # #             if m: between.append(m)

# # #     mets = [args.metric] if args.metric else METRICS
# # #     print(f"\n{'metric':>15} | {'WITHIN-base':>18} | {'WITHIN-adv':>18} | "
# # #           f"{'BETWEEN':>18} | verdict")
# # #     print("-" * 100)
# # #     summary = {}
# # #     for m in mets:
# # #         wb, wbs, wbn = _mean_std([x[m] for x in within_b])
# # #         wa, was, wan = _mean_std([x[m] for x in within_a])
# # #         bt, bts, btn = _mean_std([x[m] for x in between])
# # #         # noise ceiling = worst within-condition mean + its spread
# # #         ceiling = max([v for v in (wb + wbs, wa + was) if v == v] or [float("nan")])
# # #         if bt != bt or ceiling != ceiling:
# # #             verdict = "n/a"
# # #         elif bt > ceiling:
# # #             verdict = "ABOVE noise"
# # #         else:
# # #             verdict = "within noise"
# # #         summary[m] = {"within_baseline": [wb, wbs, wbn], "within_adversarial": [wa, was, wan],
# # #                       "between": [bt, bts, btn], "verdict": verdict}
# # #         print(f"{m:>15} | {wb:>8.3f} ±{wbs:5.3f}({wbn}) | {wa:>8.3f} ±{was:5.3f}({wan}) | "
# # #               f"{bt:>8.3f} ±{bts:5.3f}({btn}) | {verdict}")
# # #     print("-" * 100)

# # #     any_above = any(v["verdict"] == "ABOVE noise" for v in summary.values())
# # #     print("\n---- verdict ----")
# # #     if any_above:
# # #         print("The token moves at least one metric BEYOND the pipeline's own run-to-run "
# # #               "spread. Treat as PRELIMINARY: confirm with more scenes and more repeats, "
# # #               "then localize with L2/L3.")
# # #     else:
# # #         print("NO metric exceeds the within-condition noise envelope. The token's effect is "
# # #               "indistinguishable from rerun variance -- any single-pair difference you saw "
# # #               "(e.g. an object appearing/disappearing) is noise, NOT the token.")
# # #     wb_mean = summary.get("obj_set_diff", {}).get("within_baseline", [float('nan')])[0]
# # #     if wb_mean == wb_mean and wb_mean > 0.01:
# # #         print(f"\nNOTE: baselines differ from EACH OTHER (obj_set_diff={wb_mean:.3f}). The "
# # #               f"pipeline is stochastic run-to-run, so single-pair comparisons are invalid by "
# # #               f"construction. If you ran --det-check, this is pure kernel nondeterminism.")
# # #     print("\nCAVEATS: within-pairs share runs (not fully independent), so this is a "
# # #           "descriptive envelope, not a calibrated p-value. Scale scenes AND repeats before "
# # #           "claiming an effect.\n")

# # #     out = os.path.join(args.dir, "l1_variance.json")
# # #     json.dump(summary, open(out, "w"), indent=2)
# # #     print(f"[var] -> {out}")


# # # if __name__ == "__main__":
# # #     main()

# # #!/usr/bin/env python3
# # # =============================================================================
# # # adv_l1_variance.py -- does the token move the output MORE than the pipeline's
# # #                       own run-to-run noise?
# # # -----------------------------------------------------------------------------
# # # WHY THIS EXISTS
# # #   A single baseline-vs-adversarial pair cannot support any claim: reruns of the
# # #   SAME baseline already differ (different object sets, different layouts). So we
# # #   compare two DISTRIBUTIONS of differences:
# # #
# # #     WITHIN-baseline    : diff(baseline_r, baseline_s)     r<s   -> pure noise
# # #     WITHIN-adversarial : diff(adv_r, adv_s)               r<s   -> pure noise
# # #     BETWEEN            : diff(baseline_r, adv_s)          all   -> noise + token
# # #
# # #   If the token does nothing, BETWEEN is drawn from the same distribution as the
# # #   WITHINs. A token effect = BETWEEN shifted ABOVE the within envelope.
# # #   This is the distance-based (PERMANOVA-style) logic, and it is the only honest
# # #   test when the pipeline is itself stochastic.
# # #
# # #   All metrics are DIFFERENCE measures on the FINAL COMMITTED scene (i.e. after
# # #   the best-of-N refinement loop) -- higher = more different:
# # #     obj_set_diff    1 - Jaccard of the final object-label sets
# # #     n_obj_diff      |#objects_a - #objects_b|
# # #     occ_hamming     fraction of occupancy-grid cells that differ
# # #     centroid_shift  mean euclidean move of objects present in BOTH (metres)
# # #     legality_flip   final legality differs (0/1)
# # #
# # # USAGE
# # #   python adv_l1_variance.py --dir adv_out
# # #   python adv_l1_variance.py --dir adv_out --metric obj_set_diff
# # #
# # # INTERPRETATION
# # #   * within ~ 0  and between > 0  -> deterministic pipeline, real token effect.
# # #   * within ~ between            -> token effect indistinguishable from noise.
# # #   * within large                -> pipeline is stochastic; you need many repeats
# # #                                    AND many scenes before any claim is possible.
# # # =============================================================================
# # from __future__ import annotations

# # import argparse
# # import glob
# # import itertools
# # import json
# # import math
# # import os
# # import re

# # import numpy as np

# # import adv_common as C

# # METRICS = ["obj_set_diff", "n_obj_diff", "occ_hamming", "centroid_shift", "legality_flip"]


# # # ---- final (post-refinement) committed state per scene ----------------------
# # def final_scene_state(recs):
# #     """scene_id -> {objects:{label:(x,y)}, grid:[...], legality:int}
# #     Uses ONLY records flagged committed=True, i.e. the best-of-N winner that the
# #     pipeline actually commits and renders as 'initial'."""
# #     out = {}
# #     for r in recs:
# #         lab = r.get("labels", {})
# #         if not lab.get("committed"):
# #             continue
# #         sid = str(r.get("ctx", {}).get("scene_id"))
# #         e = out.setdefault(sid, {"objects": {}, "grid": None, "legality": None})
# #         e["objects"].update(C.parse_objs(r.get("output_text", "")))
# #         e["grid"] = lab.get("occupancy_str")
# #         e["legality"] = lab.get("legality")
# #     return out


# # def _canon(label: str) -> str:
# #     """Drop instance suffixes so 'wooden_table_1' vs 'wooden_table_2' both count
# #     as the requested asset type when comparing SETS of object types."""
# #     s = str(label).lower().strip()
# #     s = re.sub(r"_\d+$", "", s)
# #     return s


# # def pair_metrics(recs_a, recs_b, by_type=False):
# #     """Mean over shared scenes of each difference metric. Higher = more different."""
# #     sa, sb = final_scene_state(recs_a), final_scene_state(recs_b)
# #     scenes = sorted(set(sa) & set(sb))
# #     if not scenes:
# #         return None
# #     acc = {m: [] for m in METRICS}
# #     for s in scenes:
# #         A, B = sa[s], sb[s]
# #         ka = {(_canon(k) if by_type else k) for k in A["objects"]}
# #         kb = {(_canon(k) if by_type else k) for k in B["objects"]}
# #         union = ka | kb
# #         jac = (len(ka & kb) / len(union)) if union else 1.0
# #         acc["obj_set_diff"].append(1.0 - jac)
# #         acc["n_obj_diff"].append(abs(len(A["objects"]) - len(B["objects"])))

# #         ga, gb = A["grid"], B["grid"]
# #         if ga and gb and len(ga) == len(gb):
# #             acc["occ_hamming"].append(sum(x != y for x, y in zip(ga, gb)) / len(ga))

# #         shared = set(A["objects"]) & set(B["objects"])
# #         if shared:
# #             acc["centroid_shift"].append(float(np.mean([
# #                 math.hypot(A["objects"][l][0] - B["objects"][l][0],
# #                            A["objects"][l][1] - B["objects"][l][1]) for l in shared])))

# #         if A["legality"] is not None and B["legality"] is not None:
# #             acc["legality_flip"].append(float(A["legality"] != B["legality"]))

# #     return {m: (float(np.mean(v)) if v else float("nan")) for m, v in acc.items()}


# # def prompt_determinism(base_runs):
# #     """Separate the TWO possible causes of irreproducibility.

# #     Prompts are stored per call, so compare the prompt SEQUENCES of two baseline
# #     runs (identical condition):
# #       * prompts DIFFER  -> RNG desync. generate_random_seed() feeds 'Random seed: N'
# #                            into the prompt; seeding random/numpy/torch fixes it.
# #       * prompts IDENTICAL but outputs differ -> the MODEL forward is nondeterministic
# #                            (MoE routing / kernel atomics). Seeding cannot fix that;
# #                            you must use the variance-envelope method permanently.
# #     """
# #     import hashlib
# #     def hashes(recs):
# #         return [hashlib.md5((r.get("prompt") or "").encode()).hexdigest()[:8] for r in recs]
# #     def outs(recs):
# #         return [hashlib.md5((r.get("output_text") or "").encode()).hexdigest()[:8] for r in recs]

# #     rows = []
# #     for i, j in itertools.combinations(range(len(base_runs)), 2):
# #         ha, hb = hashes(base_runs[i]), hashes(base_runs[j])
# #         oa, ob = outs(base_runs[i]), outs(base_runs[j])
# #         n = min(len(ha), len(hb))
# #         first_prompt_div = next((k for k in range(n) if ha[k] != hb[k]), None)
# #         first_out_div = next((k for k in range(n) if oa[k] != ob[k]), None)
# #         rows.append({"pair": (i, j), "n_calls": (len(ha), len(hb)),
# #                      "first_prompt_divergence": first_prompt_div,
# #                      "first_output_divergence": first_out_div})
# #     return rows


# # def report_prompt_determinism(rows):
# #     print("\n---- prompt determinism (baseline vs baseline: identical condition) ----")
# #     if not rows:
# #         print("  n/a")
# #         return
# #     print(f"{'pair':>8} | {'calls':>10} | {'1st prompt diff':>16} | {'1st output diff':>16}")
# #     for r in rows:
# #         pd = r["first_prompt_divergence"]
# #         od = r["first_output_divergence"]
# #         print(f"{str(r['pair']):>8} | {str(r['n_calls']):>10} | "
# #               f"{('none' if pd is None else pd):>16} | {('none' if od is None else od):>16}")
# #     any_prompt_div = any(r["first_prompt_divergence"] is not None for r in rows)
# #     any_out_div = any(r["first_output_divergence"] is not None for r in rows)
# #     print("\n  -> ", end="")
# #     if any_prompt_div:
# #         print("PROMPTS DIFFER between identical runs => RNG desync (generate_random_seed "
# #               "feeds the prompt). adv_common.seed_everything() should fix this; if you "
# #               "already re-ran with the fix and prompts STILL differ, the seeding is not "
# #               "reaching utils_copy_img's `random` module.")
# #     elif any_out_div:
# #         print("PROMPTS IDENTICAL but OUTPUTS DIFFER => the MODEL forward is "
# #               "NONDETERMINISTIC (MoE routing / kernel atomics). Seeding cannot fix this. "
# #               "Use the variance-envelope method with many repeats, permanently.")
# #     else:
# #         print("FULLY DETERMINISTIC: identical prompts AND identical outputs. Single-pair "
# #               "comparisons are now valid; the token is the only source of difference.")


# # def _mean_std(xs):
# #     xs = [x for x in xs if x == x]
# #     if not xs:
# #         return float("nan"), float("nan"), 0
# #     m = float(np.mean(xs))
# #     return m, float(np.std(xs)), len(xs)


# # def _stat(D, lab):
# #     """mean(between-group distance) - mean(within-group distance)."""
# #     n = len(lab)
# #     bt, wi = [], []
# #     for i in range(n):
# #         for j in range(i + 1, n):
# #             v = D[i][j]
# #             if v != v:
# #                 continue
# #             (bt if lab[i] != lab[j] else wi).append(v)
# #     if not bt or not wi:
# #         return float("nan")
# #     return float(np.mean(bt) - np.mean(wi))


# # def permutation_test(D, lab):
# #     """EXACT permutation test over all relabelings.

# #     Runs are exchangeable if the note does nothing, so we recompute the statistic
# #     for every possible assignment of runs into two groups of the observed sizes.
# #     p = fraction of labelings with a statistic >= the observed one. This respects
# #     the fact that within-pairs share runs (no independence assumption), unlike a
# #     naive mean +/- std rule.

# #     Also returns min_p = 2 / C(n,k): the SMALLEST p this design can produce. With
# #     3 repeats per condition min_p = 0.100 -- i.e. 3 repeats CANNOT reach p<0.05 no
# #     matter how large the effect."""
# #     n = len(lab)
# #     k = int(sum(lab))
# #     T_obs = _stat(D, lab)
# #     if T_obs != T_obs:
# #         return float("nan"), float("nan"), 0, float("nan")
# #     stats = []
# #     for idx in itertools.combinations(range(n), k):
# #         l = [0] * n
# #         for i in idx:
# #             l[i] = 1
# #         s = _stat(D, l)
# #         if s == s:
# #             stats.append(s)
# #     if not stats:
# #         return T_obs, float("nan"), 0, float("nan")
# #     p = float(np.mean([s >= T_obs - 1e-12 for s in stats]))
# #     min_p = 2.0 / len(stats)
# #     return T_obs, p, len(stats), min_p


# # def load_runs(d):
# #     def grab(pat):
# #         out = []
# #         for p in sorted(glob.glob(os.path.join(d, pat)),
# #                         key=lambda x: int(re.search(r"_r(\d+)\.pkl$", x).group(1))):
# #             out.append(C.load_snapshot(p))
# #         return out
# #     return grab("baseline_r*.pkl"), grab("adversarial_r*.pkl")


# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--dir", required=True)
# #     ap.add_argument("--by-type", action="store_true",
# #                     help="compare object TYPES (strip _1/_2 suffixes) instead of instances")
# #     ap.add_argument("--metric", default=None, choices=METRICS)
# #     ap.add_argument("--alpha", type=float, default=0.05)
# #     args = ap.parse_args()

# #     base, adv = load_runs(args.dir)
# #     print(f"[var] loaded {len(base)} baseline run(s), {len(adv)} adversarial run(s)")
# #     if len(base) < 2 or len(adv) < 2:
# #         raise SystemExit("[abort] need >=2 repeats per condition. Use --repeats 5 or more.")

# #     runs = base + adv
# #     lab = [0] * len(base) + [1] * len(adv)
# #     n = len(runs)

# #     report_prompt_determinism(prompt_determinism(base))

# #     # pairwise metric matrices over ALL runs
# #     D = {m: [[float("nan")] * n for _ in range(n)] for m in METRICS}
# #     for i in range(n):
# #         for j in range(i + 1, n):
# #             pm = pair_metrics(runs[i], runs[j], args.by_type)
# #             if not pm:
# #                 continue
# #             for m in METRICS:
# #                 D[m][i][j] = D[m][j][i] = pm[m]

# #     mets = [args.metric] if args.metric else METRICS
# #     print(f"\n{'metric':>15} | {'WITHIN-base':>16} | {'WITHIN-adv':>16} | {'BETWEEN':>16} | "
# #           f"{'T':>7} | {'p_perm':>7} | verdict")
# #     print("-" * 108)
# #     summary, min_p_global = {}, None
# #     for m in mets:
# #         wb = _mean_std([D[m][i][j] for i, j in itertools.combinations(range(len(base)), 2)])
# #         wa = _mean_std([D[m][i][j] for i, j in
# #                         itertools.combinations(range(len(base), n), 2)])
# #         bt = _mean_std([D[m][i][j] for i in range(len(base)) for j in range(len(base), n)])
# #         T, p, nperm, min_p = permutation_test(D[m], lab)
# #         min_p_global = min_p if min_p_global is None else min_p_global
# #         if p != p:
# #             verdict = "n/a"
# #         elif p <= args.alpha:
# #             verdict = "EFFECT (p<=%.2f)" % args.alpha
# #         else:
# #             verdict = "within noise"
# #         summary[m] = {"within_baseline": wb, "within_adversarial": wa, "between": bt,
# #                       "T": T, "p_perm": p, "n_labelings": nperm, "min_achievable_p": min_p,
# #                       "verdict": verdict}
# #         print(f"{m:>15} | {wb[0]:>7.3f} ±{wb[1]:5.3f}({wb[2]}) | {wa[0]:>7.3f} ±{wa[1]:5.3f}({wa[2]}) | "
# #               f"{bt[0]:>7.3f} ±{bt[1]:5.3f}({bt[2]}) | {T:>+7.3f} | {p:>7.3f} | {verdict}")
# #     print("-" * 108)

# #     print("\n---- verdict ----")
# #     if min_p_global == min_p_global and min_p_global > args.alpha:
# #         print(f"UNDERPOWERED BY DESIGN: with {len(base)}+{len(adv)} runs the smallest p this "
# #               f"test can produce is {min_p_global:.3f} > alpha={args.alpha}. No result here "
# #               f"can be significant regardless of effect size. Increase --repeats "
# #               f"(5+5 -> min p=0.008).")
# #     elif any(v["verdict"].startswith("EFFECT") for v in summary.values()):
# #         print("The note shifts at least one metric beyond the exchangeability null. "
# #               "PRELIMINARY: confirm across many scenes, then localize with L2/L3.")
# #     else:
# #         print("NO metric separates from the permutation null. The note's effect is "
# #               "indistinguishable from rerun variance.")

# #     wbm = summary.get("obj_set_diff", {}).get("within_baseline", [float("nan")])[0]
# #     if wbm == wbm and wbm > 0.01:
# #         print(f"\nDETERMINISM: baselines differ from EACH OTHER (obj_set_diff={wbm:.3f}). "
# #               f"If you ran --det-check (same seed), the pipeline is NOT reproducible: "
# #               f"utils_copy_img.generate_random_seed() feeds 'Random seed: N' into the PROMPT, "
# #               f"so unseeded RNG => different prompts every run. adv_common.seed_everything() "
# #               f"now seeds random/numpy/torch per run -- re-run --det-check; obj_set_diff "
# #               f"should drop to ~0. If it does NOT, the residue is kernel/MoE nondeterminism.")
# #     print("\nCAVEAT: permutation over runs assumes runs are exchangeable under the null; "
# #           "scenes are pooled inside each metric. Scale scenes AND repeats before claiming.\n")

# #     out = os.path.join(args.dir, "l1_variance.json")
# #     json.dump(summary, open(out, "w"), indent=2, default=str)
# #     print(f"[var] -> {out}")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# # =============================================================================
# # adv_l1_variance.py -- does the token move the output MORE than the pipeline's
# #                       own run-to-run noise?
# # -----------------------------------------------------------------------------
# # WHY THIS EXISTS
# #   A single baseline-vs-adversarial pair cannot support any claim: reruns of the
# #   SAME baseline already differ (different object sets, different layouts). So we
# #   compare two DISTRIBUTIONS of differences:
# #
# #     WITHIN-baseline    : diff(baseline_r, baseline_s)     r<s   -> pure noise
# #     WITHIN-adversarial : diff(adv_r, adv_s)               r<s   -> pure noise
# #     BETWEEN            : diff(baseline_r, adv_s)          all   -> noise + token
# #
# #   If the token does nothing, BETWEEN is drawn from the same distribution as the
# #   WITHINs. A token effect = BETWEEN shifted ABOVE the within envelope.
# #   This is the distance-based (PERMANOVA-style) logic, and it is the only honest
# #   test when the pipeline is itself stochastic.
# #
# #   All metrics are DIFFERENCE measures on the FINAL COMMITTED scene (i.e. after
# #   the best-of-N refinement loop) -- higher = more different:
# #     obj_set_diff    1 - Jaccard of the final object-label sets
# #     n_obj_diff      |#objects_a - #objects_b|
# #     occ_hamming     fraction of occupancy-grid cells that differ
# #     centroid_shift  mean euclidean move of objects present in BOTH (metres)
# #     legality_flip   final legality differs (0/1)
# #
# # USAGE
# #   python adv_l1_variance.py --dir adv_out
# #   python adv_l1_variance.py --dir adv_out --metric obj_set_diff
# #
# # INTERPRETATION
# #   * within ~ 0  and between > 0  -> deterministic pipeline, real token effect.
# #   * within ~ between            -> token effect indistinguishable from noise.
# #   * within large                -> pipeline is stochastic; you need many repeats
# #                                    AND many scenes before any claim is possible.
# # =============================================================================
# from __future__ import annotations

# import argparse
# import glob
# import itertools
# import json
# import math
# import os
# import re

# import numpy as np

# import adv_common as C

# METRICS = ["obj_set_diff", "n_obj_diff", "occ_hamming", "centroid_shift", "legality_flip"]


# # ---- final (post-refinement) committed state per scene ----------------------
# def final_scene_state(recs):
#     """scene_id -> {objects:{label:(x,y)}, grid:[...], legality:int}
#     Uses ONLY records flagged committed=True, i.e. the best-of-N winner that the
#     pipeline actually commits and renders as 'initial'."""
#     out = {}
#     for r in recs:
#         lab = r.get("labels", {})
#         if not lab.get("committed"):
#             continue
#         sid = str(r.get("ctx", {}).get("scene_id"))
#         e = out.setdefault(sid, {"objects": {}, "grid": None, "legality": None})
#         e["objects"].update(C.parse_objs(r.get("output_text", "")))
#         e["grid"] = lab.get("occupancy_str")
#         e["legality"] = lab.get("legality")
#     return out


# def _canon(label: str) -> str:
#     """Drop instance suffixes so 'wooden_table_1' vs 'wooden_table_2' both count
#     as the requested asset type when comparing SETS of object types."""
#     s = str(label).lower().strip()
#     s = re.sub(r"_\d+$", "", s)
#     return s


# def pair_metrics(recs_a, recs_b, by_type=False):
#     """Mean over shared scenes of each difference metric. Higher = more different."""
#     sa, sb = final_scene_state(recs_a), final_scene_state(recs_b)
#     scenes = sorted(set(sa) & set(sb))
#     if not scenes:
#         return None
#     acc = {m: [] for m in METRICS}
#     for s in scenes:
#         A, B = sa[s], sb[s]
#         ka = {(_canon(k) if by_type else k) for k in A["objects"]}
#         kb = {(_canon(k) if by_type else k) for k in B["objects"]}
#         union = ka | kb
#         jac = (len(ka & kb) / len(union)) if union else 1.0
#         acc["obj_set_diff"].append(1.0 - jac)
#         acc["n_obj_diff"].append(abs(len(A["objects"]) - len(B["objects"])))

#         ga, gb = A["grid"], B["grid"]
#         if ga and gb and len(ga) == len(gb):
#             acc["occ_hamming"].append(sum(x != y for x, y in zip(ga, gb)) / len(ga))

#         shared = set(A["objects"]) & set(B["objects"])
#         if shared:
#             acc["centroid_shift"].append(float(np.mean([
#                 math.hypot(A["objects"][l][0] - B["objects"][l][0],
#                            A["objects"][l][1] - B["objects"][l][1]) for l in shared])))

#         if A["legality"] is not None and B["legality"] is not None:
#             acc["legality_flip"].append(float(A["legality"] != B["legality"]))

#     return {m: (float(np.mean(v)) if v else float("nan")) for m, v in acc.items()}


# def _seed_seq(recs):
#     """The 'Random seed: N' values actually injected into each prompt."""
#     out = []
#     for r in recs:
#         m = re.search(r"Random seed:\s*(\d+)", r.get("prompt") or "")
#         out.append(int(m.group(1)) if m else None)
#     return out


# def prompt_determinism(base_runs, adv_runs):
#     """Verify the seeding, correctly for BOTH run modes.

#     Two different checks, because the two modes mean different things:

#     A) ACROSS repeats (baseline_r vs baseline_s): only meaningful with --det-check
#        (same seed every repeat). WITHOUT --det-check each repeat uses a DIFFERENT
#        seed by design, and the seed text is in the first prompt, so prompts differ
#        at call 0 -- that is diversity, NOT desync.

#     B) WITHIN a matched pair (baseline_r vs adversarial_r): these SHARE a seed, so
#        their injected seed SEQUENCES must be identical. If they are not, the pair is
#        CONFOUNDED (the runs differ by seed as well as by the note) and no causal
#        reading is possible. Known cause: the first run in a process consumes the
#        global RNG differently (e.g. plot_utils allocating label colours on the first
#        visualize() call, then caching them), shifting that run's seed stream only.
#        adv_common.force_deterministic_seed_text() removes this by making the seed
#        counter-based. THIS check is valid in both modes.
#     """
#     import hashlib
#     rows_across = []
#     for i, j in itertools.combinations(range(len(base_runs)), 2):
#         sa, sb = _seed_seq(base_runs[i]), _seed_seq(base_runs[j])
#         n = min(len(sa), len(sb))
#         rows_across.append({
#             "pair": (i, j),
#             "same_seed_run": sa[:n] == sb[:n],
#             "first_seed_div": next((k for k in range(n) if sa[k] != sb[k]), None),
#         })

#     rows_pairs = []
#     for r in range(min(len(base_runs), len(adv_runs))):
#         sa, sb = _seed_seq(base_runs[r]), _seed_seq(adv_runs[r])
#         n = min(len(sa), len(sb))
#         first_div = next((k for k in range(n) if sa[k] != sb[k]), None)
#         rows_pairs.append({"repeat": r, "n_calls": (len(sa), len(sb)),
#                            "first_seed_div": first_div,
#                            "seeds_match": first_div is None and len(sa) == len(sb)})
#     return rows_across, rows_pairs


# def report_prompt_determinism(rows_across, rows_pairs):
#     print("\n---- seeding check ----")
#     across_same = [r["same_seed_run"] for r in rows_across]
#     if across_same and all(across_same):
#         print("  across repeats: baselines share the SAME seed sequence "
#               "(consistent with --det-check).")
#     else:
#         print("  across repeats: baselines use DIFFERENT seeds -- EXPECTED without "
#               "--det-check (each repeat samples a different scene). This is the "
#               "pipeline's designed diversity, NOT a bug.")

#     print("\n  matched pairs (baseline_r vs adversarial_r -- MUST share a seed):")
#     print(f"  {'repeat':>7} | {'calls':>10} | {'seeds match':>12} | {'1st seed div':>13}")
#     bad = []
#     for r in rows_pairs:
#         fd = r["first_seed_div"]
#         print(f"  {r['repeat']:>7} | {str(r['n_calls']):>10} | "
#               f"{str(r['seeds_match']):>12} | {('-' if fd is None else fd):>13}")
#         if not r["seeds_match"]:
#             bad.append(r["repeat"])
#     print("\n  -> ", end="")
#     if not rows_pairs:
#         print("n/a")
#     elif bad:
#         print(f"CONFOUNDED PAIRS at repeat(s) {bad}: baseline and adversarial got "
#               f"DIFFERENT seed sequences, so they differ by seed AS WELL AS the note -- "
#               f"their difference is NOT attributable to the note. Cause: the first run in "
#               f"a process consumes the global RNG differently (plot_utils colour cache). "
#               f"Fix: use adv_common with force_deterministic_seed_text() and re-generate.")
#     else:
#         print("ALL PAIRS CLEAN: each baseline/adversarial pair shares an identical seed "
#               "sequence, so within a pair the ONLY difference is the note. Matched-pair "
#               "differences below are CAUSAL.")


# def report_matched_pairs(base, adv, by_type=False):
#     """THE causal test once the pipeline is deterministic.

#     baseline_r and adversarial_r share a seed, so they differ ONLY by the note.
#     With determinism verified (prompt-determinism section all 'none'), ANY nonzero
#     difference here is CAUSED by the note -- no statistics needed. Across repeats
#     (different seeds) you get independent measurements of how consistently it acts."""
#     n = min(len(base), len(adv))
#     print("\n---- matched-seed pairs: baseline_r vs adversarial_r (same seed) ----")
#     print("     (deterministic pipeline => any nonzero value IS the note's effect)")
#     print(f"{'repeat':>7} | " + " | ".join(f"{m:>14}" for m in METRICS))
#     print("-" * (10 + 17 * len(METRICS)))
#     rows = []
#     for r in range(n):
#         pm = pair_metrics(base[r], adv[r], by_type)
#         if not pm:
#             continue
#         rows.append(pm)
#         print(f"{r:>7} | " + " | ".join(f"{pm[m]:>14.3f}" for m in METRICS))
#     print("-" * (10 + 17 * len(METRICS)))
#     if not rows:
#         return {}
#     out = {}
#     for m in METRICS:
#         vals = [x[m] for x in rows if x[m] == x[m]]
#         nz = sum(1 for v in vals if v > 1e-9)
#         mu = float(np.mean(vals)) if vals else float("nan")
#         out[m] = {"mean": mu, "n_nonzero": nz, "n": len(vals)}
#         print(f"{m:>14}: mean={mu:.3f}   changed in {nz}/{len(vals)} seeds")
#     changed_any = any(v["n_nonzero"] > 0 for v in out.values())
#     print("\n  -> ", end="")
#     if changed_any:
#         print("The note CHANGES the generated scene. Because paired runs share a seed "
#               "and the pipeline is deterministic, this difference is CAUSAL, not noise. "
#               "Report 'changed in k/n seeds' as the effect's consistency, then run L2 "
#               "(activation shift) and L3 (which layers carry it).")
#     else:
#         print("The note changes NOTHING: identical scenes at every seed. The model fully "
#               "IGNORES it behaviourally. Run L2 to see whether it is even ENCODED "
#               "(activations move) or ignored at ingestion -- that is now the question.")
#     return out


# def _mean_std(xs):
#     xs = [x for x in xs if x == x]
#     if not xs:
#         return float("nan"), float("nan"), 0
#     m = float(np.mean(xs))
#     return m, float(np.std(xs)), len(xs)


# def _stat(D, lab):
#     """mean(between-group distance) - mean(within-group distance)."""
#     n = len(lab)
#     bt, wi = [], []
#     for i in range(n):
#         for j in range(i + 1, n):
#             v = D[i][j]
#             if v != v:
#                 continue
#             (bt if lab[i] != lab[j] else wi).append(v)
#     if not bt or not wi:
#         return float("nan")
#     return float(np.mean(bt) - np.mean(wi))


# def permutation_test(D, lab):
#     """EXACT permutation test over all relabelings.

#     Runs are exchangeable if the note does nothing, so we recompute the statistic
#     for every possible assignment of runs into two groups of the observed sizes.
#     p = fraction of labelings with a statistic >= the observed one. This respects
#     the fact that within-pairs share runs (no independence assumption), unlike a
#     naive mean +/- std rule.

#     Also returns min_p = 2 / C(n,k): the SMALLEST p this design can produce. With
#     3 repeats per condition min_p = 0.100 -- i.e. 3 repeats CANNOT reach p<0.05 no
#     matter how large the effect."""
#     n = len(lab)
#     k = int(sum(lab))
#     T_obs = _stat(D, lab)
#     if T_obs != T_obs:
#         return float("nan"), float("nan"), 0, float("nan")
#     stats = []
#     for idx in itertools.combinations(range(n), k):
#         l = [0] * n
#         for i in idx:
#             l[i] = 1
#         s = _stat(D, l)
#         if s == s:
#             stats.append(s)
#     if not stats:
#         return T_obs, float("nan"), 0, float("nan")
#     p = float(np.mean([s >= T_obs - 1e-12 for s in stats]))
#     min_p = 2.0 / len(stats)
#     return T_obs, p, len(stats), min_p


# def load_runs(d):
#     def grab(pat):
#         out = []
#         for p in sorted(glob.glob(os.path.join(d, pat)),
#                         key=lambda x: int(re.search(r"_r(\d+)\.pkl$", x).group(1))):
#             out.append(C.load_snapshot(p))
#         return out
#     return grab("baseline_r*.pkl"), grab("adversarial_r*.pkl")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dir", required=True)
#     ap.add_argument("--by-type", action="store_true",
#                     help="compare object TYPES (strip _1/_2 suffixes) instead of instances")
#     ap.add_argument("--metric", default=None, choices=METRICS)
#     ap.add_argument("--alpha", type=float, default=0.05)
#     args = ap.parse_args()

#     base, adv = load_runs(args.dir)
#     print(f"[var] loaded {len(base)} baseline run(s), {len(adv)} adversarial run(s)")
#     if len(base) < 2 or len(adv) < 2:
#         raise SystemExit("[abort] need >=2 repeats per condition. Use --repeats 5 or more.")

#     runs = base + adv
#     lab = [0] * len(base) + [1] * len(adv)
#     n = len(runs)

#     ra, rp = prompt_determinism(base, adv)
#     report_prompt_determinism(ra, rp)
#     matched = report_matched_pairs(base, adv, args.by_type)

#     # pairwise metric matrices over ALL runs
#     D = {m: [[float("nan")] * n for _ in range(n)] for m in METRICS}
#     for i in range(n):
#         for j in range(i + 1, n):
#             pm = pair_metrics(runs[i], runs[j], args.by_type)
#             if not pm:
#                 continue
#             for m in METRICS:
#                 D[m][i][j] = D[m][j][i] = pm[m]

#     mets = [args.metric] if args.metric else METRICS
#     print(f"\n{'metric':>15} | {'WITHIN-base':>16} | {'WITHIN-adv':>16} | {'BETWEEN':>16} | "
#           f"{'T':>7} | {'p_perm':>7} | verdict")
#     print("-" * 108)
#     summary, min_p_global = {}, None
#     for m in mets:
#         wb = _mean_std([D[m][i][j] for i, j in itertools.combinations(range(len(base)), 2)])
#         wa = _mean_std([D[m][i][j] for i, j in
#                         itertools.combinations(range(len(base), n), 2)])
#         bt = _mean_std([D[m][i][j] for i in range(len(base)) for j in range(len(base), n)])
#         T, p, nperm, min_p = permutation_test(D[m], lab)
#         min_p_global = min_p if min_p_global is None else min_p_global
#         if p != p:
#             verdict = "n/a"
#         elif p <= args.alpha:
#             verdict = "EFFECT (p<=%.2f)" % args.alpha
#         else:
#             verdict = "within noise"
#         summary[m] = {"within_baseline": wb, "within_adversarial": wa, "between": bt,
#                       "T": T, "p_perm": p, "n_labelings": nperm, "min_achievable_p": min_p,
#                       "verdict": verdict}
#         print(f"{m:>15} | {wb[0]:>7.3f} ±{wb[1]:5.3f}({wb[2]}) | {wa[0]:>7.3f} ±{wa[1]:5.3f}({wa[2]}) | "
#               f"{bt[0]:>7.3f} ±{bt[1]:5.3f}({bt[2]}) | {T:>+7.3f} | {p:>7.3f} | {verdict}")
#     print("-" * 108)

#     print("\n---- verdict ----")
#     if min_p_global == min_p_global and min_p_global > args.alpha:
#         print(f"UNDERPOWERED BY DESIGN: with {len(base)}+{len(adv)} runs the smallest p this "
#               f"test can produce is {min_p_global:.3f} > alpha={args.alpha}. No result here "
#               f"can be significant regardless of effect size. Increase --repeats "
#               f"(5+5 -> min p=0.008).")
#     elif any(v["verdict"].startswith("EFFECT") for v in summary.values()):
#         print("The note shifts at least one metric beyond the exchangeability null. "
#               "PRELIMINARY: confirm across many scenes, then localize with L2/L3.")
#     else:
#         print("NO metric separates from the permutation null. The note's effect is "
#               "indistinguishable from rerun variance.")

#     wbm = summary.get("obj_set_diff", {}).get("within_baseline", [float("nan")])[0]
#     if wbm == wbm and wbm > 0.01:
#         print(f"\nDETERMINISM: baselines differ from EACH OTHER (obj_set_diff={wbm:.3f}). "
#               f"If you ran --det-check (same seed), the pipeline is NOT reproducible: "
#               f"utils_copy_img.generate_random_seed() feeds 'Random seed: N' into the PROMPT, "
#               f"so unseeded RNG => different prompts every run. adv_common.seed_everything() "
#               f"now seeds random/numpy/torch per run -- re-run --det-check; obj_set_diff "
#               f"should drop to ~0. If it does NOT, the residue is kernel/MoE nondeterminism.")
#     print("\nCAVEAT: permutation over runs assumes runs are exchangeable under the null; "
#           "scenes are pooled inside each metric. Scale scenes AND repeats before claiming.\n")

#     out = os.path.join(args.dir, "l1_variance.json")
#     json.dump(summary, open(out, "w"), indent=2, default=str)
#     print(f"[var] -> {out}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# =============================================================================
# adv_l1_variance.py -- does the token move the output MORE than the pipeline's
#                       own run-to-run noise?
# -----------------------------------------------------------------------------
# WHY THIS EXISTS
#   A single baseline-vs-adversarial pair cannot support any claim: reruns of the
#   SAME baseline already differ (different object sets, different layouts). So we
#   compare two DISTRIBUTIONS of differences:
#
#     WITHIN-baseline    : diff(baseline_r, baseline_s)     r<s   -> pure noise
#     WITHIN-adversarial : diff(adv_r, adv_s)               r<s   -> pure noise
#     BETWEEN            : diff(baseline_r, adv_s)          all   -> noise + token
#
#   If the token does nothing, BETWEEN is drawn from the same distribution as the
#   WITHINs. A token effect = BETWEEN shifted ABOVE the within envelope.
#   This is the distance-based (PERMANOVA-style) logic, and it is the only honest
#   test when the pipeline is itself stochastic.
#
#   All metrics are DIFFERENCE measures on the FINAL COMMITTED scene (i.e. after
#   the best-of-N refinement loop) -- higher = more different:
#     obj_set_diff    1 - Jaccard of the final object-label sets
#     n_obj_diff      |#objects_a - #objects_b|
#     occ_hamming     fraction of occupancy-grid cells that differ
#     centroid_shift  mean euclidean move of objects present in BOTH (metres)
#     legality_flip   final legality differs (0/1)
#
# USAGE
#   python adv_l1_variance.py --dir adv_out
#   python adv_l1_variance.py --dir adv_out --metric obj_set_diff
#
# INTERPRETATION
#   * within ~ 0  and between > 0  -> deterministic pipeline, real token effect.
#   * within ~ between            -> token effect indistinguishable from noise.
#   * within large                -> pipeline is stochastic; you need many repeats
#                                    AND many scenes before any claim is possible.
# =============================================================================
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re

import numpy as np

import adv_common as C

METRICS = ["obj_set_diff", "n_obj_diff", "occ_hamming", "centroid_shift", "legality_flip"]


# ---- final (post-refinement) committed state per scene ----------------------
def final_scene_state(recs):
    """scene_id -> {objects:{label:(x,y)}, grid:[...], legality:int}
    Uses ONLY records flagged committed=True, i.e. the best-of-N winner that the
    pipeline actually commits and renders as 'initial'."""
    out = {}
    for r in recs:
        lab = r.get("labels", {})
        if not lab.get("committed"):
            continue
        sid = str(r.get("ctx", {}).get("scene_id"))
        e = out.setdefault(sid, {"objects": {}, "grid": None, "legality": None})
        e["objects"].update(C.parse_objs(r.get("output_text", "")))
        e["grid"] = lab.get("occupancy_str")
        e["legality"] = lab.get("legality")
    return out


def _canon(label: str) -> str:
    """Drop instance suffixes so 'wooden_table_1' vs 'wooden_table_2' both count
    as the requested asset type when comparing SETS of object types."""
    s = str(label).lower().strip()
    s = re.sub(r"_\d+$", "", s)
    return s


def pair_metrics(recs_a, recs_b, by_type=False):
    """Mean over shared scenes of each difference metric. Higher = more different."""
    sa, sb = final_scene_state(recs_a), final_scene_state(recs_b)
    scenes = sorted(set(sa) & set(sb))
    if not scenes:
        return None
    acc = {m: [] for m in METRICS}
    for s in scenes:
        A, B = sa[s], sb[s]
        ka = {(_canon(k) if by_type else k) for k in A["objects"]}
        kb = {(_canon(k) if by_type else k) for k in B["objects"]}
        union = ka | kb
        jac = (len(ka & kb) / len(union)) if union else 1.0
        acc["obj_set_diff"].append(1.0 - jac)
        acc["n_obj_diff"].append(abs(len(A["objects"]) - len(B["objects"])))

        ga, gb = A["grid"], B["grid"]
        if ga and gb and len(ga) == len(gb):
            acc["occ_hamming"].append(sum(x != y for x, y in zip(ga, gb)) / len(ga))

        shared = set(A["objects"]) & set(B["objects"])
        if shared:
            acc["centroid_shift"].append(float(np.mean([
                math.hypot(A["objects"][l][0] - B["objects"][l][0],
                           A["objects"][l][1] - B["objects"][l][1]) for l in shared])))

        if A["legality"] is not None and B["legality"] is not None:
            acc["legality_flip"].append(float(A["legality"] != B["legality"]))

    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in acc.items()}


def _seed_seq(recs):
    """The 'Random seed: N' values actually injected into each prompt."""
    out = []
    for r in recs:
        m = re.search(r"Random seed:\s*(\d+)", r.get("prompt") or "")
        out.append(int(m.group(1)) if m else None)
    return out


def prompt_determinism(base_runs, adv_runs):
    """Verify the seeding, correctly for BOTH run modes.

    Two different checks, because the two modes mean different things:

    A) ACROSS repeats (baseline_r vs baseline_s): only meaningful with --det-check
       (same seed every repeat). WITHOUT --det-check each repeat uses a DIFFERENT
       seed by design, and the seed text is in the first prompt, so prompts differ
       at call 0 -- that is diversity, NOT desync.

    B) WITHIN a matched pair (baseline_r vs adversarial_r): these SHARE a seed, so
       their injected seed SEQUENCES must be identical. If they are not, the pair is
       CONFOUNDED (the runs differ by seed as well as by the note) and no causal
       reading is possible. Known cause: the first run in a process consumes the
       global RNG differently (e.g. plot_utils allocating label colours on the first
       visualize() call, then caching them), shifting that run's seed stream only.
       adv_common.force_deterministic_seed_text() removes this by making the seed
       counter-based. THIS check is valid in both modes.
    """
    import hashlib
    rows_across = []
    for i, j in itertools.combinations(range(len(base_runs)), 2):
        sa, sb = _seed_seq(base_runs[i]), _seed_seq(base_runs[j])
        n = min(len(sa), len(sb))
        rows_across.append({
            "pair": (i, j),
            "same_seed_run": sa[:n] == sb[:n],
            "first_seed_div": next((k for k in range(n) if sa[k] != sb[k]), None),
        })

    rows_pairs = []
    for r in range(min(len(base_runs), len(adv_runs))):
        sa, sb = _seed_seq(base_runs[r]), _seed_seq(adv_runs[r])
        n = min(len(sa), len(sb))
        first_div = next((k for k in range(n) if sa[k] != sb[k]), None)
        rows_pairs.append({"repeat": r, "n_calls": (len(sa), len(sb)),
                           "first_seed_div": first_div,
                           "seeds_match": first_div is None and len(sa) == len(sb)})
    return rows_across, rows_pairs


def report_prompt_determinism(rows_across, rows_pairs):
    print("\n---- seeding check ----")
    across_same = [r["same_seed_run"] for r in rows_across]
    if across_same and all(across_same):
        print("  across repeats: baselines share the SAME seed sequence "
              "(consistent with --det-check).")
    else:
        print("  across repeats: baselines use DIFFERENT seeds -- EXPECTED without "
              "--det-check (each repeat samples a different scene). This is the "
              "pipeline's designed diversity, NOT a bug.")

    print("\n  matched pairs (baseline_r vs adversarial_r -- MUST share a seed):")
    print(f"  {'repeat':>7} | {'calls':>10} | {'seeds match':>12} | {'1st seed div':>13}")
    bad = []
    for r in rows_pairs:
        fd = r["first_seed_div"]
        print(f"  {r['repeat']:>7} | {str(r['n_calls']):>10} | "
              f"{str(r['seeds_match']):>12} | {('-' if fd is None else fd):>13}")
        if not r["seeds_match"]:
            bad.append(r["repeat"])
    print("\n  -> ", end="")
    if not rows_pairs:
        print("n/a")
    elif bad:
        print(f"CONFOUNDED PAIRS at repeat(s) {bad}: baseline and adversarial got "
              f"DIFFERENT seed sequences, so they differ by seed AS WELL AS the note -- "
              f"their difference is NOT attributable to the note. Cause: the first run in "
              f"a process consumes the global RNG differently (plot_utils colour cache). "
              f"Fix: use adv_common with force_deterministic_seed_text() and re-generate.")
    else:
        print("ALL PAIRS CLEAN: each baseline/adversarial pair shares an identical seed "
              "sequence, so within a pair the ONLY difference is the note. Matched-pair "
              "differences below are CAUSAL.")


def report_matched_pairs(base, adv, by_type=False, valid_repeats=None):
    """THE causal test once the pipeline is deterministic.

    baseline_r and adversarial_r share a seed, so they differ ONLY by the note --
    BUT ONLY IF their seed sequences actually match. Pairs flagged CONFOUNDED by the
    seeding check differ by seed as well, and are EXCLUDED from the summary: including
    them inflates the apparent effect (they are comparing different scenes)."""
    n = min(len(base), len(adv))
    valid = set(range(n)) if valid_repeats is None else set(valid_repeats)
    print("\n---- matched-seed pairs: baseline_r vs adversarial_r (same seed) ----")
    print("     (deterministic pipeline => any nonzero value IS the note's effect)")
    print("     CONFOUNDED pairs are shown but EXCLUDED from the summary.")
    print(f"{'repeat':>7} | {'valid':>5} | " + " | ".join(f"{m:>14}" for m in METRICS))
    print("-" * (18 + 17 * len(METRICS)))
    rows = []
    for r in range(n):
        pm = pair_metrics(base[r], adv[r], by_type)
        if not pm:
            continue
        ok = r in valid
        if ok:
            rows.append(pm)
        print(f"{r:>7} | {('yes' if ok else 'NO'):>5} | "
              + " | ".join(f"{pm[m]:>14.3f}" for m in METRICS))
    print("-" * (18 + 17 * len(METRICS)))
    n_excl = n - len(rows)
    if n_excl:
        print(f"  EXCLUDED {n_excl} confounded pair(s); summary uses {len(rows)} valid pair(s).")
    if not rows:
        print("  -> NO VALID PAIRS. Every pair is confounded: regenerate with the updated "
              "adv_common.py (force_deterministic_seed_text) before drawing any conclusion.")
        return {}
    out = {}
    for m in METRICS:
        vals = [x[m] for x in rows if x[m] == x[m]]
        nz = sum(1 for v in vals if v > 1e-9)
        mu = float(np.mean(vals)) if vals else float("nan")
        out[m] = {"mean": mu, "n_nonzero": nz, "n": len(vals)}
        print(f"{m:>14}: mean={mu:.3f}   changed in {nz}/{len(vals)} VALID seeds")
    changed_any = any(v["n_nonzero"] > 0 for v in out.values())
    print("\n  -> ", end="")
    if len(rows) < 3:
        print(f"ONLY {len(rows)} VALID PAIR(S) -- too few to claim anything. Regenerate with "
              f"the updated adv_common.py so all repeats are clean, then re-read this table.")
    elif changed_any:
        print("The note CHANGES the generated scene. Paired runs share a seed and the "
              "pipeline is deterministic, so this is CAUSAL. Report 'changed in k/n seeds' "
              "as the effect's consistency, then run L2 and L3.")
    else:
        print("The note changes NOTHING at any valid seed: behaviourally IGNORED. Run L2 to "
              "see whether it is ENCODED at all (activations move) or ignored at ingestion.")
    return out


def _mean_std(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan"), 0
    m = float(np.mean(xs))
    return m, float(np.std(xs)), len(xs)


def _stat(D, lab):
    """mean(between-group distance) - mean(within-group distance)."""
    n = len(lab)
    bt, wi = [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = D[i][j]
            if v != v:
                continue
            (bt if lab[i] != lab[j] else wi).append(v)
    if not bt or not wi:
        return float("nan")
    return float(np.mean(bt) - np.mean(wi))


def permutation_test(D, lab):
    """EXACT permutation test over all relabelings.

    Runs are exchangeable if the note does nothing, so we recompute the statistic
    for every possible assignment of runs into two groups of the observed sizes.
    p = fraction of labelings with a statistic >= the observed one. This respects
    the fact that within-pairs share runs (no independence assumption), unlike a
    naive mean +/- std rule.

    Also returns min_p = 2 / C(n,k): the SMALLEST p this design can produce. With
    3 repeats per condition min_p = 0.100 -- i.e. 3 repeats CANNOT reach p<0.05 no
    matter how large the effect."""
    n = len(lab)
    k = int(sum(lab))
    T_obs = _stat(D, lab)
    if T_obs != T_obs:
        return float("nan"), float("nan"), 0, float("nan")
    stats = []
    for idx in itertools.combinations(range(n), k):
        l = [0] * n
        for i in idx:
            l[i] = 1
        s = _stat(D, l)
        if s == s:
            stats.append(s)
    if not stats:
        return T_obs, float("nan"), 0, float("nan")
    p = float(np.mean([s >= T_obs - 1e-12 for s in stats]))
    min_p = 2.0 / len(stats)
    return T_obs, p, len(stats), min_p


def load_runs(d):
    def grab(pat):
        out = []
        for p in sorted(glob.glob(os.path.join(d, pat)),
                        key=lambda x: int(re.search(r"_r(\d+)\.pkl$", x).group(1))):
            out.append(C.load_snapshot(p))
        return out
    return grab("baseline_r*.pkl"), grab("adversarial_r*.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--by-type", action="store_true",
                    help="compare object TYPES (strip _1/_2 suffixes) instead of instances")
    ap.add_argument("--metric", default=None, choices=METRICS)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    base, adv = load_runs(args.dir)
    print(f"[var] loaded {len(base)} baseline run(s), {len(adv)} adversarial run(s)")
    if len(base) < 2 or len(adv) < 2:
        raise SystemExit("[abort] need >=2 repeats per condition. Use --repeats 5 or more.")

    runs = base + adv
    lab = [0] * len(base) + [1] * len(adv)
    n = len(runs)

    ra, rp = prompt_determinism(base, adv)
    report_prompt_determinism(ra, rp)
    valid_repeats = [r["repeat"] for r in rp if r["seeds_match"]]
    matched = report_matched_pairs(base, adv, args.by_type, valid_repeats=valid_repeats)

    # pairwise metric matrices over ALL runs
    D = {m: [[float("nan")] * n for _ in range(n)] for m in METRICS}
    for i in range(n):
        for j in range(i + 1, n):
            pm = pair_metrics(runs[i], runs[j], args.by_type)
            if not pm:
                continue
            for m in METRICS:
                D[m][i][j] = D[m][j][i] = pm[m]

    mets = [args.metric] if args.metric else METRICS
    print(f"\n{'metric':>15} | {'WITHIN-base':>16} | {'WITHIN-adv':>16} | {'BETWEEN':>16} | "
          f"{'T':>7} | {'p_perm':>7} | verdict")
    print("-" * 108)
    summary, min_p_global = {}, None
    for m in mets:
        wb = _mean_std([D[m][i][j] for i, j in itertools.combinations(range(len(base)), 2)])
        wa = _mean_std([D[m][i][j] for i, j in
                        itertools.combinations(range(len(base), n), 2)])
        bt = _mean_std([D[m][i][j] for i in range(len(base)) for j in range(len(base), n)])
        T, p, nperm, min_p = permutation_test(D[m], lab)
        min_p_global = min_p if min_p_global is None else min_p_global
        if p != p:
            verdict = "n/a"
        elif p <= args.alpha:
            verdict = "EFFECT (p<=%.2f)" % args.alpha
        else:
            verdict = "within noise"
        summary[m] = {"within_baseline": wb, "within_adversarial": wa, "between": bt,
                      "T": T, "p_perm": p, "n_labelings": nperm, "min_achievable_p": min_p,
                      "verdict": verdict}
        print(f"{m:>15} | {wb[0]:>7.3f} ±{wb[1]:5.3f}({wb[2]}) | {wa[0]:>7.3f} ±{wa[1]:5.3f}({wa[2]}) | "
              f"{bt[0]:>7.3f} ±{bt[1]:5.3f}({bt[2]}) | {T:>+7.3f} | {p:>7.3f} | {verdict}")
    print("-" * 108)

    print("\n---- verdict ----")
    if min_p_global == min_p_global and min_p_global > args.alpha:
        print(f"UNDERPOWERED BY DESIGN: with {len(base)}+{len(adv)} runs the smallest p this "
              f"test can produce is {min_p_global:.3f} > alpha={args.alpha}. No result here "
              f"can be significant regardless of effect size. Increase --repeats "
              f"(5+5 -> min p=0.008).")
    elif any(v["verdict"].startswith("EFFECT") for v in summary.values()):
        print("The note shifts at least one metric beyond the exchangeability null. "
              "PRELIMINARY: confirm across many scenes, then localize with L2/L3.")
    else:
        print("NO metric separates from the permutation null. The note's effect is "
              "indistinguishable from rerun variance.")

    wbm = summary.get("obj_set_diff", {}).get("within_baseline", [float("nan")])[0]
    if wbm == wbm and wbm > 0.01:
        print(f"\nDETERMINISM: baselines differ from EACH OTHER (obj_set_diff={wbm:.3f}). "
              f"If you ran --det-check (same seed), the pipeline is NOT reproducible: "
              f"utils_copy_img.generate_random_seed() feeds 'Random seed: N' into the PROMPT, "
              f"so unseeded RNG => different prompts every run. adv_common.seed_everything() "
              f"now seeds random/numpy/torch per run -- re-run --det-check; obj_set_diff "
              f"should drop to ~0. If it does NOT, the residue is kernel/MoE nondeterminism.")
    print("\nCAVEAT: permutation over runs assumes runs are exchangeable under the null; "
          "scenes are pooled inside each metric. Scale scenes AND repeats before claiming.\n")

    out = os.path.join(args.dir, "l1_variance.json")
    json.dump(summary, open(out, "w"), indent=2, default=str)
    print(f"[var] -> {out}")


if __name__ == "__main__":
    main()