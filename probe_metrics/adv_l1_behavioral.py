#!/usr/bin/env python3
# =============================================================================
# adv_l1_behavioral.py  --  LEVEL 1: does the token propagate to the OUTPUTS?
# -----------------------------------------------------------------------------
# Offline diff of BASELINE vs ADVERSARIAL snapshots. No activations, no model.
# Answers, cheaply and first: does the adversarial token change what the pipeline
# generates, and at which stage does divergence first appear / grow / die?
#
# Reported:
#   * call counts (a structural change = the token altered object/region count)
#   * first call index whose text changed (where propagation first manifests)
#   * per placement call: text changed?, per-object centroid shift, occupancy
#     Hamming, legality (overlap) flip, refinement board-diff difference
#
#   python adv_l1_behavioral.py --dir adv_out
# =============================================================================
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import adv_common as C


def stage_diff(rec_a, rec_b):
    stream, ppairs, (na, nb) = C.align(rec_a, rec_b)

    first_div = None
    for i, (a, b) in enumerate(stream):
        if a["output_text"].strip() != b["output_text"].strip():
            first_div = i
            break

    per = []
    for a, b in ppairs:
        oa, ob = C.parse_objs(a["output_text"]), C.parse_objs(b["output_text"])
        shared = set(oa) & set(ob)
        shift = float(np.mean([np.hypot(oa[l][0] - ob[l][0], oa[l][1] - ob[l][1])
                               for l in shared])) if shared else float("nan")
        occ_a = a["labels"].get("occupancy_str", [])
        occ_b = b["labels"].get("occupancy_str", [])
        ham = int(sum(x != y for x, y in zip(occ_a, occ_b))) if occ_a and occ_b else None
        ch_a = a["labels"].get("changed")
        ch_b = b["labels"].get("changed")
        refine_diff = (int(sum(bool(x) != bool(y) for x, y in zip(ch_a, ch_b)))
                       if ch_a and ch_b else None)
        per.append({
            "key": list(C.placement_key(a)),
            "iteration": a["ctx"].get("iteration"),
            "text_changed": a["output_text"].strip() != b["output_text"].strip(),
            "obj_centroid_shift": shift,
            "occupancy_hamming": ham,
            "legality_a": a["labels"].get("legality"),
            "legality_b": b["labels"].get("legality"),
            "legality_flipped": a["labels"].get("legality") != b["labels"].get("legality"),
            "refine_boarddiff_change": refine_diff,
        })

    shifts = [p["obj_centroid_shift"] for p in per if p["obj_centroid_shift"] == p["obj_centroid_shift"]]
    hams = [p["occupancy_hamming"] for p in per if p["occupancy_hamming"] is not None]
    return {
        "n_calls": (na, nb),
        "structural_divergence": na != nb,
        "first_divergent_call_index": first_div,
        "n_placement_pairs": len(per),
        "placement_text_changed": int(sum(p["text_changed"] for p in per)),
        "legality_flips": int(sum(bool(p["legality_flipped"]) for p in per)),
        "mean_centroid_shift": float(np.mean(shifts)) if shifts else float("nan"),
        "mean_occupancy_hamming": float(np.mean(hams)) if hams else float("nan"),
        "per_placement": per,
    }


def report(d):
    na, nb = d["n_calls"]
    print("\n=============== L1: BEHAVIORAL PROPAGATION ===============")
    print(f"calls: baseline={na}  adversarial={nb}"
          + ("   <-- STRUCTURAL divergence (object/region count changed)"
             if d["structural_divergence"] else ""))
    fd = d["first_divergent_call_index"]
    print(f"first divergent call index : "
          f"{fd if fd is not None else 'NONE (no output text changed anywhere)'}")
    print(f"placement calls compared   : {d['n_placement_pairs']}")
    print(f"  text changed             : {d['placement_text_changed']}")
    print(f"  legality (overlap) flips : {d['legality_flips']}")
    print(f"  mean centroid shift (m)  : {d['mean_centroid_shift']:.3f}")
    print(f"  mean occupancy Hamming   : {d['mean_occupancy_hamming']:.3f}")
    behav = (fd is not None) or d["structural_divergence"] or d["placement_text_changed"] > 0
    print("\n  -> "
          + ("PROPAGATES to outputs. Proceed to L2 to see where in the activations "
             "the change lives, then L3 to localize."
             if behav else
             "NO behavioral change. The token did not alter any output. Run L2: if "
             "activations also don't move, it is ignored at ingestion; if they move "
             "but outputs don't, it is encoded-then-discarded."))
    print("=========================================================\n")
    return behav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with baseline.pkl + adversarial.pkl")
    args = ap.parse_args()
    rec_a = C.load_snapshot(os.path.join(args.dir, "baseline.pkl"))
    rec_b = C.load_snapshot(os.path.join(args.dir, "adversarial.pkl"))
    d = stage_diff(rec_a, rec_b)
    report(d)
    out = os.path.join(args.dir, "l1_behavioral.json")
    json.dump(d, open(out, "w"), indent=2, default=str)
    print(f"[l1] -> {out}")


if __name__ == "__main__":
    main()