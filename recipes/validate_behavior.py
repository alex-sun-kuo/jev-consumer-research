"""Evaluate frozen predictions on held-out observed events; does not fit a model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from recipes.run_study import valid_probability


def evaluate(data):
    rows = data["rows"]
    if not rows:
        raise ValueError("empty_holdout")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate_observation")
    baseline = data["baseline_from_training"]
    if not valid_probability(baseline):
        raise ValueError("invalid_baseline")
    if any(type(r["observed"]) is not int or r["observed"] not in (0, 1) for r in rows):
        raise ValueError("observed_must_be_binary")
    if any(not valid_probability(r["prediction"]) for r in rows):
        raise ValueError("invalid_prediction")
    bins = defaultdict(list)
    for r in rows:
        bins[min(int(r["prediction"] * 5), 4)].append(r)
    brier = sum((r["prediction"] - r["observed"])**2 for r in rows)/len(rows)
    base = sum((baseline - r["observed"])**2 for r in rows)/len(rows)
    return {"provenance": data["provenance"], "event": data["event"], "n": len(rows),
            "brier": brier, "baseline_brier": base, "improvement_over_baseline": base-brier,
            "calibration_bins": [{"bin": k, "n": len(rs),
                "mean_prediction": sum(r["prediction"] for r in rs)/len(rs),
                "observed_rate": sum(r["observed"] for r in rs)/len(rs)} for k, rs in sorted(bins.items())]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("observations", type=Path)
    args = p.parse_args()
    print(json.dumps(evaluate(json.loads(args.observations.read_text())), indent=2))
