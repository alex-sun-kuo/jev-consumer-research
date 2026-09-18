"""Read reactions to one stimulus at a time; no A/B pairing required."""
import argparse
from collections import defaultdict
from pathlib import Path

from recipes.run_study import valid_probability
from recipes.summarize import load, table_text

BARRIER_KEYS = ("relevant", "price_objection", "proof_gap", "confusion")


def mean_distribution(rows, question):
    totals = defaultdict(float)
    for row in rows:
        for option, value in row["answers"][question]["probabilities"].items():
            if not valid_probability(value):
                raise ValueError("invalid_probability")
            totals[option] += value
    return {option: total / len(rows) for option, total in totals.items()}


def mean_noul(rows, question):
    values = [row["answers"][question]["noul"] for row in rows]
    if not all(valid_probability(value) for value in values):
        raise ValueError("invalid_probability")
    return sum(values) / len(values)


def stimulus_read(rows):
    """Summarize each stimulus separately; failed requests stay counted, never zero."""
    by_stimulus = defaultdict(list)
    for row in rows:
        by_stimulus[(row["kind"], row["stimulus_id"])].append(row)
    reads = []
    for (kind, stimulus_id), members in sorted(by_stimulus.items()):
        available = sorted((r for r in members if r["status"] == "ok"), key=lambda r: r["profile_id"])
        read = {"kind": kind, "stimulus_id": stimulus_id, "available": len(available),
                "unavailable": len(members) - len(available),
                "sentiment": None, "actions": None, "barriers": None, "profiles": []}
        if available:
            read.update(sentiment=mean_distribution(available, "sentiment"),
                        actions=mean_distribution(available, "action"),
                        barriers={key: mean_noul(available, key) for key in BARRIER_KEYS},
                        profiles=[(r["profile_id"], r["answers"]["action"]["probabilities"]["engage"],
                                   r["answers"]["sentiment"]["choice"]) for r in available])
        reads.append(read)
    return reads


def concept_report(reads):
    out = ["# Concept read", "",
           "Mean model scores per stimulus; each available profile counts once.", ""]
    for read in reads:
        out += [f"## {table_text(read['kind'])} / {table_text(read['stimulus_id'])}", "",
                f"Available profiles: {read['available']}. Unavailable: {read['unavailable']}.", ""]
        if not read["available"]:
            continue
        for title, scores in (("Sentiment", read["sentiment"]), ("Next action", read["actions"]),
                              ("Barrier signal", read["barriers"])):
            out += [f"| {title} | Mean score |", "| --- | ---: |"]
            out += [f"| {table_text(option).replace('_', ' ')} | {value:.3f} |" for option, value in scores.items()]
            out.append("")
        mean_engage = sum(engage for _, engage, _ in read["profiles"]) / len(read["profiles"])
        out.append(f"Engagement by profile (panel mean {mean_engage:.3f}):")
        out += [f"- {table_text(profile)}: engage {engage:.3f} ({engage - mean_engage:+.3f}), sentiment {table_text(choice)}"
                for profile, engage, choice in read["profiles"]]
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(concept_report(stimulus_read(load(args.results))))


if __name__ == "__main__":
    main()
