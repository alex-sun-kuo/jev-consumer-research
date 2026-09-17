"""Compare matched profiles; do not pool clicks with checkout starts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from recipes.run_study import valid_probability


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    for r in rows:
        key = (r["profile_id"], r["stimulus_id"])
        if key in seen:
            raise ValueError("duplicate_cell")
        seen.add(key)
    fields = ("study_id", "revision", "mode", "demographics_masked")
    if any(len({r[f] for r in rows}) > 1 for f in fields):
        raise ValueError("mixed_studies_or_modes")
    return rows


def engage(row):
    if row["status"] != "ok":
        return None
    p = row["answers"]["action"]["probabilities"]["engage"]
    if not valid_probability(p):
        raise ValueError("invalid_probability")
    return p


def group_values(rows, group_by):
    """Read a scalar demographic field, or an explicit context.<field>."""
    parts = group_by.split(".")
    if len(parts) == 1:
        namespace, field = "demographics", parts[0]
    elif len(parts) == 2:
        namespace, field = parts
    else:
        raise ValueError("group_by_must_be_demographics_or_context_field")
    if namespace not in ("demographics", "context") or not field:
        raise ValueError("group_by_must_be_demographics_or_context_field")
    if rows and not any(field in row.get(namespace, {}) for row in rows):
        raise ValueError(f"unknown_group_by: {group_by}")
    groups = []
    for row in rows:
        value = row.get(namespace, {}).get(field)
        if isinstance(value, (dict, list)):
            raise ValueError("group_by_requires_scalar_field")
        groups.append("unspecified" if value is None or value == "" else str(value))
    return groups


def table_text(value):
    return str(value).replace("|", "/").replace("\n", " ").replace("\r", " ")


def available_groups(rows):
    """Expose only scalar fields that can be used consistently across this panel."""
    result = []
    for section in ("demographics", "context"):
        keys = {key for row in rows for key in row.get(section, {})}
        for key in sorted(keys):
            if all(not isinstance(row.get(section, {}).get(key), (list, dict)) for row in rows):
                result.append(key if section == "demographics" else f"context.{key}")
    return result


def comparisons(rows, group_by="age_band"):
    pairs = defaultdict(dict)
    for row, group in zip(rows, group_values(rows, group_by)):
        key = (row["profile_id"], row["kind"])
        if row["variant"] in pairs[key]:
            raise ValueError("multiple_variants_per_profile_kind")
        pairs[key][row["variant"]] = (row, group)
    groups = defaultdict(lambda: {"pairs": 0, "unavailable_pairs": 0, "deltas": []})
    for (_, kind), pair in pairs.items():
        group = next(iter(pair.values()))[1]
        if any(value[1] != group for value in pair.values()):
            raise ValueError("inconsistent_pair_group")
        bucket = groups[(kind, group)]
        bucket["pairs"] += 1
        a, b = (pair[key][0] if key in pair else None for key in ("A", "B"))
        if a is None or b is None or engage(a) is None or engage(b) is None:
            bucket["unavailable_pairs"] += 1
        else:
            bucket["deltas"].append(engage(b) - engage(a))
    return groups


def report(rows, group_by, *, include_cells=True):
    out = ["# AI focus-group simulation report", "",
           "A positive delta means B received higher simulated immediate-engagement scores.",
           "Each profile has equal scenario weight, not population weight. No significance tests.", "",
           "Groups may contain only one profile; compare needs and budgets before attributing differences to a demographic label.", "",
           f"Grouping: {group_by}. Mode: {rows[0]['mode'] if rows else 'empty'}.", "",
           "| Surface | Group | Complete pairs | Unavailable/missing pairs | Mean B−A score |",
           "| --- | --- | ---: | ---: | ---: |"]
    for (kind, group), value in sorted(comparisons(rows, group_by).items()):
        ds = value["deltas"]
        delta = f"{sum(ds)/len(ds):+.3f}" if ds else "unavailable"
        safe_group = table_text(group)
        out.append(f"| {kind} | {safe_group} | {len(ds)} | {value['unavailable_pairs']} | {delta} |")
    if not include_cells:
        return "\n".join(out) + "\n"
    out += ["", "## Per-cell review", "",
            "Noul values are model yes-scores.", ""]
    for r in sorted(rows, key=lambda x: (x["kind"], x["profile_id"], x["variant"])):
        if r["status"] != "ok":
            out.append(f"- {r['profile_id']} / {r['stimulus_id']}: unavailable")
            continue
        a = r["answers"]
        out.append(f"- {r['profile_id']} / {r['stimulus_id']}: sentiment={a['sentiment']['choice']}; "
                   f"next action={a['action']['choice']}; "
                   f"unknown action mass={a['action']['probabilities']['unknown']:.2f}; "
                   f"price objection={a['price_objection']['noul']:.2f}; "
                   f"proof gap={a['proof_gap']['noul']:.2f}")
    return "\n".join(out) + "\n"


def diagnostic_report(rows, group_by="context.purchase_stage"):
    """Separate message judgments and next-action scores by surface and variant."""
    groups = defaultdict(list)
    for row, group in zip(rows, group_values(rows, group_by)):
        groups[(row["kind"], row["variant"], group)].append(row)
    metrics = {
        "Relevance": ("relevant", "noul"),
        "Price objection": ("price_objection", "noul"),
        "Proof gap": ("proof_gap", "noul"),
        "Confusion": ("confusion", "noul"),
        "Research": ("action", "research"),
        "Defer": ("action", "defer"),
        "Unknown": ("action", "unknown"),
    }
    out = ["# Message and decision diagnostics", "",
           f"Grouping: {group_by}. Means use available scenarios only; unavailable counts are separate.", "",
           "Noul columns are yes-scores; Research, Defer, and Unknown are action-distribution scores.", "",
           "| Surface | Variant | Group | Available | Unavailable | " + " | ".join(metrics) + " |",
           "| --- | --- | --- | ---: | ---: | " + " | ".join(["---:"] * len(metrics)) + " |"]
    for (kind, variant, group), members in sorted(groups.items()):
        available = [row for row in members if row["status"] == "ok"]
        means = []
        for answer, field in metrics.values():
            values = [(row["answers"][answer]["probabilities"][field] if answer == "action"
                       else row["answers"][answer][field]) for row in available]
            if not all(valid_probability(value) for value in values):
                raise ValueError("invalid_probability")
            means.append(f"{sum(values) / len(values):.3f}" if values else "unavailable")
        out.append(f"| {kind} | {variant} | {table_text(group)} | {len(available)} | "
                   f"{len(members) - len(available)} | " + " | ".join(means) + " |")
    return "\n".join(out) + "\n"


def sensitivity(original, masked):
    if not original or not masked:
        raise ValueError("empty_inputs")
    if any(r["demographics_masked"] for r in original) or not all(r["demographics_masked"] for r in masked):
        raise ValueError("expected_visible_then_masked")
    for field in ("study_id", "revision", "mode"):
        if original[0][field] != masked[0][field]:
            raise ValueError("incompatible_runs")
    right = {(r["profile_id"], r["stimulus_id"]): r for r in masked}
    if set(right) != {(r["profile_id"], r["stimulus_id"]) for r in original}:
        raise ValueError("unmatched_cells")
    changes, unavailable = [], 0
    for r in original:
        a, b = engage(r), engage(right[(r["profile_id"], r["stimulus_id"])])
        if a is None or b is None:
            unavailable += 1
        else:
            changes.append(abs(a-b))
    return {"complete_cells": len(changes), "unavailable_cells": unavailable,
            "mean_absolute_score_change": sum(changes)/len(changes) if changes else None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--group-by", default="age_band",
                   help="Demographic field (e.g. income_level) or context.purchase_stage")
    p.add_argument("--masked", type=Path)
    p.add_argument("--diagnostics", action="store_true")
    args = p.parse_args()
    rows = load(args.results)
    print(json.dumps(sensitivity(rows, load(args.masked)), indent=2) if args.masked
          else diagnostic_report(rows, args.group_by) if args.diagnostics else report(rows, args.group_by))
