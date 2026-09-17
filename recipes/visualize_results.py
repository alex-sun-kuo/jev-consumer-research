"""Seaborn A/B plots using matched, available profile pairs."""
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns
import math

from recipes.run_study import valid_probability


def chart_data(rows, kind="ad"):
    """Use only matched, available A/B pairs for one surface."""
    pairs = {}
    for row in rows:
        if row["kind"] != kind:
            continue
        pair = pairs.setdefault(row["profile_id"], {})
        if row["variant"] in pair:
            raise ValueError("duplicate_profile_variant")
        pair[row["variant"]] = row
    if not pairs:
        raise ValueError("no_rows_for_surface")
    actions = ("ignore", "engage", "research", "defer", "unknown")
    complete = []
    for profile_id, pair in sorted(pairs.items()):
        if any(variant not in pair or pair[variant]["status"] != "ok" for variant in ("A", "B")):
            continue
        distributions = []
        for variant in ("A", "B"):
            probabilities = pair[variant]["answers"]["action"]["probabilities"]
            if (set(probabilities) != set(actions)
                    or not all(valid_probability(p) for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5)):
                raise ValueError("invalid_action_distribution")
            distributions.append(probabilities)
        complete.append((profile_id, *distributions))
    count = len(complete)
    return {
        "kind": kind,
        "modes": sorted({row["mode"] for pair in pairs.values() for row in pair.values()}),
        "complete_pairs": count,
        "excluded_pairs": len(pairs) - count,
        "actions": [(action, sum(a[action] for _, a, _ in complete) / count,
                     sum(b[action] for _, _, b in complete) / count) for action in actions] if count else [],
        "profiles": [(profile, a["engage"], b["engage"]) for profile, a, b in complete],
    }



def plot_charts(data):
    """Return a Matplotlib figure; Seaborn plots existing scores without error bars."""
    height = max(5.5, 2.3 + 0.65 * len(data["profiles"]))
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"), plt.rc_context({"text.parse_math": False}):
        figure, axes = plt.subplots(1, 2, figsize=(12, height), layout="constrained")
        mode = ", ".join(data["modes"]).replace("_", " ")
        figure.suptitle(
            f"A/B results · {data['kind']}\n{mode} · {data['complete_pairs']} complete pairs · "
            f"{data['excluded_pairs']} missing/unavailable pairs excluded",
            fontsize=13,
        )
        figure.supxlabel("Model scores, not observed consumer rates · Equal weight per complete profile", fontsize=10)
        if not data["complete_pairs"]:
            for axis in axes:
                axis.set_axis_off()
            axes[0].text(0.5, 0.5, "No complete A/B pairs to plot", ha="center", transform=axes[0].transAxes)
            return figure

        palette = dict(zip(("A", "B"), sns.color_palette("colorblind", 2)))
        panels = (
            ("Action probabilities (PMF)", data["actions"], "Action", "Mean probability", False),
            ("Engagement by profile", data["profiles"], "Profile · B−A difference", "Engagement score", True),
        )
        for axis, (title, values, ylabel, xlabel, show_delta) in zip(axes, panels):
            labels = [fill(label, width=24) + (f"\nB−A {b-a:+.3f}" if show_delta else "")
                      for label, a, b in values]
            # Use numeric row IDs so long/wrapped labels never merge separate profiles.
            frame = {"row": [], "variant": [], "score": []}
            for index, (_, a, b) in enumerate(values):
                for variant, value in (("A", a), ("B", b)):
                    frame["row"].append(index)
                    frame["variant"].append(variant)
                    frame["score"].append(value)
            sns.barplot(data=frame, x="score", y="row", hue="variant", orient="h",
                        order=list(range(len(values))), hue_order=["A", "B"], palette=palette,
                        saturation=1, errorbar=None, gap=0.15, ax=axis)
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1.1))
            axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
            axis.set_yticks(range(len(labels)), labels=labels)
            axis.grid(axis="y", visible=False)
            axis.legend(title="Variant", loc="lower right", frameon=False)
            for index, container in enumerate(axis.containers):
                if index == 1:
                    for bar in container:
                        bar.set_hatch("//")
                axis.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
            sns.despine(ax=axis, left=True)
        return figure
