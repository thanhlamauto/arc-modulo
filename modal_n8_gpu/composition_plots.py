"""Plots for the frozen, TRAIN-only paired composition diagnostic."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot(root: Path):
    analysis = root / "analysis"
    edges = rows(analysis / "edge_old_vs_new.csv")
    policies = rows(analysis / "policy_residuals.csv")
    paired = rows(analysis / "paired_prompt_policy_rows.csv")
    labels = [x["policy"] for x in policies]
    colors = ["#2563eb", "#ea580c", "#7c3aed", "#059669"]

    nonzero_edges = [row for row in edges if float(row["old_mean_delta"]) != 0
                     or float(row["new_mean_delta"]) != 0]
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes((.09, .13, .55, .76))
    x = np.array([float(row["old_mean_delta"]) for row in nonzero_edges])
    y = np.array([float(row["new_mean_delta"]) for row in nonzero_edges])
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    gap = max((hi - lo) * .08, .003)
    ax.plot([lo-gap, hi+gap], [lo-gap, hi+gap], color="0.6", linestyle="--")
    ax.scatter(x, y, color="#2563eb", s=45)
    for i, (xi, yi) in enumerate(zip(x, y), 1):
        ax.annotate(str(i), (xi, yi), fontsize=9,
                    xytext=(4, -12) if i == 7 else (4, 4),
                    textcoords="offset points")
    ax.set(xlabel="Old 20: mean edge regret", ylabel="New 20: mean edge regret",
           title="Edge estimation on unseen TRAIN prompts", xlim=(lo-gap, hi+gap),
           ylim=(lo-gap, hi+gap))
    fig.text(.67, .88, "Numbered edges (zero edges omitted)", fontsize=10,
             fontweight="bold")
    for i, row in enumerate(nonzero_edges, 1):
        fig.text(.67, .88 - i * .075, f"{i}. {row['edge_key']}", fontsize=9)
    fig.savefig(analysis / "edge_old_vs_new.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row, color in zip(policies, colors):
        px = float(row["new_mean_predicted_regret"])
        ay = float(row["new_mean_actual_regret"])
        ax.scatter(px, ay, color=color, label=row["policy"], s=90)
        ax.plot([px, px], [px, ay], color=color, alpha=.6)
    xx = [float(row["new_mean_predicted_regret"]) for row in policies]
    yy = [float(row["new_mean_actual_regret"]) for row in policies]
    lo, hi = min(xx + yy), max(xx + yy)
    gap = max((hi-lo)*.1, .005)
    ax.plot([lo-gap, hi+gap], [lo-gap, hi+gap], color="0.6", linestyle="--")
    ax.set(xlabel="Mean predicted regret (sum of edges)",
           ylabel="Mean actual full-policy regret",
           title="Frozen policies: predicted versus actual", xlim=(lo-gap, hi+gap),
           ylim=(lo-gap, hi+gap))
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis / "policy_predicted_vs_actual.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [[float(row["composition_residual"]) for row in paired
             if row["policy"] == name] for name in labels]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0, color="0.6", linestyle="--")
    ax.set(ylabel="Actual regret − sum of edge regrets",
           title="Prompt-level composition residuals (new TRAIN)")
    fig.tight_layout()
    fig.savefig(analysis / "policy_residuals.png", dpi=180)
    plt.close(fig)
    return [analysis / name for name in ("edge_old_vs_new.png",
                                         "policy_predicted_vs_actual.png",
                                         "policy_residuals.png")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    for path in plot(parser.parse_args().run):
        print(path)
