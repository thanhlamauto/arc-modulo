"""Five research figures for the frozen adaptive-edge-racing experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def _csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path: Path):
    return json.loads(path.read_text())


def make(root: Path):
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    candidate = _json(root / "stage_a/candidate_set.json")
    d = _json(root / "racing/discriminating_edges.json")
    compiled = _json(root / "racing/aer_compiled_policy.json")
    rounds = _csv(root / "racing/round_summary.csv")
    costs = _json(root / "analysis/cost_summary.json")
    audit = _csv(root / "audit/policy_results.csv")

    # Figure 1: pipeline, with frozen sizes from this run.
    fig, ax = plt.subplots(figsize=(12, 2.7))
    ax.axis("off")
    labels = [f"Coarse graph\n5 × 92 edges",
              f"Bootstrap proposals\n5,000 resamples\n{len(candidate['graph_candidates'])} graph paths",
              f"Discriminating edges\n{d['discriminating_edge_count']} of 92",
              f"Fresh racing\n{compiled['racing_trajectories_used']} trajectories",
              f"Frozen static policy\n{compiled['selection_status']}"]
    xs = [.025, .22, .415, .61, .805]
    for x, label in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x, .28), .16, .45,
                     boxstyle="round,pad=.008", facecolor="#eff6ff",
                     edgecolor="#2563eb", transform=ax.transAxes))
        ax.text(x+.08, .505, label, ha="center", va="center", fontsize=10,
                transform=ax.transAxes)
    for x in xs[:-1]:
        ax.annotate("", xy=(x+.19, .505), xytext=(x+.17, .505),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5),
                    xycoords=ax.transAxes)
    fig.tight_layout()
    fig.savefig(plots / "01_adaptive_compiler_diagram.png", dpi=180)
    plt.close(fig)

    # Figure 2: candidate survival.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [0] + [int(row["trajectories"]) for row in rounds]
    y = [len(candidate["candidates"])] + [int(row["survivor_count"]) for row in rounds]
    ax.step(x, y, where="post", color="#2563eb", linewidth=2)
    ax.scatter(x, y, color="#2563eb", s=55)
    ax.set(xlabel="Fresh racing trajectories", ylabel="Surviving candidate paths",
           title="Sequential candidate survival", xticks=[0, 5, 10, 15, 20],
           ylim=(0, len(candidate["candidates"])+1))
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(plots / "02_candidate_survival.png", dpi=180)
    plt.close(fig)

    # Figure 3: all pairs active at the start of each observed round.
    ids = [item["policy_id"] for item in candidate["candidates"]]
    short = {pid: f"C{i+1}" for i, pid in enumerate(ids)}
    pair_rows = _csv(root / "racing/pairwise_intervals.csv")
    for r in range(1, len(rounds)+1):
        subset = [row for row in pair_rows if int(row["round"]) == r]
        subset.sort(key=lambda row: (short[row["policy_p"]], short[row["policy_q"]]))
        height = max(5, .29*len(subset)+1.9)
        fig, ax = plt.subplots(figsize=(9, height))
        pos = np.arange(len(subset))[::-1]
        mean = np.array([float(row["mean_d_p_minus_q"]) for row in subset])
        low = np.array([float(row["ci_lower"]) for row in subset])
        high = np.array([float(row["ci_upper"]) for row in subset])
        colors = ["#dc2626" if lo > 0 or hi < 0 else "#2563eb"
                  for lo, hi in zip(low, high)]
        for p, m, lo, hi, color in zip(pos, mean, low, high, colors):
            ax.plot([lo, hi], [p, p], color=color, lw=1.5)
            ax.scatter([m], [p], color=color, s=25)
        ax.axvline(0, color="0.5", linestyle="--")
        ax.set_yticks(pos, [f"{short[row['policy_p']]}−{short[row['policy_q']]}"
                            for row in subset], fontsize=7)
        ax.set(xlabel="Paired mean path-cost difference with adjusted CI",
               title=f"Round {r}: {5*r} racing trajectories, all active pairs")
        ax.grid(axis="x", alpha=.2)
        legend_items = [f"{short[item['policy_id']]}={item['steps']}/{item['taus']}"
                        for item in candidate["candidates"]]
        legend = "\n".join("  ".join(legend_items[i:i+3])
                           for i in range(0, len(legend_items), 3))
        fig.text(.02, .015, legend, fontsize=6.5, linespacing=1.2)
        fig.tight_layout(rect=(0, .11, 1, 1))
        fig.savefig(plots / f"03_pairwise_round_{r}.png", dpi=180)
        plt.close(fig)

    # Figure 4: independent TRAIN audit means and prompt-bootstrap CI.
    prompt_rows = _csv(root / "audit/prompt_policy_rows.csv")
    rng = np.random.default_rng(20261003)
    picks = rng.integers(0, 20, size=(20_000, 20))
    means = []
    lows = []
    highs = []
    for row in audit:
        vals = np.array([float(x["ImageReward"]) for x in prompt_rows
                         if x["policy_id"] == row["policy_id"]])
        boot = vals[picks].mean(axis=1)
        means.append(float(vals.mean()))
        lo, hi = np.quantile(boot, [.025, .975])
        lows.append(lo)
        highs.append(hi)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(audit))
    colors = ["#2563eb", "#ea580c", "#7c3aed", "#059669", "#64748b"]
    ax.bar(x, means, color=colors[:len(audit)])
    ax.errorbar(x, means, yerr=[np.array(means)-np.array(lows),
                                np.array(highs)-np.array(means)],
                fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x, [row["alias"] for row in audit], rotation=20, ha="right")
    ax.set(ylabel="Terminal ImageReward", title="Independent TRAIN audit: 20 paired prompts")
    fig.tight_layout()
    fig.savefig(plots / "04_audit_reward.png", dpi=180)
    plt.close(fig)

    # Figure 5: total compiler cost includes historically paid Stage A.
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
    left.bar(["Uniform G20", "AER"], [costs["uniform_G20_estimated_Modal_USD"],
                                     costs["AER_total_estimated_Modal_USD"]],
             color=["#64748b", "#2563eb"])
    left.set(ylabel="Estimated Modal USD", title="Total compiler cost (not invoice)")
    right.bar("AER", costs["stage_a_reused_estimated_Modal_USD"],
              color="#94a3b8", label="Stage A, historical")
    right.bar("AER", costs["stage_b_new_estimated_Modal_USD"],
              bottom=costs["stage_a_reused_estimated_Modal_USD"],
              color="#2563eb", label="Stage B, new")
    right.set(ylabel="Estimated Modal USD", title="AER accounting")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "05_compiler_cost.png", dpi=180)
    plt.close(fig)
    return sorted(p for p in plots.glob("*.png") if not p.name.startswith("._"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    for path in make(parser.parse_args().run):
        print(path)
