"""Six publication-readable figures from frozen offline-compiler artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make(root: Path):
    folder = root / "plots"
    folder.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 10})

    fig, ax = plt.subplots(figsize=(12, 2.5))
    labels = ["Dense reference", "Post-event\nsnapshots", "Shared no-search\ntrunks",
              "Shared verifier\nper destination", "Tau forks +\nexact suffixes",
              "Terminal rewards", "Static graph", "Compiled policy"]
    xs = np.linspace(0.06, 0.94, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, .5, label, ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=.35", "fc": "#e8f1ff", "ec": "#3467a4"})
    for a, b in zip(xs, xs[1:]):
        ax.annotate("", xy=(b-.047, .5), xytext=(a+.047, .5),
                    arrowprops={"arrowstyle": "->", "color": "#3467a4"})
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout(); fig.savefig(folder / "01_offline_compiler_architecture.png"); plt.close(fig)

    policies = _read(root / "graphs/policies_by_n.json")
    audit = _read(root / "audit/summary.json")
    plan = _read(root / "audit/audit_policy_plan.json")
    reward = {row["policy_id"]: row["mean_ImageReward"] for row in audit["policy_results"]}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["n"] for r in policies], [reward[r["policy_id"]] for r in policies],
            "o-", lw=2, label="Compiled Pn")
    for name, color in (("MANUAL8", "#aa4d22"), ("TRANSFER4_TO_8", "#366f49")):
        ax.axhline(reward[plan["policy_aliases"][name]], ls="--", color=color, label=name)
    ax.set_xticks([1, 5, 10, 20]); ax.set_xlabel("Calibration trajectories")
    ax.set_ylabel("Held-out mean ImageReward")
    ax.set_ylim(min(reward.values()) - .04, max(reward.values()) + .03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(folder / "02_reward_vs_trajectories.png"); plt.close(fig)

    cost = _read(root / "analysis/compilation_cost.csv")
    fig, left = plt.subplots(figsize=(7, 4))
    ns = [int(r["n"]) for r in cost]
    left.plot(ns, [float(r["gpu_hours"])*60 for r in cost], "o-", color="#345c9c")
    left.set_ylabel("Summed GPU minutes", color="#345c9c")
    right = left.twinx()
    right.plot(ns, [float(r["estimated_modal_usd"]) for r in cost], "s--", color="#9b5a20")
    right.set_ylabel("Estimated Modal USD", color="#9b5a20")
    left.set_xticks([1, 5, 10, 20]); left.set_xlabel("Calibration trajectories")
    fig.tight_layout(); fig.savefig(folder / "03_compiler_cost.png"); plt.close(fig)

    benchmark = _read(root / "replay/benchmark.json")
    naive = float(benchmark["five_edge_naive_seconds"])
    shared = float(benchmark["five_edge_shared_seconds"])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Independent full runs", "Shared-tree profiler"], [naive, shared],
                  color=["#a86a63", "#4779ac"])
    for bar, v in zip(bars, (naive, shared)):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f}s", ha="center", va="bottom")
    ax.set_ylabel("Wall seconds for same five edges")
    ax.set_title(f"Measured speedup {naive/shared:.2f}×")
    fig.tight_layout(); fig.savefig(folder / "04_naive_vs_shared.png"); plt.close(fig)

    convergence_path = root / "analysis/edge_convergence.csv"
    if not convergence_path.exists():
        convergence_path = root / "analysis/edge_convergence_interim.csv"
    convergence = _read(convergence_path) if convergence_path.exists() else []
    fig, ax = plt.subplots(figsize=(7, 4))
    if convergence:
        ax.plot([int(r["n"]) for r in convergence],
                [float(r["spearman_edge_rank"]) for r in convergence], "o-")
    ax.set_xticks([1, 5, 10]); ax.set_ylim(-1, 1); ax.set_ylabel("Spearman edge-rank correlation")
    ax.set_xlabel(f"Calibration trajectories (reference G{policies[-1]['n']})")
    fig.tight_layout(); fig.savefig(folder / "05_edge_rank_convergence.png"); plt.close(fig)

    stability = _read(root / "graphs/bootstrap_stability.csv")
    freq = {int(r["n"]): float(r["exact_path_frequency"]) for r in stability if r["n"] != "1"}
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"P{r['n']}\n{r['steps']}\n{r['taus']}" for r in policies]
    ax.bar(range(1, len(policies)), [freq[r["n"]] for r in policies[1:]],
           color="#4675a8")
    ax.set_xticks(range(len(policies)), labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Bootstrap exact-path frequency")
    ax.text(0, .04, "N/A\n(n=1)", ha="center", color="#333333", fontsize=8)
    fig.tight_layout(); fig.savefig(folder / "06_policy_stability.png"); plt.close(fig)
    return [str(p) for p in sorted(folder.glob("*.png"))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print("\n".join(make(parser.parse_args().root)))
