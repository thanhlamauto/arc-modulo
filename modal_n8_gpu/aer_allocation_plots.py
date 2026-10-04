"""Six standalone scientific figures for AER budget allocation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aer_plan import read


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def generate(root: Path):
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    colors = {5: "#1673b1", 10: "#df7d19"}
    def save(name):
        plt.tight_layout()
        plt.savefig(out / name, dpi=180)
        plt.close()
    plt.figure(figsize=(7, 4.5))
    for arm in (5, 10):
        p = read(root / f"proposal/g{arm}_candidates.json")
        counts = sorted(p["all_bootstrap_winner_counts"].values(), reverse=True)
        mass = np.cumsum(counts)/5000
        plt.plot(np.arange(1, len(mass)+1), mass, label=f"G{arm}", color=colors[arm])
        plt.scatter([min(16, len(mass))], [mass[min(16, len(mass))-1]], color=colors[arm])
    plt.axhline(.9, linestyle="--", color="gray", linewidth=1)
    plt.axvline(16, linestyle=":", color="gray", linewidth=1)
    plt.xlabel("Number of graph candidates")
    plt.ylabel("Cumulative bootstrap selection mass")
    plt.title("Proposal concentration")
    plt.legend()
    save("01_bootstrap_mass.png")

    curve = rows(root / "budget/compiler_cost_curve.csv")
    plt.figure(figsize=(7, 4.5))
    for arm in (5, 10):
        data = [x for x in curve if int(x["arm"]) == arm]
        plt.step([0]+[int(x["racing_trajectories"]) for x in data],
                 [int(data[0]["coarse_candidates"])]+[int(x["survivor_count"]) for x in data],
                 where="post", marker="o", color=colors[arm], label=f"AER{arm}")
    plt.xlabel("Fresh racing trajectories")
    plt.ylabel("Surviving schedules")
    plt.title("Sequential elimination at equal sample counts")
    plt.legend()
    save("02_survivors_vs_trajectories.png")

    plt.figure(figsize=(7, 4.5))
    for arm in (5, 10):
        data = [x for x in curve if int(x["arm"]) == arm]
        plt.step([float(data[0]["coarse_USD"])]+[float(x["total_compiler_USD_upper_estimate"]) for x in data],
                 [int(data[0]["coarse_candidates"])]+[int(x["survivor_count"]) for x in data],
                 where="post", marker="o", color=colors[arm], label=f"AER{arm}")
    plt.xlabel("Total compiler cost (USD; arm subset upper estimate)")
    plt.ylabel("Surviving schedules")
    plt.title("Broad versus targeted allocation at equal budget")
    plt.legend()
    save("03_survivors_vs_total_compiler_cost.png")

    plt.figure(figsize=(7, 4.5))
    final = [next(x for x in reversed(curve) if int(x["arm"]) == arm) for arm in (5, 10)]
    broad = [float(x["coarse_USD"]) for x in final]
    race = [float(x["targeted_USD_upper_estimate"]) for x in final]
    pos = np.arange(2)
    plt.bar(pos, broad, color="#47657c", label="Broad 92-edge calibration")
    plt.bar(pos, race, bottom=broad, color="#e6a04c", label="Targeted racing upper estimate")
    plt.xticks(pos, ["AER5", "AER10"])
    plt.ylabel("Total compiler cost (USD)")
    plt.title("Compiler cost decomposition")
    plt.legend()
    save("04_compiler_cost_decomposition.png")

    audit = rows(root / "audit/policy_results.csv")
    order = ["G10_TOP1", "AER5_FINAL", "AER10_FINAL", "TRANSFER4_TO_8", "MANUAL8"]
    lookup = {x["alias"]: x for x in audit}
    plt.figure(figsize=(8, 4.5))
    vals = [float(lookup[x]["mean_ImageReward"]) for x in order]
    plt.bar(range(5), vals, color=["#7d7d7d", colors[5], colors[10], "#519368", "#b46791"])
    for i, value in enumerate(vals):
        plt.text(i, value+0.012, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, max(vals)*1.10)
    plt.xticks(range(5), order, rotation=20, ha="right")
    plt.ylabel("Mean ImageReward on 20 independent TRAIN prompts")
    plt.title("Frozen-policy audit (descriptive means)")
    save("05_audit_reward.png")

    difficult = rows(root / "analysis/difficulty_scores.csv")
    plt.figure(figsize=(7, 4.5))
    for arm in (5, 10):
        data = [x for x in difficult if int(x["arm"]) == arm]
        x = [int(r["round"]) + (-.07 if arm == 5 else .07) for r in data]
        y = [max(float(r["difficulty_score"]), 1e-8) for r in data]
        markers = [r["eliminated_this_round"] == "True" for r in data]
        plt.scatter([v for v, mark in zip(x, markers) if not mark],
                    [v for v, mark in zip(y, markers) if not mark],
                    color=colors[arm], alpha=.55, s=22, label=f"AER{arm} survived")
        plt.scatter([v for v, mark in zip(x, markers) if mark],
                    [v for v, mark in zip(y, markers) if mark],
                    color=colors[arm], marker="x", s=42, label=f"AER{arm} eliminated")
    plt.yscale("log")
    plt.xticks(range(1, 7), [5, 10, 15, 20, 25, 30])
    plt.xlabel("Racing trajectories at checkpoint")
    plt.ylabel("Paired variance / max(|mean gap|, 1e-6)^2")
    plt.title("Descriptive comparison difficulty")
    plt.legend(fontsize=8, ncol=2)
    save("06_difficulty_vs_elimination.png")
    return sorted(p.name for p in out.glob("*.png") if not p.name.startswith("._"))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    print(generate(root))
