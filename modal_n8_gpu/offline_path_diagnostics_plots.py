"""Static figures for the existing-data additive-path diagnostics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make(root: Path):
    folder = root / "diagnostics"
    folder.mkdir(exist_ok=True)
    ranking = rows(folder / "predicted_vs_audit_rank.csv")
    names = ["P1", "P5=P10", "P20", "TRANSFER4_TO_8"]
    ns = [1, 5, 10, 20]
    rank_map = {(int(row["n"]), row["policy"]): row for row in ranking}
    values = np.array([[int(rank_map[n, name]["predicted_rank"]) for n in ns] +
                       [int(rank_map[20, name]["audit_rank"])] for name in names])
    fig, ax = plt.subplots(figsize=(8.5, 4))
    im = ax.imshow(values, vmin=1, vmax=4, cmap="RdYlGn_r", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=11)
    ax.set_xticks(range(5), ["G1", "G5", "G10", "G20", "Audit"])
    ax.set_yticks(range(4), names)
    ax.set_title("Rank among four frozen static policies (1 = best)")
    ax.axvline(3.5, color="black", lw=1)
    fig.colorbar(im, ax=ax, label="Rank", ticks=[1, 2, 3, 4])
    fig.tight_layout()
    fig.savefig(folder / "01_predicted_vs_audit_rank.png", dpi=180)
    plt.close(fig)

    terms = rows(folder / "p20_vs_transfer_edge_differences.csv")
    positions = np.arange(len(terms))
    means = np.array([float(row["mean_cost_difference"]) for row in terms])
    sem = np.array([float(row["paired_trajectory_SEM"]) for row in terms])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(positions, means, color=["#3b73ad" if v < 0 else "#bd775c" for v in means])
    ax.errorbar(positions, means, yerr=sem, fmt="none", color="#202020", capsize=5)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(positions,
                  [f"{r['P20_edge']}\nvs\n{r['TRANSFER4_TO_8_edge']}" for r in terms],
                  fontsize=8)
    ax.set_ylabel("P20 − transfer edge cost (mean ± paired SEM)")
    ax.set_title("First-order terms favor P20; total margin is uncertain")
    ax.text(.98, .96, "negative favors P20", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "02_p20_transfer_edge_decomposition.png", dpi=180)
    plt.close(fig)
    return sorted(folder.glob("*.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print("\n".join(map(str, make(parser.parse_args().root))))
