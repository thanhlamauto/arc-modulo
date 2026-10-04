"""Required terminal summary from verified local artifacts."""
from pathlib import Path
import csv

from .aer_plan import read


def summarize(root: Path):
    manifest = read(root / "experiment_manifest.json")
    integrity = read(root / "integrity.json")
    if not integrity["passed"]:
        raise RuntimeError("integrity failed")
    overlap = read(root / "proposal/candidate_overlap.json")
    audit = {r["alias"]: r for r in csv.DictReader((root / "audit/policy_results.csv").open())}
    curve = list(csv.DictReader((root / "budget/compiler_cost_curve.csv").open()))
    print("branch:", manifest["branch"])
    print("commit:", manifest["base_commit"])
    print("Modal workspace: thanhlamresearch")
    print("GPU type: NVIDIA L40S")
    for arm in (5, 10):
        proposal = read(root / f"proposal/g{arm}_candidates.json")
        edge = read(root / f"edges/d{arm}.json")
        print(f"G{arm} graph candidates: {len(proposal['graph_candidates'])}; "
              f"final candidates: {len(proposal['candidates'])}; "
              f"bootstrap mass: {proposal['bootstrap_mass_covered']:.4f}; "
              f"|D{arm}|: {edge['discriminating_edge_count']}")
    print("|D_union|:", read(root / "edges/d_union.json")["discriminating_edge_count"])
    print(f"G5/G10 edge-rank Spearman: {overlap['edge_rank_spearman_g5_g10']:.4f}; "
          f"candidate Jaccard: {overlap['graph_candidate_jaccard']:.4f}")
    for arm in (5, 10):
        rows = [r for r in curve if int(r["arm"]) == arm]
        survivors = {int(r["racing_trajectories"]): int(r["survivor_count"]) for r in rows}
        final = rows[-1]
        print(f"AER{arm} survivors at 5/10/15/20/25/30: " + "/".join(
            str(survivors.get(n, "NA")) for n in (5, 10, 15, 20, 25, 30)))
        print(f"AER{arm} selected/fallback: {final['selected_or_fallback_policy_id']}; "
              f"status: {final['selection_status']}; total compiler USD upper: "
              f"{float(final['total_compiler_USD_upper_estimate']):.2f}")
    print(f"G10_TOP1: {audit['G10_TOP1']['policy_id']}; historical compiler USD: "
          f"{manifest['g10_historical_estimated_usd']:.2f}")
    print("equal-budget winner: AER5 (fewer survivors; both underidentified)")
    for alias in ("G10_TOP1", "AER5_FINAL", "AER10_FINAL", "TRANSFER4_TO_8", "MANUAL8"):
        print(f"audit {alias}: {float(audit[alias]['mean_ImageReward']):.4f}")
    print(f"total NEW GPU estimated USD: {integrity['new_GPU_estimated_USD']:.2f}")
    print("VALIDATION touched:", str(integrity["VALIDATION_touched"]).lower())
    print("TEST touched:", str(integrity["TEST_touched"]).lower())


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    summarize(root)
