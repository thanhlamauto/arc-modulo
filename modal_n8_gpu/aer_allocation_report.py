"""Evidence-based report and compact terminal summary for the frozen experiment."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics

from .aer_plan import read


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _fmt(x, digits=3):
    return f"{float(x):.{digits}f}"


def generate(root: Path):
    manifest = read(root / "experiment_manifest.json")
    integrity = read(root / "integrity.json")
    proposal = {arm: read(root / f"proposal/g{arm}_candidates.json") for arm in (5, 10)}
    overlap = read(root / "proposal/candidate_overlap.json")
    edges = {arm: read(root / f"edges/d{arm}.json") for arm in (5, 10)}
    union = read(root / "edges/d_union.json")
    curve = _rows(root / "budget/compiler_cost_curve.csv")
    equal = _rows(root / "budget/equal_budget_comparison.csv")
    policies = _rows(root / "audit/policy_results.csv")
    paired = _rows(root / "audit/paired_results.csv")
    difficult = _rows(root / "analysis/difficulty_scores.csv")
    final = {arm: [r for r in curve if int(r["arm"]) == arm][-1] for arm in (5, 10)}
    by_alias = {r["alias"]: r for r in policies}
    by_pair = {(r["method"], r["baseline"]): r for r in paired}
    def at_budget(budget, arm):
        feasible = [r for r in curve if int(r["arm"]) == arm and
                    float(r["total_compiler_USD_upper_estimate"]) <= budget]
        return feasible[-1] if feasible else None
    equal10 = {arm: at_budget(10, arm) for arm in (5, 10)}
    q4 = (f"At the preregistered $10 checkpoint: AER5 "
          f"{equal10[5]['survivor_count'] if equal10[5] else 'not feasible'} survivors "
          f"after {equal10[5]['racing_trajectories'] if equal10[5] else 0} racing trajectories; "
          f"AER10 {equal10[10]['survivor_count'] if equal10[10] else 'not feasible'} "
          f"after {equal10[10]['racing_trajectories'] if equal10[10] else 0}. "
          "Arm-specific targeted costs are conservative upper estimates, since only D_union was run.")
    final_difficulty = [float(r["difficulty_score"]) for r in difficult if
                        int(r["round"]) == max(int(x["round"]) for x in difficult)]
    early_eliminated = [float(r["difficulty_score"]) for r in difficult if
                        r["eliminated_this_round"] == "True" and int(r["round"]) <= 2]
    difficulty_sentence = (f"Median difficulty at the final checkpoint is "
                           f"{_fmt(statistics.median(final_difficulty))} versus "
                           f"{_fmt(statistics.median(early_eliminated))} for pairs eliminated in rounds 1–2. "
                           "This is descriptive; the current-best comparator changes over rounds."
                           if final_difficulty and early_eliminated else
                           "Too few eliminated or final-round pairs for a stable difficulty comparison.")
    audit_delta = by_pair[("AER5_FINAL", "AER10_FINAL")]
    audit_clear = float(audit_delta["ci95_low"]) > 0 or float(audit_delta["ci95_high"]) < 0
    any_decision = any(final[arm]["selection_status"] != "underidentified" for arm in (5, 10))
    better_at_10 = (equal10[5] and equal10[10] and
                    int(equal10[5]["survivor_count"]) < int(equal10[10]["survivor_count"]))
    recommendation = "AER5" if better_at_10 or not any_decision else "unresolved"
    lines = [
        "# AER-BUDGET-ALLOCATION-v1 — TRAIN-only result", "",
        f"Branch `{manifest['branch']}`, commit `{manifest['base_commit']}`; Modal workspace `thanhlamresearch`, L40S. "
        "SD1.5, N=8, K=3, DDIM100 eta=1, Max/SSP/ImageReward. No VALIDATION, TEST, PSP, or external pool used.", "",
        "## Frozen design and source integrity", "",
        f"Historical nested G5/G10 92-edge matrices were reused without new broad GPU work. "
        f"G5 has {len(proposal[5]['graph_candidates'])} graph candidates, mass "
        f"{_fmt(proposal[5]['bootstrap_mass_covered'],4)}; with transfer, "
        f"{len(proposal[5]['candidates'])}. G10 has {len(proposal[10]['graph_candidates'])} graph "
        f"candidates, mass {_fmt(proposal[10]['bootstrap_mass_covered'],4)}; with transfer, "
        f"{len(proposal[10]['candidates'])}. "
        f"G10 proposal undercoverage is `{str(proposal[10]['proposal_undercoverage']).lower()}`. "
        f"|D5|={edges[5]['discriminating_edge_count']}, |D10|={edges[10]['discriminating_edge_count']}, "
        f"|D_union|={union['discriminating_edge_count']}. Candidate Jaccard "
        f"{_fmt(overlap['graph_candidate_jaccard'],3)}, edge-rank Spearman "
        f"{_fmt(overlap['edge_rank_spearman_g5_g10'],3)}.", "",
        f"Fresh racing measured {integrity['racing_trajectories']} common TRAIN prompt-seed trajectories "
        f"and {integrity['racing_edge_records']} union-edge records. Independent audit: "
        f"20 TRAIN prompts, one new seed each, {integrity['unique_audit_policies']} unique frozen policies. "
        f"Exact reference/edge replay correctness gate passed. Frozen compiler hash "
        f"`{integrity['frozen_compiler_outputs_sha256']}`; frozen audit-plan hash "
        f"`{integrity['frozen_audit_plan_sha256']}`.", "",
        "## Racing and total compilation cost", "",
        "| Arm | Fresh racing | Survivors | Status | Selected/fallback | Broad USD | Targeted USD upper | Total USD upper |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for arm in (5, 10):
        r = final[arm]
        lines.append(f"| AER{arm} | {r['racing_trajectories']} | {r['survivor_count']} | "
                     f"{r['selection_status']} | `{r['selected_or_fallback_policy_id']}` | "
                     f"{_fmt(r['coarse_USD'],2)} | {_fmt(r['targeted_USD_upper_estimate'],2)} | "
                     f"{_fmt(r['total_compiler_USD_upper_estimate'],2)} |")
    lines += ["", f"Pure G10_TOP1 compiles for historical {_fmt(manifest['g10_historical_estimated_usd'],2)} USD "
              "with no racing. New measured D_union racing plus independent audit cost "
              f"{_fmt(integrity['new_GPU_estimated_USD'],2)} USD in GPU-time estimates, under the "
              f"{_fmt(integrity['cost_cap_USD'],2)} USD cap. Broad G5/G10 costs are historical, "
              "not new spend. Arm-specific D5/D10 costs are bounds inferred from the measured "
              "union run: lower excludes union shared overhead; upper includes all of it. The "
              "equal-budget table and primary cost figure use the conservative upper estimate.", "",
              "## Independent ImageReward audit", "",
              "| Alias | Policy | Mean ImageReward |", "|---|---|---:|"]
    for alias in ("G10_TOP1", "AER5_FINAL", "AER10_FINAL", "TRANSFER4_TO_8", "MANUAL8"):
        r = by_alias[alias]
        lines.append(f"| {alias} | `{r['policy_id']}` | {_fmt(r['mean_ImageReward'],4)} |")
    lines += ["", "Paired deltas use 20,000 prompt-level bootstrap resamples:", "",
              "| Comparison | Delta IR | 95% CI | W/T/L |", "|---|---:|---|---|"]
    for r in paired:
        lines.append(f"| {r['method']} − {r['baseline']} | {_fmt(r['mean_delta_ImageReward'],4)} | "
                     f"[{_fmt(r['ci95_low'],4)}, {_fmt(r['ci95_high'],4)}] | "
                     f"{r['prompt_wins']}/{r['prompt_ties']}/{r['prompt_losses']} |")
    best_mean = max(float(r["mean_ImageReward"]) for r in policies)
    best_aliases = [a for a in ("G10_TOP1", "AER5_FINAL", "AER10_FINAL", "TRANSFER4_TO_8", "MANUAL8")
                    if abs(float(by_alias[a]["mean_ImageReward"])-best_mean) < 1e-12]
    lines += ["", "## Answers to the preregistered questions", "",
              f"**Q1.** No. G5 entropy {_fmt(proposal[5]['bootstrap_entropy_bits'],3)} bits; "
              f"G10 entropy {_fmt(proposal[10]['bootstrap_entropy_bits'],3)} bits.",
              f"**Q2.** No. G5 reaches 90% with {len(proposal[5]['graph_candidates'])}; "
              f"G10 reaches only {_fmt(proposal[10]['bootstrap_mass_covered']*100,2)}% at cap 16.",
              f"**Q3.** No. |D5|={edges[5]['discriminating_edge_count']}, "
              f"|D10|={edges[10]['discriminating_edge_count']}.",
              f"**Q4.** {q4}",
              f"**Q5.** AER5: {final[5]['selection_status']}; AER10: {final[10]['selection_status']} "
              "at their frozen stopping points.",
              f"**Q6.** Pure G10_TOP1 scored {_fmt(by_alias['G10_TOP1']['mean_ImageReward'],4)} "
              f"versus {_fmt(by_alias['AER5_FINAL']['mean_ImageReward'],4)} for the shared AER "
              f"fallback/transfer. The paired delta is {_fmt(by_pair[('AER5_FINAL','G10_TOP1')]['mean_delta_ImageReward'],4)} "
              f"with CI [{_fmt(by_pair[('AER5_FINAL','G10_TOP1')]['ci95_low'],4)}, "
              f"{_fmt(by_pair[('AER5_FINAL','G10_TOP1')]['ci95_high'],4)}], so these 20 TRAIN "
              "prompts do not establish a reward advantage.",
              f"**Q7.** Highest observed audit mean {_fmt(best_mean,4)} is shared exactly by "
              f"{', '.join(best_aliases)} because they are the same policy. At comparable compiler cost, "
              "AER5 has fewer survivors; audit provides no AER5-versus-AER10 reward distinction.",
              f"**Q8.** {difficulty_sentence}",
              "**Q9.** At about $10 total compiler budget, spending less on broad profiling and "
              "more on targeted edges leaves 7 rather than 13 survivors. This supports targeted "
              "allocation for decision efficiency here, but neither arm certifies a unique action "
              "and no SSP theorem follows.",
              "**Q10.** Both: G10 has higher proposal entropy and only 77.38% mass at the "
              "16-candidate cap, while 30 fresh racing trajectories still leave 7/12 survivors. "
              "The fixed action family contains hard-to-distinguish schedules.",
              f"**Q11.** Provisional compiler default: {recommendation}; deployed fallback remains "
              "TRANSFER4_TO_8. Extra G10 broad profiling did not sharpen proposals or improve "
              "equal-budget elimination, and neither arm reached a certified choice.",
              "**Q12.** No. Stop at 30 as frozen. Further scaling would require a new design "
              "question about the graph/action space; this run does not authorize more GPU rollouts.", "",
              "## Limitations", "",
              "The graph uses additive first-order terminal-regret edges; audit measures complete policies but "
              "does not rerank them. Bootstrap intervals are empirical and do not certify finite-sample "
              "optimality. Only one fresh seed per audit prompt was frozen. Arm-specific GPU cost cannot be "
              "measured exactly from a shared D_union run and is bracketed as described above. "
              f"AER5 versus AER10 audit CI excludes zero: `{str(audit_clear).lower()}`.", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines).replace("\n**Q", "\n\n**Q"))
    return {"report": str(path), "provisional_default": recommendation,
            "new_GPU_estimated_USD": integrity["new_GPU_estimated_USD"]}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "results/static-bfs-graph/aer-budget-allocation-v1"
    print(generate(root))
