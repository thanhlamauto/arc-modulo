"""CPU checks for paired AER elimination and conservative fallback."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from modal_n8_gpu import aer_racing


SOURCE = (Path(__file__).resolve().parents[1] /
          "results/static-bfs-graph/adaptive-edge-racing-v1")


def fixture(root: Path, penalty_edge: str | None):
    for name in ("stage_a/candidate_set.json", "racing/discriminating_edges.json",
                 "racing/racing_trajectory_plan.json"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
    plan = json.loads((root / "racing/racing_trajectory_plan.json").read_text())
    keys = json.loads((root / "racing/discriminating_edges.json").read_text())[
        "discriminating_edges"]
    for rank, item in enumerate(plan["trajectories"][:5]):
        folder = root / f"racing/raw/trajectory_{rank:02d}"
        (folder / "edges").mkdir(parents=True)
        (folder / "reference.json").write_text(json.dumps({
            "prompt_index": item["prompt_id"], "trial_seed": item["seed"],
            "selected_final_score": 1.0}))
        (folder / "timing.json").write_text(json.dumps({
            "prompt_id": item["prompt_id"], "seed": item["seed"],
            "edge_count": len(keys)}))
        for key in keys:
            delta = 1.0 if key == penalty_edge else 0.0
            (folder / "edges" / f"{key}.json").write_text(json.dumps({
                "prompt_index": item["prompt_id"], "trial_seed": item["seed"],
                "edge_key": key, "backend": "CUDA_Modal", "delta": delta,
                "selected_final_score": 1.0-delta}))


class RacingLogicTest(unittest.TestCase):
    def test_equal_paths_choose_incumbent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, None)
            state = aer_racing.analyze_round(root, 1)
            candidate_set = json.loads((root / "stage_a/candidate_set.json").read_text())
            self.assertTrue(state["stop"])
            self.assertEqual(state["selection_status"], "practically_equivalent")
            self.assertEqual(state["selected_policy_id"],
                             candidate_set["incumbent_policy_id"])

    def test_dominated_candidates_eliminated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            penalty = "-1_to_40_tau_2"
            fixture(root, penalty)
            state = aer_racing.analyze_round(root, 1)
            candidates = json.loads((root / "stage_a/candidate_set.json").read_text())[
                "candidates"]
            penalized = {x["policy_id"] for x in candidates if penalty in x["edge_keys"]}
            self.assertTrue(penalized)
            self.assertTrue(penalized.isdisjoint(state["survivor_policy_ids"]))
            self.assertTrue({x["policy_id"] for x in state["eliminations"]} >= penalized)


if __name__ == "__main__":
    unittest.main()
