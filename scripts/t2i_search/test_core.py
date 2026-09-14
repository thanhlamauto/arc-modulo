import json
from pathlib import Path
import tempfile
import unittest

from core import (
    ConfigError,
    ExperimentSpec,
    PairKey,
    Shard,
    plan_summary,
    status_summary,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ExperimentSpec.load(REPO_ROOT / "configs/t2i_search.json")

    def test_full_grid_has_atomic_pairs(self):
        keys = list(self.spec.pairs())
        self.assertEqual(len(keys), 2 * 100 * 4)
        self.assertEqual(len({key.id for key in keys}), len(keys))
        self.assertEqual(plan_summary(self.spec, keys, Shard())["method_runs"], 1600)

    def test_shards_are_disjoint_and_cover_grid(self):
        keys = list(self.spec.pairs())
        shards = [Shard(index, 7).select(keys) for index in range(7)]
        flattened = [key for shard in shards for key in shard]
        self.assertEqual(set(flattened), set(keys))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_pair_method_order_matches_frozen_protocol(self):
        self.assertEqual(self.spec.method_order(PairKey(4, 0, 42)), ("bfs", "vasr"))
        self.assertEqual(self.spec.method_order(PairKey(4, 0, 43)), ("vasr", "bfs"))
        self.assertEqual(self.spec.method_order(PairKey(8, 1, 42)), ("vasr", "bfs"))

    def test_selection_keeps_protocol_order(self):
        keys = list(self.spec.pairs(particles=(8,), seeds=(43, 45), prompt_range=range(2, 4)))
        self.assertEqual(
            keys,
            [
                PairKey(8, 2, 43),
                PairKey(8, 2, 45),
                PairKey(8, 3, 43),
                PairKey(8, 3, 45),
            ],
        )

    def test_invalid_shard_fails_before_runtime(self):
        for value in ("1/1", "-1/2", "0/0", "bad"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                Shard.parse(value)


class ArtifactTests(unittest.TestCase):
    def test_atomic_immutable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json(path, {"value": 1}, immutable=True)
            write_json(path, {"value": 1}, immutable=True)
            with self.assertRaises(ConfigError):
                write_json(path, {"value": 2}, immutable=True)

    def test_status_counts_resume_artifacts_without_accelerator(self):
        base = ExperimentSpec.load(REPO_ROOT / "configs/t2i_search.json")
        with tempfile.TemporaryDirectory() as directory:
            spec = base.override_paths(run_root=Path(directory))
            key = next(spec.pairs())
            pair = spec.output / "pairs" / f"n{key.particles}" / f"seed{key.seed}" / "prompt_000.json"
            pair.parent.mkdir(parents=True)
            pair.write_text(json.dumps({"valid": True}))
            for method in ("bfs", "vasr"):
                raw = spec.output / "raw" / f"n{key.particles}" / method / f"seed{key.seed}" / "prompt_000.json"
                raw.parent.mkdir(parents=True)
                raw.write_text("{}")
            status = status_summary(spec)
            self.assertEqual(status["completed_pairs"], 1)
            self.assertEqual(status["raw_method_records"], {"bfs": 1, "vasr": 1})
            self.assertFalse(status["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
