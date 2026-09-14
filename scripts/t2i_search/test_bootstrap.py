import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bootstrap import bootstrap_artifacts
from core import ExperimentSpec


REPO_ROOT = Path(__file__).resolve().parents[2]


class BootstrapTests(unittest.TestCase):
    def test_reuses_verified_cache_and_copies_frozen_gamma_evidence(self):
        base = ExperimentSpec.load(REPO_ROOT / "configs/t2i_search.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            cache = root / "cache"
            asset = cache / "ImageReward/test.bin"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"verified model data")
            baseline.mkdir()
            (baseline / "asset_manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": "ImageReward/test.bin",
                                "bytes": asset.stat().st_size,
                                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                            }
                        ]
                    }
                )
            )
            spec = base.override_paths(
                run_root=root / "run",
                baseline_results=baseline,
                cache_root=cache,
            )
            evidence = REPO_ROOT / "results/t2i-search/classical-search/table1_reproduced.csv"
            status = bootstrap_artifacts(spec, gamma_evidence=evidence)
            self.assertTrue(status["ready"])
            self.assertEqual(status["verified_assets"], 1)
            self.assertTrue((spec.output / "restored_assets.json").is_file())
            self.assertTrue((spec.output / "restored_environment.txt").is_file())
            self.assertEqual(
                (spec.run_root / "harness/provenance/table1_reproduced.csv").read_bytes(),
                evidence.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
