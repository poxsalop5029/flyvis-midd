"""Validate public metadata without running figure inference."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTests(unittest.TestCase):
    def test_metadata_schema_and_copied_checksums(self):
        folders = [ROOT / "data/precomputed" / f"fig{i:02d}" for i in range(1, 7)]
        folders += [ROOT / "data/precomputed/supplementary" / f"figS{i}" for i in range(1, 5)]
        keys = {"figure", "artifact_version", "source_repository", "source_commit_sha",
                "source_paths", "source_files", "cohort", "statistical_unit", "stimulus",
                "rendering_contract", "temporal_contract", "metric", "sign_convention",
                "estimands", "precomputed_policy", "limitations"}
        for folder in folders:
            with self.subTest(figure=folder.name):
                text = (folder / "metadata.json").read_text()
                data = json.loads(text)
                self.assertTrue(keys <= data.keys(), keys - data.keys())
                self.assertNotRegex(text, r"/[U]sers/")
                for record in data["source_files"]:
                    filename = record["filename"]
                    self.assertEqual(Path(filename).name, filename)
                    self.assertEqual(hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                                     record["copied_sha256"], filename)
                    commit = record["source_commit_sha"]
                    if commit is not None:
                        self.assertRegex(commit, r"^[0-9a-f]{40}$")

    def test_fig1_fig2_shared_canonical_contract(self):
        from flyvis_midd.evaluation import CANONICAL_MIDD
        for figure in ("fig01", "fig02"):
            data = json.loads((ROOT / "data/precomputed" / figure / "metadata.json").read_text())
            timing = data["temporal_contract"]
            self.assertEqual(timing["dt"], CANONICAL_MIDD.dt)
            self.assertEqual(timing["fade_seconds"], CANONICAL_MIDD.fade_seconds)
            self.assertEqual(timing["n_frames"], CANONICAL_MIDD.n_frames)
            self.assertEqual((timing["analysis_start"], timing["analysis_stop"]), (10, 50))
            self.assertEqual(data["rendering_contract"]["extent"], CANONICAL_MIDD.eye_extent)
            self.assertEqual(data["rendering_contract"]["kernel_size"], CANONICAL_MIDD.eye_kernel_size)
            self.assertEqual(data["metric"]["annulus_r_max_fraction"], [.14, .98])
            self.assertEqual(len(data["model_ids"]), 50)
