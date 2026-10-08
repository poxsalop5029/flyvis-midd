"""Static contract checks for the public D1 acquisition route; no inference."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_d1_source_currents.py"
CANONICAL = ROOT / "data/precomputed/supplementary/biological_constraints/D1/pilot/pilot_target_node_distribution.csv"
EXPECTED_SHA256 = "fbb384f5e7cbfc11fdad7feae0df57c95e00f4d1dd8b9342a40946e9a5d970ca"
EXPECTED_COLUMNS = (
    "target_type", "source_type", "target_node_id", "mean_current_frames_10_49",
    "model_id", "phase_index", "phase_deg", "condition",
)


class D1AcquisitionContractTests(unittest.TestCase):
    def test_generator_path_config_and_no_stale_acquisition_roots(self):
        source = SCRIPT.read_text()
        self.assertIn('OUTPUT = ROOT / "data/precomputed/supplementary/biological_constraints/D1/pilot/pilot_target_node_distribution.csv"', source)
        self.assertNotRegex(source, r"addon_D1|addon_B1|data/precomputed/addon_|notebooks/addon")
        tree = ast.parse(source)
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            assignments[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        self.assertEqual(assignments["EXPECTED_SHA256"], EXPECTED_SHA256)
        self.assertEqual(tuple(assignments["COLUMNS"]), EXPECTED_COLUMNS)
        self.assertEqual(assignments["EXPECTED_ROWS"], 1_661_184)
        self.assertEqual(assignments["FLYVIS_COMMIT"], "92b3845cc426dd309a1a0e1b3890156c42e14021")
        self.assertTrue(CANONICAL.is_file())

    def test_retained_canonical_csv_hash_and_header(self):
        digest = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_SHA256)
        with CANONICAL.open(encoding="utf-8") as stream:
            self.assertEqual(tuple(stream.readline().rstrip("\n").split(",")), EXPECTED_COLUMNS)


if __name__ == "__main__":
    unittest.main()
