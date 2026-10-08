"""Regression checks for the deterministic canonical Figure 6 reconstruction."""
import unittest
from pathlib import Path
import pandas as pd
from flyvis_midd.figure_analysis import input_manifest, load_tables, fig6_panel_data

ROOT = Path(__file__).resolve().parents[1]

class Figure6ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = input_manifest('6', ROOT)
        cls.tables = load_tables(manifest, ROOT)
        cls.panels = fig6_panel_data(cls.tables)

    def test_canonical_panel_units_and_saved_regressions(self):
        self.assertEqual(len(self.panels['panel_b_intervention']), 20)
        self.assertEqual(len(self.panels['panel_c_physical']), 40)
        self.assertEqual(len(self.panels['panel_d_pairing']), 10)
        self.assertTrue(self.panels['panel_d_pairing'].n_mismatched_decoders.eq(9).all())
        self.assertEqual(len(self.panels['panel_e_direction']), 100)
        checks = self.panels['fig6_QA'].set_index('check').passed
        self.assertTrue(checks.all())
        for check in ['intervention derived-table agreement',
                      'matched/mismatch derived-table agreement',
                      'direction-summary derived-table agreement']:
            self.assertTrue(bool(checks[check]), check)

    def test_renderer_consumes_validated_data_without_csv_reads(self):
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        from flyvis_midd import figure_rendering
        with patch.object(figure_rendering, 'csv', side_effect=AssertionError('unexpected CSV read')):
            fig = figure_rendering.fig6(self.panels)
        try:
            self.assertEqual(len(fig.axes), 5)
            self.assertEqual(len(fig.axes[-1].images[0].get_array()), 10)
        finally:
            plt.close(fig)

    def test_duplicate_intervention_keys_stop(self):
        tables = dict(self.tables)
        source_key = next(k for k in tables if k.endswith('replacement_metrics.csv'))
        tables[source_key] = pd.concat([tables[source_key], tables[source_key].iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            fig6_panel_data(tables)

    def test_incomplete_direction_matrix_stops(self):
        tables = dict(self.tables)
        source_key = next(k for k in tables if k.endswith('crossswap_direction_accuracy.csv'))
        tables[source_key] = tables[source_key].iloc[:-1].copy()
        with self.assertRaises(AssertionError):
            fig6_panel_data(tables)

if __name__ == '__main__':
    unittest.main()
