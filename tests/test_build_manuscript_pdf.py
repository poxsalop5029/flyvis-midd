"""Layout-contract tests; no PDF toolchain or analysis dependencies required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/build_manuscript_pdf.py'
spec = importlib.util.spec_from_file_location('manuscript_pdf_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def manuscript():
    return ('# Test title\n\n## Results\n\n' + ''.join(
        f'### Figure {i}. Result {i}\n\nResult paragraph {i}.\n\n' for i in builder.MAIN
    ) + '## Discussion\n\nDiscussion unchanged.\n\n## Figure legends\n\n' + ''.join(
        f'### Figure {i}. Legend {i}\n\nFormal legend {i}.\n\n' for i in builder.MAIN
    ) + '## Supplementary figure legends\n\n' + ''.join(
        f'### Supplementary Figure {i}. Title {i}\n\nFormal legend {i}.\n\n'
        for i in builder.SUPPLEMENTARY
    ) + '## References\n\nExisting reference unchanged.\n')


class BuildLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for image in builder.FIGURES.values():
            path = self.root / image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_figures_follow_results_and_formal_legends_appear_once(self):
        source = manuscript()
        result = builder.prepare_markdown(source, self.root)
        self.assertEqual(result.count('\\includegraphics['), len(builder.FIGURES))
        self.assertNotIn('## Figure legends\n', result)
        for i in builder.MAIN:
            body = result.index(f'Result paragraph {i}.')
            image = result.index(builder.FIGURES[i].as_posix())
            legend = result.index(f'Formal legend {i}.')
            self.assertLess(body, image)
            self.assertLess(image, legend)
            if i != '6':
                self.assertLess(legend, result.index(f'### Figure {int(i)+1}. Result'))
        for i in builder.FIGURES:
            self.assertEqual(result.count(f'Formal legend {i}.'), 1)
        supplementary = result.index('## Supplementary figure legends')
        for i in builder.SUPPLEMENTARY:
            image = result.index(builder.FIGURES[i].as_posix())
            self.assertLess(supplementary, image)
            self.assertLess(image, result.index(f'Formal legend {i}.'))
            supplementary = result.index(f'Formal legend {i}.')
        self.assertIn('Discussion unchanged.', result)
        self.assertIn('Existing reference unchanged.', result)

    def test_seven_scientific_sections_keep_temporal_section_and_single_figures(self):
        source = manuscript()
        for i in builder.MAIN:
            source = source.replace(f'### Figure {i}. Result {i}', f'### Scientific result {i}')
        source = source.replace('### Scientific result 6', '### Prolonged static stimulation\n\nTemporal paragraph.\n\n### Scientific result 6')
        result = builder.prepare_markdown(source, self.root)
        self.assertEqual(result.count('\\includegraphics['), len(builder.FIGURES))
        self.assertLess(result.index('Temporal paragraph.'), result.index(builder.FIGURES['6'].as_posix()))
        self.assertIn('Temporal paragraph.\n\n### Scientific result 6', result)
        for key in builder.FIGURES:
            self.assertEqual(result.count(f'Formal legend {key}.'), 1)
        bad = source.replace('### Prolonged static stimulation\n\nTemporal paragraph.\n\n', '')
        with self.assertRaisesRegex(ValueError, 'seven nonempty'):
            builder.prepare_markdown(bad, self.root)

    def test_supplementary_tables_include_saved_biological_constraints(self):
        import csv
        figures_tables=self.root/'figures/supplementary/tables'
        figures_tables.mkdir(parents=True)
        fixtures={
            'TableS1_source_contracts.csv':['analysis','cohort_units','renderer_sign','support','boundary'],
            'TableS2_model_unit_estimates.csv':['source_file'],
            'TableS3_crossswap_SS.csv':['group','window','metric','effect','fraction','ci_low','ci_high'],
            'TableS4_reproduction_inventory.csv':['path','sha256','route'],
        }
        for name,headers in fixtures.items():
            with (figures_tables/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=headers); writer.writeheader()
                writer.writerow({key:'fixture' for key in headers})
        biological=self.root/'data/precomputed/supplementary/biological_constraints/tables'
        biological.mkdir(parents=True)
        for name,headers,row in [
            ('TableS5A_t4c_phenotype.csv',['Endpoint','Cluster 0 (n=31)','Cluster 1 (n=7)','Cluster 2 (n=12)','Effect / p (raw / Holm)'],{'Endpoint':'phase-mean signed A'}),
            ('TableS5B_activity_and_current.csv',['Analysis / endpoint','Estimate/statistic','95% CI','Raw p','Adjusted p','Status'],{'Analysis / endpoint':'B1 P1 mean A_activity'})]:
            with (biological/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=headers); writer.writeheader(); writer.writerow(row)
        result=builder.supplementary_tables_markdown(self.root)
        self.assertIn('Table S5A. T4c phenotype associations with MIDD response',result)
        self.assertIn('Table S5B. Prespecified T4/T5 activity and ON/OFF source-current summaries',result)
        self.assertIn('phase-mean signed A',result)
        self.assertIn('B1 P1 mean A_activity',result)

    def test_missing_figure_fails_explicitly(self):
        (self.root / builder.FIGURES['3']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing figure: Figure 3'):
            builder.prepare_markdown(manuscript(), self.root)

    def test_missing_and_duplicate_headings_fail(self):
        source = manuscript()
        missing = source.replace('### Supplementary Figure S4.', '### Unexpected S4.')
        with self.assertRaises(ValueError):
            builder.prepare_markdown(missing, self.root)
        duplicate = source.replace('## Discussion', '### Figure 1. Duplicate\n\nText\n\n## Discussion')
        with self.assertRaisesRegex(ValueError, 'Duplicate heading'):
            builder.prepare_markdown(duplicate, self.root)

    def test_missing_dependencies_preserve_sources_and_prepare_fallback(self):
        source_path = self.root / 'manuscript/manuscript.md'
        source_path.parent.mkdir(parents=True)
        source_path.write_text(manuscript())
        before = source_path.read_bytes()
        with patch.object(builder, 'ROOT', self.root), patch.object(builder, 'find_program', return_value=None):
            self.assertEqual(builder.main([]), 1)
        self.assertEqual(source_path.read_bytes(), before)
        self.assertTrue((self.root / 'manuscript/_build/manuscript_with_figures.md').exists())
        self.assertFalse((self.root / 'manuscript/flyvis_midd_manuscript.pdf').exists())


if __name__ == '__main__':
    unittest.main()
