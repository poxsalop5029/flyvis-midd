"""Scientific contracts for stronger comparators, stable denominators and radial ROI."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from flyvis_midd.round2_analysis import composition_comparators, robust_intervention_metrics, receptor_sampling_geometry
ROOT=Path(__file__).resolve().parents[1]

class RemainingRound2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=pd.read_csv(ROOT/'data/precomputed/fig03/fig3_3MIDD_components_per_model.csv')
        cls.p=pd.read_csv(ROOT/'data/precomputed/fig03/fig3_canonical_pair_predictions.csv')
        cls.raw=pd.read_csv(ROOT/'data/precomputed/fig06/replacement_metrics.csv')

    def test_original_cv_replay_and_complete_donor_support(self):
        metrics,donors=composition_comparators(self.a,self.p)
        ref=pd.read_csv(ROOT/'data/precomputed/fig03/composition_cv_metrics.csv').query("split=='LOLO'").sort_values('model_id')
        own=metrics.query("predictor=='P0'").sort_values('model_id')
        np.testing.assert_allclose(own[['cv_r2','cv_rmse','sign_accuracy']],ref[['cv_r2','cv_rmse','sign_accuracy']],atol=1e-12)
        self.assertEqual(len(donors),90)
        self.assertTrue((donors.model_id!=donors.donor_id).all())
        self.assertTrue((donors.groupby('model_id').donor_id.nunique()==9).all())

    def test_target_excluded_from_ensemble_predictor(self):
        first,_=composition_comparators(self.a,self.p);changed=self.a.copy();target=sorted(changed.model_id.unique())[0]
        changed.loc[changed.model_id==target,'a3']+=np.linspace(-1,2,7)**2
        second,_=composition_comparators(changed,self.p)
        for name in ['P1','P2']:
            a=first[(first.model_id==target)&(first.predictor==name)].iloc[0]
            b=second[(second.model_id==target)&(second.predictor==name)].iloc[0]
            np.testing.assert_allclose([a.slope,a.intercept,a.cv_r2,a.cv_rmse],[b.slope,b.intercept,b.cv_r2,b.cv_rmse],atol=1e-12)

    def test_primary_intervention_identity_no_weak_A_exclusion(self):
        d=robust_intervention_metrics(self.raw)
        np.testing.assert_allclose(d.normalized_change*d.G_noop,d.intervention_value-d.baseline,atol=1e-14)
        self.assertTrue((d.groupby(['condition','intervention']).model_id.nunique()==10).all())
        self.assertTrue(np.isfinite(d.normalized_change).all())
        # Same physical paired magnitude, not intervention magnitude.
        q=d[(d.model_id==4)&(d.condition=='MIDD')]
        self.assertEqual(q.G_noop.nunique(),1)

    def test_invalid_denominator_rejected(self):
        bad=self.raw.copy();bad.loc[(bad.intervention=='noop')&bad.condition.isin(['CW','CCW']),'mean_flow_magnitude']=0
        with self.assertRaises(ValueError):robust_intervention_metrics(bad)

    def test_ROI_from_actual_coordinates(self):
        geo=pd.read_csv(ROOT/'data/precomputed/fig04/receptor_geometry.csv')
        actual=receptor_sampling_geometry(geo[['y','x']].to_numpy())
        np.testing.assert_allclose(actual.nearest_neighbor_spacing,geo.nearest_neighbor_spacing)
        self.assertTrue(np.array_equal(actual.outer_ROI,geo.outer_ROI))
        self.assertEqual(actual.outer_ROI.sum(),532)
        self.assertTrue((actual.loc[actual.outer_ROI,'period_over_spacing']>=2).all())
        self.assertTrue((~actual.outer_ROI|actual.full_annulus).all())

    def test_official_performance_checkpoint_identity_and_support(self):
        out=ROOT/'data/precomputed/fig05';d=pd.read_csv(out/'official50_sintel_epe.csv');m=pd.read_csv(out/'official50_model_manifest.csv');q=pd.read_csv(out/'official50_canonical_replay_QA.csv')
        self.assertEqual(set(d.model_id),set(m.model_id));self.assertEqual(d.model_id.nunique(),50)
        self.assertTrue((d.n_samples==15).all());self.assertTrue(np.isfinite(d.fixed_eval_epe).all())
        self.assertEqual(len(q),100);self.assertTrue((q.absolute_error<1e-6).all())
        for p in m.checkpoint_sha256:self.assertEqual(len(p),64)

    def test_crossswap_separate_physical_directions(self):
        d=pd.read_csv(ROOT/'data/precomputed/fig06/crossswap_direction_accuracy.csv')
        self.assertEqual(len(d),100);self.assertTrue(np.array_equal(d.both_correct,(d.right_u>0)&(d.left_u<0)))
        self.assertTrue(np.array_equal(d.Q_correct,d.Q>0));self.assertEqual(d[d.matched].both_correct.sum(),10)
        self.assertEqual(d[~d.matched].both_correct.sum(),9)

class PhaseAndManuscriptTests(unittest.TestCase):
    def test_phase_reversal_estimands_are_distinct(self):
        p=pd.read_csv(ROOT/'data/precomputed/fig04/official50_phase_pairs.csv')
        m=pd.read_csv(ROOT/'data/precomputed/fig04/official50_phase_model_metrics.csv')
        self.assertEqual(len(p),600)
        self.assertEqual(p.model_id.nunique(),50)
        fractions=p.groupby('model_id').actual_reversal.mean()
        means=p.groupby('model_id')[['T_original','T_mirror']].mean()
        np.testing.assert_allclose(fractions,m.sort_values('model_id').phasewise_reversal_fraction)
        self.assertEqual(((means.T_original*means.T_mirror)<0).sum(),23)
        self.assertAlmostEqual(fractions.mean(),.435)
        self.assertNotEqual(float(fractions.mean()),23/50)

    def test_same_operator_shuffle_support_and_fixed_controls(self):
        d=pd.read_csv(ROOT/'data/precomputed/fig02/official50_shuffle_draw_metrics.csv')
        self.assertEqual(len(d),1500)
        self.assertTrue((d.groupby(['model_id','group']).draw.nunique()==10).all())
        learning=pd.read_csv(ROOT/'data/precomputed/fig05/learning_fixed_control_pairs.csv')
        self.assertEqual(set(learning.optimizer_step),{0,10000,50000,100000})
        self.assertTrue(learning[learning.condition=='uniform singleton'].A.isna().all())

    def test_supplementary_tables_are_embedded(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pdf_builder',ROOT/'scripts/build_manuscript_pdf.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        text=module.supplementary_tables_markdown(ROOT)
        for key in range(1,5):self.assertIn('Table S'+str(key)+'.',text)
        self.assertIn('official50_performance_primary_relations',text)
        self.assertIn('comparator_summary',text)

if __name__=='__main__':unittest.main()
