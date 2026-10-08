"""Scientific contract checks for artifact-only canonical figure entry points."""
import json
from pathlib import Path
import unittest
import numpy as np
from flyvis_midd.figure_analysis import evaluate_scalar_cv, percentile_model_bootstrap

ROOT = Path(__file__).resolve().parents[1]

class FigureIntegrationTests(unittest.TestCase):
    def test_cv_affine_transfer_and_disjoint_holdout(self):
        x = np.arange(7,dtype=float); y=0.8*x+0.01
        splits=[(np.delete(np.arange(7),i),np.array([i])) for i in range(7)]
        metric=evaluate_scalar_cv(x,y,splits,np.ones(7,dtype=bool))[0]
        np.testing.assert_allclose(metric,[1,0,1],atol=1e-12)
        with self.assertRaises(AssertionError):
            evaluate_scalar_cv(x,y,[(np.array([0,1]),np.array([1,2]))],np.ones(7,dtype=bool))

    def test_bootstrap_reproducible_finite_model_unit_contract(self):
        a=percentile_model_bootstrap([1,2,3],np.random.default_rng(5))
        b=percentile_model_bootstrap([1,2,3],np.random.default_rng(5))
        self.assertEqual(a,b)
        self.assertLessEqual(a[1],a[0]);self.assertGreaterEqual(a[2],a[0])
        with self.assertRaises(ValueError):
            percentile_model_bootstrap([1,np.nan],np.random.default_rng(5))

    def test_sixteen_routes_and_immutable_source_paths(self):
        spec=json.loads((ROOT/'src/flyvis_midd/figure_inputs.json').read_text())
        self.assertEqual(set(spec),set(map(str,range(1,7)))|{'S'+str(i) for i in range(1,11)})
        for key,d in spec.items():
            self.assertTrue((ROOT/d['notebook']).is_file(),key)
            self.assertTrue(all((ROOT/p).is_file() for p in d['input_paths']),key)
            n=json.loads((ROOT/d['notebook']).read_text())
            active='\n'.join(''.join(c['source']) for c in n['cells'][:21])
            self.assertNotIn('runpy.run_path',active);self.assertNotIn('phase2_scripts',active)
            self.assertNotIn('phase3_scripts',active)
            self.assertIn('input_manifest',active)

if __name__=='__main__':unittest.main()
