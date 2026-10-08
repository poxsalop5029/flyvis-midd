"""Lightweight contract tests: no FlyVis downloads or ensemble inference."""
import unittest
from dataclasses import replace
from pathlib import Path
import csv
import numpy as np
import torch

from flyvis_midd.evaluation import CANONICAL_MIDD, analyze_midd_flow, aggregate_midd_metrics
from flyvis_midd.flow import run_flow_inference

ROOT = Path(__file__).resolve().parents[1]


class MIDDContractTests(unittest.TestCase):
    def test_reconstruction_reversal_and_negative_independence(self):
        o = np.array([-2., 0., 3., -4., 1.])
        m = np.array([3., 0., 2., -1., -1.])
        result = aggregate_midd_metrics({"original": o, "mirror": m, "negative": o*1e6})
        np.testing.assert_allclose(result["S"]+result["A"], o)
        np.testing.assert_allclose(result["S"]-result["A"], m)
        np.testing.assert_array_equal(result["actual_reversal"], result["M"]>0)
        without_negative = aggregate_midd_metrics({"original": o, "mirror": m})
        for key in ("A", "S", "M", "actual_reversal"):
            np.testing.assert_array_equal(result[key], without_negative[key])

    def test_frame_slice_flip_once_and_input_not_mutated(self):
        centers = np.array([[0.,1.],[1.,0.],[0.,-1.],[-1.,0.],
                            [0.,2.],[2.,0.],[0.,-2.],[-2.,0.]])
        radius = np.linalg.norm(centers, axis=1)
        # Raw decoder vectors yield the CW tangent only after one v flip.
        raw = np.stack([-centers[:,0]/radius, -centers[:,1]/radius], axis=0)
        flow = np.broadcast_to(raw, (1,60,2,8)).copy()
        flow[:,:10] *= -100
        flow[:,50:] *= -100
        original = flow.copy()
        metric = analyze_midd_flow(flow, centers, config=replace(CANONICAL_MIDD,n_frames=60))
        self.assertAlmostEqual(metric["mean_signed_tangent"], 1.)
        self.assertEqual(metric["annulus_receptors"], 4)
        np.testing.assert_array_equal(flow, original)
        # Alternate supported output layout must produce the same measurement.
        other = analyze_midd_flow(flow.transpose(0,1,3,2), centers,
                                  config=replace(CANONICAL_MIDD,n_frames=60))
        self.assertEqual(other["mean_signed_tangent"], metric["mean_signed_tangent"])
        with self.assertRaises(ValueError):
            analyze_midd_flow(flow,centers,flow_coordinates="image")
        with self.assertRaises(ValueError):
            analyze_midd_flow(flow,centers,config=replace(CANONICAL_MIDD,flip_decoder_y=False))
        with self.assertRaises(ValueError):
            analyze_midd_flow(flow[:, :49], centers)

    def test_explicit_fade_and_recursive_decoder_eval(self):
        class Network(torch.nn.Module):
            def fade_in_state(self, fade, dt, frame):
                self.fade_args=(fade,dt,frame.clone());return object()
            def simulate(self, lum, dt, initial_state):
                self.sim_args=(dt,initial_state)
                return lum.repeat(1,1,2,1)
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.dropout=torch.nn.Dropout(.5);self.batchnorm=torch.nn.BatchNorm1d(4)
            def forward(self, response):return self.dropout(response)
        net, dec = Network(), Decoder()
        lum = torch.arange(200,dtype=torch.float64).reshape(1,50,1,4)
        first = run_flow_inference(lum,net,dec)
        second = run_flow_inference(lum,net,dec)
        self.assertFalse(net.training)
        self.assertFalse(dec.dropout.training)
        self.assertFalse(dec.batchnorm.training)
        self.assertEqual(net.fade_args[:2], (1.,.02))
        torch.testing.assert_close(net.fade_args[2],lum.float()[:,0])
        self.assertEqual(tuple(first["flow"].shape),(1,50,2,4))
        torch.testing.assert_close(first["flow"],second["flow"])
        torch.testing.assert_close(first["flow"],lum.float().repeat(1,1,2,1))

    def test_official_50_saved_table_regression(self):
        def read(name):
            with (ROOT/'data/precomputed/fig02'/name).open() as f:
                return list(csv.DictReader(f))
        lum = read('fig2B_luminance_A_per_model.csv')
        four = read('fig2D_4MIDD_A_per_model.csv')
        for rows,count,mean in [([r for r in lum if int(r['C'])==48],21,-.00934561245223449),
                                ([r for r in lum if int(r['C'])==207],18,.00825856),
                                (four,12,.00320904)]:
            self.assertEqual({r['model_id'] for r in rows},
                             {f'flow/0000/{i:03d}' for i in range(50)})
            result=aggregate_midd_metrics({v:np.array([float(r[v]) for r in rows])
                                           for v in ('original','mirror')})
            self.assertEqual(int(result['actual_reversal'].sum()),count)
            self.assertAlmostEqual(result['A'].mean(),mean,places=6)
            np.testing.assert_allclose(result['A'],[float(r['A']) for r in rows],atol=1e-12)
            np.testing.assert_array_equal(result['actual_reversal'],result['M']>0)


if __name__ == '__main__':
    unittest.main()
