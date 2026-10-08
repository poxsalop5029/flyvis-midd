"""Small synthetic stimulus/flow tests; no FlyVis models or downloads."""
import unittest
import numpy as np
import torch
from flyvis_midd.stimuli import (
    build_3midd_sequences, build_4midd_sequences, directed_pair_counts, render_sector_image,
)
from flyvis_midd.flow import canonical_flow_layout
from flyvis_midd.rotation import image_rotation_flow_metrics
from flyvis_midd.propeller import make_propeller_frames, estimate_propeller_phase


class StimulusAndFlowTests(unittest.TestCase):
    def assert_balanced(self, sequence):
        counts = directed_pair_counts(sequence)
        for (left, right), count in counts.items():
            self.assertEqual(count, counts[(right, left)])

    def test_3midd_sequences(self):
        variants = build_3midd_sequences(48)
        self.assertEqual(set(variants), {"original", "mirror", "negative"})
        for seq in variants.values():
            self.assertEqual(seq.dtype, np.uint8)
            self.assertEqual(seq.shape, (72,))
        np.testing.assert_array_equal(variants["mirror"], variants["original"][::-1])
        self.assertEqual(set(variants["original"]), {0, 48, 255})
        self.assert_balanced(variants["negative"])

    def test_4midd_sequences(self):
        variants = build_4midd_sequences(85, 170)
        self.assertEqual(set(variants), {"original", "mirror", "negative"})
        for seq in variants.values():
            self.assertEqual(seq.dtype, np.uint8)
            self.assertEqual(seq.shape, (96,))
        # A cyclic reversal keeps the first zero-valued sector at the same phase.
        np.testing.assert_array_equal(variants["mirror"], np.roll(variants["original"][::-1], 1))
        self.assertEqual(set(variants["original"]), {0, 85, 170, 255})
        self.assert_balanced(variants["negative"])

    def test_sector_ring_geometry_and_luminance(self):
        image, mask = render_sector_image([0, 48, 255], image_size=64,
                                         inner_diameter=16, outer_diameter=48)
        self.assertEqual(image.shape, (64, 64))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(mask.dtype, np.bool_)
        yy, xx = np.indices(image.shape)
        radius = np.hypot(yy - 31.5, xx - 31.5)
        np.testing.assert_array_equal(mask, (radius >= 8) & (radius <= 24))
        self.assertEqual(set(image[mask]), {0, 48, 255})
        self.assertTrue(np.all(image[~mask] == 255))

    def test_flow_layout_equivalence(self):
        flow = torch.arange(2*3*2*5).reshape(2, 3, 2, 5)
        torch.testing.assert_close(canonical_flow_layout(flow), flow)
        torch.testing.assert_close(canonical_flow_layout(flow.permute(0, 1, 3, 2)), flow)

    def test_invalid_flow_layout(self):
        for shape in [(3, 2, 5), (1, 3, 4, 5), (1, 2, 3, 4, 5)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                canonical_flow_layout(torch.zeros(shape))

    def test_rotation_direction_and_conversion(self):
        # Positions on the horizontal diameter isolate the decoder v convention.
        centers = np.array([[0., -1.], [0., 1.]])
        raw_cw = np.array([[0., 1.], [0., -1.]])
        kw = {"inner_radius_fraction": 0., "outer_radius_fraction": 1.}
        cw = image_rotation_flow_metrics(centers, raw_cw, **kw)
        ccw = image_rotation_flow_metrics(centers, -raw_cw, **kw)
        self.assertAlmostEqual(cw["mean_signed_tangent"], 1.)
        self.assertAlmostEqual(ccw["mean_signed_tangent"], -1.)
        preconverted = raw_cw * [1., -1.]
        once = image_rotation_flow_metrics(centers, preconverted, flip_decoder_y=False, **kw)
        twice = image_rotation_flow_metrics(centers, preconverted, **kw)
        self.assertEqual(once["mean_signed_tangent"], cw["mean_signed_tangent"])
        self.assertEqual(twice["mean_signed_tangent"], ccw["mean_signed_tangent"])

    def test_propeller_phase_direction(self):
        slopes = []
        for clockwise in (True, False):
            movie = make_propeller_frames(n_frames=30, size=64, fps=50,
                                          speed_hz=.5, clockwise=clockwise)
            phase = np.array([estimate_propeller_phase(frame) for frame in movie])
            phase = np.unwrap(3*phase)/3
            slopes.append(np.polyfit(np.arange(len(phase))/50, phase, 1)[0])
        self.assertGreater(slopes[0], 0.)
        self.assertLess(slopes[1], 0.)
        self.assertAlmostEqual(slopes[0], -slopes[1], places=6)
