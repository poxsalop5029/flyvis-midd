# Copied unchanged from fvpg/src/flyvis_illusion/stimuli/midd.py.
"""MIDD sector-image and luminance-sequence generators."""

from __future__ import annotations

from collections import Counter

import numpy as np


def build_3midd_sequences(c: int, n_units: int = 24) -> dict[str, np.ndarray]:
    """Return original, mirror, and balanced-negative 3MIDD sector values."""
    c = int(c)
    return {
        "original": np.tile([0, c, 255], n_units).astype(np.uint8),
        "mirror": np.tile([255, c, 0], n_units).astype(np.uint8),
        "negative": np.tile([0, c, 0, 255, c, 255], n_units // 2).astype(np.uint8),
    }


def build_4midd_sequences(
    c1: int, c2: int, n_units: int = 24
) -> dict[str, np.ndarray]:
    """Return original, mirror, and balanced-negative 4MIDD sector values."""
    c1, c2 = int(c1), int(c2)
    return {
        "original": np.tile([0, c1, 255, c2], n_units).astype(np.uint8),
        "mirror": np.tile([0, c2, 255, c1], n_units).astype(np.uint8),
        "negative": np.tile([0, c1, 255, c2, 0, c2, 255, c1], n_units // 2).astype(np.uint8),
    }


def directed_pair_counts(sequence) -> Counter:
    """Count directed transitions, including the circular last-to-first pair."""
    values = [int(value) for value in np.asarray(sequence).tolist()]
    return Counter(zip(values, values[1:] + values[:1]))


def render_sector_image(
    sequence,
    *,
    image_size: int = 382,
    inner_diameter: int = 55,
    outer_diameter: int = 382,
    phase_deg: float = 0.0,
):
    """Render sector luminances into a uint8 ring image and return its mask."""
    sequence = np.asarray(sequence, dtype=np.uint8)
    if sequence.ndim != 1 or len(sequence) == 0:
        raise ValueError("sequence must be a non-empty 1-D array")
    yy, xx = np.indices((image_size, image_size), dtype=float)
    center = (image_size - 1) / 2
    dx, dy = xx - center, yy - center
    radius = np.hypot(dx, dy)
    ring_mask = (radius >= inner_diameter / 2) & (radius <= outer_diameter / 2)
    angle = np.mod(np.arctan2(dy, dx) - np.deg2rad(phase_deg), 2 * np.pi)
    sector_index = np.floor(angle / (2 * np.pi / len(sequence))).astype(int)
    image = np.full((image_size, image_size), 255, dtype=np.uint8)
    image[ring_mask] = sequence[sector_index[ring_mask]]
    return image, ring_mask



def load_midd_sequences() -> np.ndarray:
    """Return the six static Fig. 1B MIDD movies for FlyVis.

    The output has shape ``(6, 50, 382, 382)`` in condition, frame, y, x
    order, with float32 luminance normalized to [0, 1]. Conditions are
    3MIDD 048 Original/Mirror, 3MIDD 207 Original/Mirror, then 4MIDD
    085/170 Original/Mirror. Each source image is repeated at 50 fps.
    """
    n_frames = 50
    image_size = 382
    groups = (
        ("3MIDD_048", build_3midd_sequences(48)),
        ("3MIDD_207", build_3midd_sequences(207)),
        ("4MIDD_085_170", build_4midd_sequences(85, 170)),
    )
    source_images = []
    for _, variants in groups:
        for variant in ("original", "mirror"):
            image, _ = render_sector_image(
                variants[variant],
                image_size=image_size,
                inner_diameter=55,
                outer_diameter=382,
                phase_deg=0.0,
            )
            source_images.append(image)
    images = np.stack(source_images).astype(np.float32) / 255.0
    return np.repeat(images[:, None], n_frames, axis=1).astype(np.float32, copy=False)
