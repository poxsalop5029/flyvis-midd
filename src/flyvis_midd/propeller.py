"""Reusable propeller stimulus generation in image coordinates."""

from __future__ import annotations

import numpy as np

PROPELLER_CONDITIONS = {
    "CW_slow": {"clockwise": True, "speed_hz": 0.5},
    "CW_mid": {"clockwise": True, "speed_hz": 1.0},
    "CW_fast": {"clockwise": True, "speed_hz": 2.0},
    "CCW_slow": {"clockwise": False, "speed_hz": 0.5},
    "CCW_mid": {"clockwise": False, "speed_hz": 1.0},
    "CCW_fast": {"clockwise": False, "speed_hz": 2.0},
    "Static": {"static": True, "speed_hz": 0.0},
}


def make_propeller_frames(
    n_frames=50, fps=50, size=128, n_blades=3, speed_hz=1.0,
    clockwise=True, static=False, hub_radius=5, blade_width=0.20,
):
    """Return a binary propeller movie with shape ``(frame, y, x)``."""
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2
    radius = np.hypot(xx - center, yy - center)
    angle = np.arctan2(yy - center, xx - center)
    direction = 1.0 if clockwise else -1.0
    frames = []
    for frame in range(n_frames):
        phase = 0.0 if static else direction * 2 * np.pi * speed_hz * frame / fps
        wrapped = np.angle(np.exp(1j * n_blades * (angle - phase))) / n_blades
        blades = (
            (np.abs(wrapped) <= blade_width)
            & (radius >= hub_radius)
            & (radius <= size * 0.46)
        )
        frames.append((blades | (radius < hub_radius)).astype(np.float32))
    return np.stack(frames)


def make_propeller_stimulus_set(conditions=None, **frame_kwargs):
    """Generate an ordered condition stack from a condition mapping."""
    conditions = PROPELLER_CONDITIONS if conditions is None else conditions
    return np.stack([
        make_propeller_frames(**frame_kwargs, **parameters)
        for parameters in conditions.values()
    ])


def estimate_propeller_phase(frame, n_blades=3):
    """Estimate blade phase; increasing phase means clockwise in an image."""
    frame = np.asarray(frame, dtype=float)
    yy, xx = np.mgrid[:frame.shape[0], :frame.shape[1]]
    center_x = (frame.shape[1] - 1) / 2
    center_y = (frame.shape[0] - 1) / 2
    angle = np.arctan2(yy - center_y, xx - center_x)
    moment = np.sum(frame * np.exp(1j * n_blades * angle))
    return float(np.angle(moment) / n_blades)
