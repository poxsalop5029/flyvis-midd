"""Shared single-BoxEye MIDD measurement contract for Figures 1 and 2."""

from dataclasses import asdict, dataclass
import numpy as np
import torch

from .flow import canonical_flow_layout
from .rotation import image_rotation_flow_metrics


@dataclass(frozen=True)
class MIDDEvaluationConfig:
    """Explicit timing, renderer and metric settings for canonical MIDD flow."""

    dt: float = 1 / 50
    fade_seconds: float = 1.0
    n_frames: int = 50
    eye_extent: int = 15
    eye_kernel_size: int = 13
    analysis_start: int = 10
    analysis_stop: int = 50
    inner_radius_fraction: float = 0.14
    outer_radius_fraction: float = 0.98
    flip_decoder_y: bool = True

    def provenance(self):
        return {
            **asdict(self),
            "fade_source": "stimulus frame 0",
            "fade_frames_concatenated": False,
            "flow_coordinates": "raw decoder (u,v)",
            "image_conversion": "(u,v) -> (u,-v), once inside metric",
            "rotation_sign": "CW positive",
            "reversal": "T_original * T_mirror < 0; negative excluded",
        }


CANONICAL_MIDD = MIDDEvaluationConfig()


def analyze_midd_flow(flow, centers_yx, *, config=CANONICAL_MIDD,
                      flow_coordinates="decoder"):
    """Measure one static MIDD movie, accepting raw decoder flow only.

    Flow is canonicalized to (samples, frames, 2, receptors). One spatial
    sample is required; separate stimulus conditions must be analyzed
    separately. Frames 10:50 and the 0.14–0.98 r_max annulus are defaults.
    The caller must not flip v: image_rotation_flow_metrics performs the
    sole coordinate conversion. Already-converted input declared as image
    coordinates is rejected. Arrays alone cannot reveal an undeclared flip.
    The returned annulus_mask is available for display/QA, not scalar tables.
    """
    if flow_coordinates != "decoder":
        raise ValueError("Expected raw decoder (u,v); do not pre-flip decoder v")
    if not config.flip_decoder_y:
        raise ValueError("Canonical CW-positive MIDD requires one decoder-v flip")
    if not (0 <= config.analysis_start < config.analysis_stop <= config.n_frames):
        raise ValueError("Analysis window must lie within recorded stimulus frames")
    if not (0 <= config.inner_radius_fraction < config.outer_radius_fraction <= 1):
        raise ValueError("Invalid annulus radius fractions")
    value = canonical_flow_layout(torch.as_tensor(flow).detach().cpu()).numpy()
    centers = np.asarray(centers_yx, float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError("Expected finite receptor coordinates (receptors,2)")
    expected = (1, config.n_frames, 2, len(centers))
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(f"Expected finite flow {expected}, got {value.shape}")
    selected = value[0, config.analysis_start:config.analysis_stop].transpose(0, 2, 1)
    return image_rotation_flow_metrics(
        centers, selected,
        inner_radius_fraction=config.inner_radius_fraction,
        outer_radius_fraction=config.outer_radius_fraction,
        flip_decoder_y=config.flip_decoder_y,
    )


def aggregate_midd_metrics(metrics_by_variant):
    """Return vectorized A/S/M and raw reversal from Original/Mirror T.

    Each value is a scalar/array or a metric dictionary containing
    mean_signed_tangent. Negative is optional and reported independently;
    it never contributes to A, S, M or actual_reversal. Arrays must share
    a shape. For scalar input, scalar Python values are returned.
    """
    def tangent(variant):
        value = metrics_by_variant[variant]
        if isinstance(value, dict):
            value = value["mean_signed_tangent"]
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {variant} tangent")
        return array

    original, mirror = tangent("original"), tangent("mirror")
    if original.shape != mirror.shape:
        raise ValueError("Original and Mirror tangents must have the same shape")
    a = (original - mirror) / 2
    s = (original + mirror) / 2
    result = {"T_original": original, "T_mirror": mirror, "A": a,
              "abs_A": np.abs(a), "S": s, "abs_S": np.abs(s),
              "M": np.abs(a) - np.abs(s),
              "actual_reversal": original * mirror < 0}
    if "negative" in metrics_by_variant:
        negative = tangent("negative")
        if negative.shape != original.shape:
            raise ValueError("Negative tangent must have the same shape as the pair")
        result["T_negative"] = negative
    return {key: value.item() if value.ndim == 0 else value
            for key, value in result.items()}
