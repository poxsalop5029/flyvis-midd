"""Flow inference, coordinate conversion, and vector-field helpers."""

from __future__ import annotations

import torch


def canonical_flow_layout(flow):
    """Normalize decoder output to ``(samples, frames, 2, hexels)``."""
    if flow.ndim != 4:
        raise ValueError(f"Expected 4D flow, got {tuple(flow.shape)}")
    if flow.shape[2] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Cannot identify flow component axis: {tuple(flow.shape)}")


def run_flow_inference(
    luminance,
    network,
    decoder,
    dt=1 / 50,
    fade=1.0,
    device=torch.device("cpu"),
):
    """Canonical explicit fade-in -> simulate -> recursive-eval flow decoding.

    Fade-in uses stimulus frame 0 and its frames are not concatenated to
    the recorded movie. Both modules use .eval(), including decoder child
    BatchNorm/Dropout modules. Return raw decoder (u,v), with no sign flip;
    analyze_midd_flow applies image-coordinate conversion exactly once.
    """
    luminance = luminance.float().to(device)
    network = network.to(device).eval()
    decoder = decoder.to(device).eval()
    with torch.no_grad():
        initial_state = network.fade_in_state(fade, dt, luminance[:, 0])
        responses = network.simulate(luminance, dt, initial_state=initial_state)
        decoded_flow = decoder(responses)
    return {
        "luminance": luminance.detach().cpu(),
        "responses": responses.detach().cpu(),
        "flow": canonical_flow_layout(decoded_flow).detach().cpu(),
    }


def extract_cell_type(responses, network, cell_type):
    """Extract one connectome cell type from simulated network responses."""
    from flyvis.utils.activity_utils import LayerActivity

    activity = LayerActivity(responses, network.connectome, keepref=True)
    return activity[cell_type].detach().cpu()


def flow_uv_to_polar(flow, flip_y_for_display=True, eps=1e-8):
    """Convert ``u, v`` flow into radius, angle in degrees, and a valid mask."""
    flow = canonical_flow_layout(flow)
    u = flow[:, :, 0].detach().cpu()
    v = flow[:, :, 1].detach().cpu()
    if flip_y_for_display:
        v = -v
    radius = torch.hypot(u, v)
    theta_deg = torch.rad2deg(torch.atan2(v, u))
    valid_theta = radius > eps
    return radius, theta_deg, valid_theta


def map_normal_positions_to_layout(receptors, layout):
    """Map normal BoxEye centers to a resized layout image coordinate system."""
    local = receptors.receptor_centers.detach().cpu().float()
    normal_hw = receptors.min_frame_size.detach().clone().cpu().float()
    source_yx = local + torch.floor(normal_hw / 2)
    target_hw = torch.as_tensor(layout["image"].shape).float()
    scale_yx = target_hw / normal_hw
    target_yx = (source_yx + 0.5) * scale_yx - 0.5 + layout["image_origin"]
    return target_yx, scale_yx


def merge_duplicate_vectors(yx, vectors):
    """Average duplicate coordinates in a ``(frames, positions, 2)`` field."""
    unique_yx, inverse, counts = torch.unique(
        yx, dim=0, return_inverse=True, return_counts=True
    )
    merged = torch.zeros(
        vectors.shape[0], len(unique_yx), 2,
        dtype=vectors.dtype, device=vectors.device,
    )
    merged.index_add_(1, inverse.to(vectors.device), vectors)
    counts = counts.to(vectors.device)
    return unique_yx, merged / counts[None, :, None], counts


def merge_spatial_samples(yx, *samples):
    """Average sample arrays that share duplicate ``(y, x)`` coordinates.

    Every sample array must use its first axis for position; trailing dimensions
    are preserved. This supports scalars, vectors, and other per-position data.
    """
    yx = torch.as_tensor(yx).detach().cpu().reshape(-1, 2)
    unique_yx, inverse, counts = torch.unique(
        yx, dim=0, return_inverse=True, return_counts=True
    )
    merged = []
    for sample in samples:
        sample = torch.as_tensor(sample).detach().cpu()
        if sample.ndim == 0 or sample.shape[0] != len(yx):
            raise ValueError(
                "each sample must have one first-axis value per coordinate, "
                f"got {tuple(sample.shape)} for {len(yx)} coordinates"
            )
        output = torch.zeros(
            (len(unique_yx), *sample.shape[1:]), dtype=sample.dtype
        )
        output.index_add_(0, inverse, sample)
        divisor = counts.reshape(-1, *(1 for _ in sample.shape[1:]))
        merged.append(output / divisor)
    return unique_yx, *merged, counts


def merge_spatial_axis(yx, values, position_axis=-2):
    """Average duplicate coordinates along one axis of an arbitrary array."""
    values = torch.as_tensor(values).detach().cpu()
    position_axis %= values.ndim
    coordinates = torch.as_tensor(yx).detach().cpu().reshape(-1, 2)
    if values.shape[position_axis] != len(coordinates):
        raise ValueError(
            f"position axis has {values.shape[position_axis]} values for "
            f"{len(coordinates)} coordinates"
        )
    position_first = values.movedim(position_axis, 0)
    unique, merged, counts = merge_spatial_samples(coordinates, position_first)
    return unique, merged.movedim(0, position_axis), counts


def merge_layout_field(centers_yx, luminance, flow):
    """Merge duplicate coordinates in a multi-field rendering and flow pair.

    Parameters
    ----------
    centers_yx
        Layout coordinates with shape ``(fields, hexels, 2)`` or ``(N, 2)``.
    luminance
        Rendered luminance with shape ``(fields, frames, 1, hexels)``.
    flow
        Decoded flow with shape ``(fields, frames, 2, hexels)``.

    Returns
    -------
    tuple
        Unique coordinates ``(N, 2)``, luminance ``(frames, N)``, and flow
        ``(frames, N, 2)``. Duplicate layout-boundary values are averaged.
    """
    yx = torch.as_tensor(centers_yx).detach().cpu().float().reshape(-1, 2)
    luminance = torch.as_tensor(luminance).detach().cpu().float()
    flow = canonical_flow_layout(torch.as_tensor(flow)).detach().cpu().float()
    if luminance.ndim != 4 or luminance.shape[2] != 1:
        raise ValueError(
            "luminance must have shape (fields, frames, 1, hexels), "
            f"got {tuple(luminance.shape)}"
        )
    if flow.shape[:2] != luminance.shape[:2] or flow.shape[3] != luminance.shape[3]:
        raise ValueError(
            "flow and luminance field/frame/hexel dimensions must match: "
            f"{tuple(flow.shape)} vs {tuple(luminance.shape)}"
        )
    frames = luminance.shape[1]
    lum = luminance.permute(1, 0, 3, 2).reshape(frames, -1, 1)
    vectors = flow.permute(1, 0, 3, 2).reshape(frames, -1, 2)
    if len(yx) != lum.shape[1]:
        raise ValueError(
            f"coordinate count {len(yx)} does not match field values {lum.shape[1]}"
        )
    unique_yx, inverse, counts = torch.unique(
        yx, dim=0, return_inverse=True, return_counts=True
    )
    merged_luminance = torch.zeros(frames, len(unique_yx), 1, dtype=lum.dtype)
    merged_flow = torch.zeros(frames, len(unique_yx), 2, dtype=vectors.dtype)
    merged_luminance.index_add_(1, inverse, lum)
    merged_flow.index_add_(1, inverse, vectors)
    return (
        unique_yx,
        (merged_luminance / counts[None, :, None])[:, :, 0],
        merged_flow / counts[None, :, None],
    )


def native_boxeye_field(receptors, luminance, flow):
    """Return native BoxEye coordinates, luminance, and canonical flow."""
    yx = receptors.receptor_centers.detach().cpu().float()
    luminance = torch.as_tensor(luminance).detach().cpu().float()
    flow = torch.as_tensor(flow).detach().cpu().float()
    if luminance.ndim != 3 or luminance.shape[1] != 1:
        raise ValueError(
            "luminance must have shape (frames, 1, hexels), "
            f"got {tuple(luminance.shape)}"
        )
    if flow.ndim != 3:
        raise ValueError(f"flow must be 3D, got {tuple(flow.shape)}")
    if flow.shape[1] == 2:
        vectors = flow.permute(0, 2, 1)
    elif flow.shape[-1] == 2:
        vectors = flow
    else:
        raise ValueError(f"Cannot identify flow component axis: {tuple(flow.shape)}")
    values = luminance[:, 0]
    if values.shape[:2] != vectors.shape[:2] or len(yx) != values.shape[1]:
        raise ValueError(
            "receptor, luminance, and flow dimensions do not match: "
            f"{len(yx)}, {tuple(values.shape)}, {tuple(vectors.shape)}"
        )
    return yx, values, vectors


def flow_roi_metrics(
    centers_yx,
    flow,
    *,
    baseline_flow,
    frame_slice=slice(None),
    direction_deg,
    radius_fraction=0.65,
    flip_y_for_display=True,
):
    """Summarize baseline-subtracted flow inside a central circular ROI."""
    if not 0 < radius_fraction <= 1:
        raise ValueError(
            f"radius_fraction must be in (0, 1], got {radius_fraction}"
        )
    yx = torch.as_tensor(centers_yx).detach().cpu().float().reshape(-1, 2)
    flow = torch.as_tensor(flow).detach().cpu().float()
    baseline = torch.as_tensor(baseline_flow).detach().cpu().float()
    if flow.shape != baseline.shape or flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(
            "flow and baseline_flow must share shape (frames, positions, 2), "
            f"got {tuple(flow.shape)} and {tuple(baseline.shape)}"
        )
    if flow.shape[1] != len(yx):
        raise ValueError(
            f"coordinate count {len(yx)} does not match flow positions {flow.shape[1]}"
        )
    radius = torch.linalg.vector_norm(yx - yx.mean(0), dim=1)
    mask = radius <= radius_fraction * radius.max()
    delta = (flow - baseline)[frame_slice]
    u = delta[:, :, 0]
    v = -delta[:, :, 1] if flip_y_for_display else delta[:, :, 1]
    magnitude = torch.hypot(u, v)
    theta = torch.deg2rad(torch.tensor(float(direction_deg)))
    direction_y = -torch.sin(theta) if flip_y_for_display else torch.sin(theta)
    projection = u * torch.cos(theta) + v * direction_y
    cosine = torch.where(magnitude > 1e-8, projection / magnitude, 0)
    vector_mean = torch.stack(
        [u[:, mask].mean(1), v[:, mask].mean(1)], dim=1
    )
    return {
        "n_unique": int(len(yx)),
        "central_nodes": int(mask.sum()),
        "mean_flow_magnitude": float(magnitude[:, mask].mean()),
        "peak_flow_magnitude": float(magnitude[:, mask].max()),
        "direction_projection": float(projection[:, mask].mean()),
        "direction_cosine": float(cosine[:, mask].mean()),
        "roi_vector_mean_magnitude": float(
            torch.linalg.vector_norm(vector_mean, dim=1).mean()
        ),
    }


def idw_resample(source_yx, source_vectors, target_yx, k=6, eps=1e-8):
    """Interpolate vectors to target coordinates using normalized IDW."""
    if not 1 <= k <= len(source_yx):
        raise ValueError(f"k must be between 1 and {len(source_yx)}, got {k}")
    distance, index = torch.topk(
        torch.cdist(target_yx, source_yx), k, largest=False
    )
    weight = (distance + eps).reciprocal()
    weight /= weight.sum(1, keepdim=True)
    mapped = (source_vectors[:, index] * weight[None, :, :, None]).sum(2)
    return mapped, distance, weight, index
