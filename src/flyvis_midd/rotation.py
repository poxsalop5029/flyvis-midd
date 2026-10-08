"""CW-positive rotation metrics."""
import numpy as np


def rotation_basis(centers_yx, center_yx=None):
    centers = np.asarray(centers_yx, float).reshape(-1,2)
    center = centers.mean(0) if center_yx is None else np.asarray(center_yx, float)
    delta = centers-center; radial = np.column_stack((delta[:,1], -delta[:,0]))
    radius = np.linalg.norm(radial, axis=1); radial /= np.maximum(radius, np.finfo(float).eps)[:,None]
    tangent = np.column_stack((radial[:,1], -radial[:,0]))
    return radius, radial, tangent


def rotation_flow_metrics(centers_yx, flow_uv, *, center_yx=None,
                          inner_radius_fraction=.14, outer_radius_fraction=.98,
                          flip_decoder_y=True, flip_y_for_display=None):
    if flip_y_for_display is not None: flip_decoder_y = flip_y_for_display
    flow = np.asarray(flow_uv, float); centers = np.asarray(centers_yx, float).reshape(-1,2)
    if flow.shape[-2:] != (len(centers),2): raise ValueError('flow must end in (positions,2)')
    if flip_decoder_y:
        flow = flow.copy(); flow[...,1] *= -1
    radius, radial, tangent = rotation_basis(centers, center_yx)
    mask = (radius >= inner_radius_fraction*radius.max()) & (radius <= outer_radius_fraction*radius.max())
    if not mask.any(): raise ValueError('annulus contains no positions')
    selected = flow[...,mask,:]; magnitude=np.linalg.norm(selected,axis=-1)
    tangential=(selected*tangent[mask]).sum(-1); radial_component=(selected*radial[mask]).sum(-1)
    mean_mag=float(magnitude.mean()); mean_tangent=float(tangential.mean())
    return {'mean_flow_magnitude':mean_mag,'mean_signed_tangent':mean_tangent,
            'mean_abs_tangent':float(np.abs(tangential).mean()),
            'signed_tangent_ratio':mean_tangent/max(mean_mag,np.finfo(float).eps),
            'rotation_coherence':abs(mean_tangent)/max(float(np.abs(tangential).mean()),np.finfo(float).eps),
            'mean_signed_radial':float(radial_component.mean()),
            'annulus_positions':int(mask.sum()),'annulus_receptors':int(mask.sum()),'annulus_mask':mask}


def image_rotation_flow_metrics(centers_yx, flow_uv, *, center_yx=None,
                                inner_radius_fraction=.14,
                                outer_radius_fraction=.98,
                                flip_decoder_y=True,
                                flip_y_for_display=None):
    """CW-positive metric in image coordinates, matching fvpg MIDD Notebook 01.

    ``centers_yx`` must use image coordinates (+y downward), normally
    ``BoxEye.receptor_centers``. Decoder ``v`` is negated exactly once before
    projection, matching the canonical ``(u, v) -> (u, -v)`` contract.
    """
    if flip_y_for_display is not None:
        flip_decoder_y = flip_y_for_display
    flow = np.asarray(flow_uv, float)
    centers = np.asarray(centers_yx, float).reshape(-1, 2)
    if flow.shape[-2:] != (len(centers), 2):
        raise ValueError("flow must end in (positions,2)")
    if flip_decoder_y:
        flow = flow.copy()
        flow[..., 1] *= -1
    center = centers.mean(0) if center_yx is None else np.asarray(center_yx, float)
    dy = centers[:, 0] - center[0]
    dx = centers[:, 1] - center[1]
    radius = np.hypot(dx, dy)
    denom = np.maximum(radius, np.finfo(float).eps)
    radial = np.column_stack((dx / denom, dy / denom))
    tangent = np.column_stack((-dy / denom, dx / denom))
    mask = ((radius >= inner_radius_fraction * radius.max())
            & (radius <= outer_radius_fraction * radius.max()))
    if not mask.any():
        raise ValueError("annulus contains no positions")
    selected = flow[..., mask, :]
    magnitude = np.linalg.norm(selected, axis=-1)
    tangential = (selected * tangent[mask]).sum(-1)
    radial_component = (selected * radial[mask]).sum(-1)
    mean_mag = float(magnitude.mean())
    mean_tangent = float(tangential.mean())
    mean_abs_tangent = float(np.abs(tangential).mean())
    eps = np.finfo(float).eps
    return {
        "mean_flow_magnitude": mean_mag,
        "mean_signed_tangent": mean_tangent,
        "mean_abs_tangent": mean_abs_tangent,
        "signed_tangent_ratio": mean_tangent / max(mean_mag, eps),
        "rotation_coherence": abs(mean_tangent) / max(mean_abs_tangent, eps),
        "mean_signed_radial": float(radial_component.mean()),
        "annulus_positions": int(mask.sum()),
        "annulus_receptors": int(mask.sum()),
        "annulus_mask": mask,
    }
