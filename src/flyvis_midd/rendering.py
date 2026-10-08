"""Reusable rendering helpers for Flyvis notebooks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def as_gray_tensor(image):
    """Return a detached float32 grayscale image with shape ``(H, W)``."""
    image = torch.as_tensor(image).detach().cpu()
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(-1)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        return image[:3].mean(0)
    raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")


def resize_to_boxeye_frame(image, receptors):
    """Resize a small image to the frame size required by ``BoxEye``."""
    image = as_gray_tensor(image)
    size = tuple(int(value) for value in receptors.min_frame_size)
    if tuple(image.shape) == size:
        return image
    return F.interpolate(
        image[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0]


def manual_crop_means(image, receptors):
    """Reproduce each BoxEye value from its local kernel mean."""
    image = as_gray_tensor(image)
    height, width = image.shape
    centers = receptors.receptor_centers.detach().cpu()
    centers = (centers + torch.tensor([height // 2, width // 2])).long()
    left, right, top, bottom = receptors.pad
    padded = F.pad(image[None, None], (left, right, top, bottom))[0, 0]
    kernel = int(receptors.kernel_size)
    patches = torch.stack([
        padded[y:y + kernel, x:x + kernel]
        for y, x in centers.tolist()
    ])
    return patches.mean((1, 2))


def compare_renderers(image, receptors, *, resize=False, atol=1e-6):
    """Compare official BoxEye output with manual crop means."""
    image = resize_to_boxeye_frame(image, receptors) if resize else as_gray_tensor(image)
    official = receptors(image[None, None]).squeeze().detach().cpu()
    manual = manual_crop_means(image, receptors)
    difference = official - manual
    return {
        "image": image,
        "official": official,
        "manual": manual,
        "difference": difference,
        "allclose": torch.allclose(official, manual, atol=atol, rtol=0),
        "max_abs_diff": difference.abs().max().item(),
        "mean_abs_diff": difference.abs().mean().item(),
    }


def make_layout_geometry(row_counts, receptors):
    """Build staggered field centers and receptor coordinates."""
    row_counts = tuple(row_counts)
    local_yx = receptors.receptor_centers.detach().cpu().float()
    span_y = float(local_yx[:, 0].max() - local_yx[:, 0].min())
    span_x = float(local_yx[:, 1].max() - local_yx[:, 1].min())
    row_step = span_y * 3 / 4
    max_count = max(row_counts)
    equal_row_counts = len(set(row_counts)) == 1
    centers = []
    for row, count in enumerate(row_counts):
        offset = (
            span_x / 2 * (row % 2)
            if equal_row_counts
            else span_x / 2 * (max_count - count)
        )
        centers.extend((row * row_step, col * span_x + offset) for col in range(count))
    field_centers = torch.tensor(centers)
    receptor_centers = field_centers[:, None, :] + local_yx[None, :, :]
    receptor_bounds_min = receptor_centers.reshape(-1, 2).min(0).values
    receptor_bounds_max = receptor_centers.reshape(-1, 2).max(0).values
    layout_center_yx = (receptor_bounds_min + receptor_bounds_max) / 2
    middle_row = len(row_counts) // 2
    if len(row_counts) % 2 == 1 and row_counts[middle_row] % 2 == 1:
        center_index = sum(row_counts[:middle_row]) + row_counts[middle_row] // 2
        square_center_yx = field_centers[center_index]
        bottom_left_index = sum(row_counts[:-1])
        distance_a = square_center_yx[0] - receptor_centers[0, :, 0].min()
        outer_left_x = torch.minimum(
            receptor_centers[0, :, 1].min(),
            receptor_centers[bottom_left_index, :, 1].min(),
        )
        distance_b = square_center_yx[1] - outer_left_x
    else:
        square_center_yx = layout_center_yx
        distance_a, distance_b = square_center_yx - receptor_bounds_min
    square_half_size = torch.minimum(distance_a, distance_b)
    bounds_half_size = torch.min(torch.minimum(
        layout_center_yx - receptor_bounds_min,
        receptor_bounds_max - layout_center_yx,
    ))
    return {
        "row_counts": row_counts,
        "field_centers": field_centers,
        "receptor_centers": receptor_centers,
        "layout_center_yx": layout_center_yx,
        "receptor_bounds_min": receptor_bounds_min,
        "receptor_bounds_max": receptor_bounds_max,
        "square_half_size": square_half_size,
        "bounds_half_size": bounds_half_size,
        # Explicit aliases used when documenting the inscribed-square contract.
        "image_center_yx": square_center_yx,
        "distance_a": distance_a,
        "distance_b": distance_b,
        "inscribed_half_size": square_half_size,
        "image_top_left_yx": square_center_yx - square_half_size,
        "image_bottom_right_yx": square_center_yx + square_half_size,
        "top_left": square_center_yx - square_half_size,
        "bottom_right": square_center_yx + square_half_size,
        "span_y": span_y,
        "span_x": span_x,
        "target_height": (len(row_counts) - 1) * row_step + span_y / 2,
        "target_width": (max_count - 1) * span_x,
    }


def place_image_in_layout(frame, geometry, *, fit_mode="preserve_aspect"):
    """Scale and center one frame within the honeycomb layout.

    ``preserve_aspect`` fits the original aspect ratio in the layout target.
    ``inscribed_square`` reproduces the square placement used by snakeunit.
    ``full_vertical`` fits the source to the inscribed vertical span while
    preserving aspect ratio, as used by the Ouchi stimulus contract.
    """
    frame = torch.as_tensor(frame).detach().cpu().float()
    if fit_mode == "preserve_aspect":
        scale = min(
            geometry["target_height"] / frame.shape[0],
            geometry["target_width"] / frame.shape[1],
        )
        size = tuple(round(scale * value) for value in frame.shape)
    elif fit_mode == "inscribed_square":
        side = int(round(float(2 * geometry["square_half_size"]))) + 1
        size = (side, side)
    elif fit_mode == "full_vertical":
        height = int(round(float(2 * geometry["bounds_half_size"]))) + 1
        width = int(round(height * frame.shape[1] / frame.shape[0]))
        size = (height, width)
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode!r}")
    image = F.interpolate(
        frame[None, None], size=size, mode="bilinear", align_corners=False
    )[0, 0]
    image_size_yx = torch.tensor(image.shape, dtype=torch.float32)
    if fit_mode == "inscribed_square":
        origin_yx = geometry["image_center_yx"] - geometry["square_half_size"]
    elif fit_mode == "full_vertical":
        origin_yx = geometry["layout_center_yx"] - (image_size_yx - 1) / 2
    else:
        origin_yx = geometry["layout_center_yx"] - image_size_yx / 2
    return image, origin_yx


def crop_means_at_centers(
    image,
    centers_yx,
    origin_yx,
    kernel_size,
    *,
    padding_mode="constant",
    padding_value=0.0,
):
    """Compute local means at arbitrary coordinates using average pooling."""
    centers = torch.round(centers_yx - origin_yx).long()
    height, width = image.shape
    half = kernel_size // 2
    outside = max(
        0,
        int(-centers[:, 0].min()),
        int(-centers[:, 1].min()),
        int(centers[:, 0].max() - height + 1),
        int(centers[:, 1].max() - width + 1),
    )
    pad = half + outside
    if padding_mode == "constant":
        padded = F.pad(
            image[None, None],
            (pad, pad, pad, pad),
            mode=padding_mode,
            value=float(padding_value),
        )
    else:
        padded = F.pad(
            image[None, None], (pad, pad, pad, pad), mode=padding_mode
        )
    mean_map = F.avg_pool2d(padded, kernel_size, stride=1)[0, 0]
    indices = centers + pad - half
    return mean_map[indices[:, 0], indices[:, 1]]


def render_layout_frame(
    frame,
    receptors,
    row_counts,
    geometry=None,
    *,
    fit_mode="preserve_aspect",
    padding_mode="constant",
    padding_value=0.0,
):
    """Render one frame and return values plus geometry for plotting."""
    geometry = geometry or make_layout_geometry(row_counts, receptors)
    image, origin = place_image_in_layout(frame, geometry, fit_mode=fit_mode)
    flat_values = crop_means_at_centers(
        image,
        geometry["receptor_centers"].reshape(-1, 2),
        origin,
        int(receptors.kernel_size),
        padding_mode=padding_mode,
        padding_value=padding_value,
    )
    result = dict(geometry)
    result.update({
        "values": flat_values.reshape(len(geometry["field_centers"]), -1),
        "image": image,
        "image_origin": origin,
    })
    return result


def render_layout_video(
    video,
    receptors,
    row_counts=(3, 2, 3),
    *,
    fit_mode="preserve_aspect",
    padding_mode="constant",
    padding_value=0.0,
):
    """Render one ``(frame, height, width)`` movie into layout receptors."""
    geometry = make_layout_geometry(row_counts, receptors)
    values = []
    first_image = first_origin = None
    for frame_index, frame in enumerate(video):
        image, origin = place_image_in_layout(frame, geometry, fit_mode=fit_mode)
        flat = crop_means_at_centers(
            image,
            geometry["receptor_centers"].reshape(-1, 2),
            origin,
            int(receptors.kernel_size),
            padding_mode=padding_mode,
            padding_value=padding_value,
        )
        values.append(flat.reshape(len(geometry["field_centers"]), -1))
        if frame_index == 0:
            first_image, first_origin = image, origin
    result = dict(geometry)
    result.update({
        "values": torch.stack(values),
        "image": first_image,
        "image_origin": first_origin,
    })
    return result


def image_bounds(image_origin_yx, image_shape_yx):
    """Return inclusive plotting bounds for a placed image."""
    origin = torch.as_tensor(image_origin_yx).detach().cpu().float().reshape(2)
    shape = torch.as_tensor(image_shape_yx).detach().cpu().float().reshape(2)
    return {
        "x0": float(origin[1]),
        "x1": float(origin[1] + shape[1]),
        "y0": float(origin[0]),
        "y1": float(origin[0] + shape[0]),
    }
