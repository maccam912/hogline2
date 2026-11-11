"""Unfiltered per-camera measurements; never interpolate across missing frames."""

import math

from .tracking_geometry import project


def raw_mask_measurement(mask, view, anchor=None):
    import numpy as np

    yy, xx = np.where(mask)
    if not len(xx):
        return None
    if anchor is not None:
        point = list(anchor)
    elif view["kind"] == "overhead":
        point = [float(xx.mean()), float(yy.mean())]
    else:
        bottom = np.quantile(yy, 0.94)
        point = [float(np.median(xx[yy >= bottom])), float(yy.max())]
    point = [point[0] + view["crop"][0], point[1] + view["crop"][1]]
    try:
        position = project(view["image_to_ice"], point)
    except ValueError:
        return None
    if not all(math.isfinite(x) for x in position):
        return None
    return dict(
        point=point,
        xy=position,
        area=int(mask.sum()),
        bbox=[int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
        clipped=bool(
            xx.min() == 0 or yy.min() == 0 or xx.max() == mask.shape[1] - 1 or yy.max() == mask.shape[0] - 1
        ),
    )


def add_raw_velocity(previous, observation, frame, seconds):
    """No gates, clipping, smoothing, gap bridging, or cross-view differencing."""
    output = {
        **observation,
        "frame": frame,
        "source_seconds": seconds,
        "velocity": None,
        "speed_mps": None,
        "pixel_speed": None,
    }
    same_track = previous is not None and all(
        previous.get(key) == observation.get(key) for key in ("view_id", "stone_id")
    )
    if same_track and previous["frame"] == frame - 1:
        dt = seconds - previous["source_seconds"]
        if dt > 0:
            velocity = [(observation["xy"][j] - previous["xy"][j]) / dt for j in range(2)]
            output.update(
                velocity=velocity,
                speed_mps=math.hypot(*velocity),
                pixel_speed=math.dist(observation["point"], previous["point"]) / dt,
            )
    return output
