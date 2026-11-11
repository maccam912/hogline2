"""Ice-plane calibration. Image coordinates always refer to the original composite."""

import hashlib
import json
import math

from .tracking_dimensions import RADIUS as RADIUS
from .tracking_dimensions import STANDARD_SHEET, TEE_DISTANCE, sheet_geometry


def solve(a, b):
    rows = [list(map(float, row)) + [float(value)] for row, value in zip(a, b, strict=True)]
    n = len(b)
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(rows[i][col]))
        if abs(rows[pivot][col]) < 1e-10:
            raise ValueError("Calibration points are degenerate")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for i in range(n):
            if i != col:
                scale = rows[i][col]
                rows[i] = [v - scale * w for v, w in zip(rows[i], rows[col], strict=True)]
    return [row[-1] for row in rows]


def homography(pairs):
    if len(pairs) != 4:
        raise ValueError("Four ring control points are required")
    a, b = [], []
    for (x, y), (u, v) in pairs:
        a.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        b.extend([u, v])
    return solve(a, b) + [1.0]


def project(h, point):
    x, y = point
    z = h[6] * x + h[7] * y + h[8]
    if abs(z) < 1e-8:
        raise ValueError("Point is at the calibration horizon")
    return [(h[0] * x + h[1] * y + h[2]) / z, (h[3] * x + h[4] * y + h[5]) / z]


def camera_reference(view, tee_distance=TEE_DISTANCE, geometry=None):
    """Bind a fitted camera to its exact crop, landmarks, orientation and sheet scale."""
    reference = {
        "id": view["id"],
        "kind": view["kind"],
        "end": view["end"],
        "rotation": view.get("rotation_degrees", 0),
        "crop": list(map(float, view["crop"])),
        "landmarks": {k: list(map(float, v)) for k, v in view["landmarks"].items()},
        "tee_distance": float(tee_distance),
        "geometry": {k: float(v) for k, v in (geometry or STANDARD_SHEET).items()},
    }
    return hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()


def inverse_homography(h):
    if len(h) != 9 or not all(math.isfinite(v) for v in h):
        raise ValueError("Camera transform must contain nine finite values")
    matrix = [h[i : i + 3] for i in (0, 3, 6)]
    columns = [solve(matrix, [float(i == j) for i in range(3)]) for j in range(3)]
    return [columns[j][i] for i in range(3) for j in range(3)]


def world_points(end, tee_distance=TEE_DISTANCE, geometry=None):
    geometry = geometry or STANDARD_SHEET
    radius = geometry["house_radius_m"]
    y, direction = (0.0, 1) if end == "A" else (tee_distance, -1)
    return {
        "left": [-radius * direction, y],
        "right": [radius * direction, y],
        "back": [0, y - radius * direction],
        "front": [0, y + radius * direction],
        "button": [0, y],
        "hack": [0, y - geometry["hack_from_tee_m"] * direction],
    }


def prepare_view(view, tee_distance=None, geometry=None):
    tee_distance = tee_distance or view.get("tee_distance_m", TEE_DISTANCE)
    points = view["landmarks"]
    expected = world_points(view["end"], tee_distance, geometry)
    if view.get("rotation_degrees", 0) == 180:
        cy = expected["button"][1]
        expected = {name: [-xy[0], 2 * cy - xy[1]] for name, xy in expected.items()}
    pairs = [(points[name], expected[name]) for name in ("left", "right", "back", "front")]
    h = homography(pairs)
    inverse = homography([(b, a) for a, b in pairs])
    camera = view.get("camera_calibration")
    if camera:
        if camera["reference_sha256"] != camera_reference(view, tee_distance, geometry):
            raise ValueError(f"{view['id']}: camera fit does not match the current landmarks or sheet")
        h = list(camera["image_to_ice"])
        inverse = inverse_homography(h)
    error = math.dist(project(h, points["button"]), expected["button"])
    if error > 0.25:
        raise ValueError(f"{view['id']}: button disagrees with ring calibration by {error:.2f} m")
    return {
        **view,
        "tee_distance_m": tee_distance,
        "image_to_ice": h,
        "ice_to_image": inverse,
        "button_error_m": error,
    }


def validate_calibration(value, record):
    if value.get("source_sha256") != record["source_sha256"]:
        raise ValueError("Calibration belongs to a different source recording")
    if not value.get("views"):
        raise ValueError("Define at least one camera view")
    tee_distance = value.get("tee_distance_m", TEE_DISTANCE)
    if not math.isfinite(tee_distance) or not 25 <= tee_distance <= 45:
        raise ValueError("Button spacing must be finite and between 25 and 45 metres")
    geometry = sheet_geometry(value)
    if any(not math.isfinite(v) or v <= 0 for v in geometry.values()):
        raise ValueError("Sheet dimensions must be finite and positive")
    margin = (geometry["length_m"] - tee_distance) / 2
    if margin <= geometry["hack_from_tee_m"]:
        raise ValueError("Hacks must fit between the buttons and backboards")
    if geometry["width_m"] < 2 * geometry["house_radius_m"]:
        raise ValueError("The house must fit within the sheet width")
    if 2 * geometry["hog_from_tee_m"] >= tee_distance:
        raise ValueError("Hog lines must lie between the buttons")
    w, h = record["media"]["width"], record["media"]["height"]
    names = set()
    for view in value["views"]:
        if not view["id"].replace("_", "").isalnum() or view["id"] in names:
            raise ValueError("View names must be unique letters, digits and underscores")
        names.add(view["id"])
        if view["kind"] not in {"perspective", "overhead"} or view["end"] not in {
            "A",
            "B",
        }:
            raise ValueError("Select a view type and physical house A or B")
        x0, y0, x1, y1 = view["crop"]
        if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
            raise ValueError("View crop lies outside the source image")
        for name in ("left", "right", "back", "front", "button"):
            p = view["landmarks"].get(name)
            if not p or len(p) != 2 or not all(math.isfinite(v) for v in p):
                raise ValueError(f"{view['id']}: mark {name}")
            if not x0 <= p[0] <= x1 or not y0 <= p[1] <= y1:
                raise ValueError("Landmark lies outside its view")
        hack = view["landmarks"].get("hack")
        if hack and (
            len(hack) != 2
            or not all(math.isfinite(v) for v in hack)
            or not x0 <= hack[0] <= x1
            or not y0 <= hack[1] <= y1
        ):
            raise ValueError("Hack lies outside its view")
        camera = view.get("camera_calibration")
        if camera and camera["reference_sha256"] != camera_reference(view, tee_distance, geometry):
            # Editing landmarks/crops/dimensions invalidates the previous fit.
            view.pop("camera_calibration", None)
        prepare_view(view, tee_distance, geometry)
    return value


def segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy or 1)))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def projection_variance(view, point, pixel_sigma=2.0):
    """Diagonal approximation to image-to-ice uncertainty via a local Jacobian.

    The floor also accounts for imperfect ring calibration/body-height estimates.
    This is model uncertainty, not an empirically calibrated confidence guarantee.
    """
    h = view["image_to_ice"]
    xy = project(h, point)
    dx = project(h, [point[0] + 1, point[1]])
    dy = project(h, [point[0], point[1] + 1])
    local_y = xy[1] if view["end"] == "A" else view.get("tee_distance_m", TEE_DISTANCE) - xy[1]
    floor = 0.045 if view["kind"] == "overhead" else 0.08 + abs(local_y) * 0.015
    return [
        min(
            25.0,
            floor**2 + pixel_sigma**2 * ((dx[i] - xy[i]) ** 2 + (dy[i] - xy[i]) ** 2),
        )
        for i in range(2)
    ]
