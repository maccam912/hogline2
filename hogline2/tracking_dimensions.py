"""Shared standard sheet dimensions; venue values remain calibration data."""

RADIUS = 1.829
TEE_DISTANCE = 34.747
STANDARD_SHEET = {
    "length_m": 45.720,
    "width_m": 4.750,
    "hog_from_tee_m": 6.401,
    "hack_from_tee_m": 3.658,
    "back_from_tee_m": 1.829,
    "house_radius_m": 1.829,
}


def sheet_geometry(calibration=None):
    return {**STANDARD_SHEET, **(calibration or {}).get("sheet_geometry", {})}
