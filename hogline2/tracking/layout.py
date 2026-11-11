"""Where the camera panels sit in the broadcast mosaic, and which end each end is thrown from.

stonetrack calibrates every camera from the paint on the ice, so all it needs from here is the
panel layout: a crop, a camera kind and the house it looks at, per panel. The layout that ships
with stonetrack (the club's four-view mosaic) is the default. A different mosaic can be described
in `data/layout.json`, or for one recording in `data/recordings/<id>/layout.json`.
"""

from ..config import PROJECT
from ..files import read_json

DEFAULT_LAYOUT = PROJECT / "stonetrack" / "layouts" / "wausau.json"


def camera_layout(config, record):
    """The panel layout for one recording, scaled to its frame size."""
    media = record.get("media") or {}
    if not media.get("width"):
        raise ValueError("The recording has not been downloaded yet")
    layout = (
        read_json(config.data / "recordings" / record["id"] / "layout.json")
        or read_json(config.data / "layout.json")
        or read_json(DEFAULT_LAYOUT)
    )
    sx, sy = media["width"] / layout["width"], media["height"] / layout["height"]
    if abs(sx - sy) > 0.001:
        raise ValueError(
            f"The camera layout is {layout['width']}×{layout['height']} but this recording is "
            f"{media['width']}×{media['height']}. Add a layout.json for this mosaic."
        )
    views = [
        dict(
            id=v["id"],
            kind=v["kind"],
            end=v["end"],
            crop=[int(round(c * (sx if i % 2 == 0 else sy))) for i, c in enumerate(v["crop"])],
            rotation_degrees=v.get("rotation_degrees", 0),
        )
        for v in layout["views"]
    ]
    return dict(name=layout.get("name", "custom"), width=media["width"], height=media["height"], views=views)


def other(end):
    return "A" if end == "B" else "B"


def delivery_end(record, number):
    """Ends alternate direction; the first end's direction is a per-recording setting."""
    first = record.get("first_delivery_end", "B")
    return first if number % 2 == 1 else other(first)
