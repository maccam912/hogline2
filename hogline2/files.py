"""Small JSON-on-disk helpers shared by the web app and the tracking process."""

import json
import uuid
from pathlib import Path


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except ValueError:
        return default


def write_json(path, value, indent=None):
    """Write through a temporary file so a reader never sees half a document."""
    path = Path(path)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=indent))
    temp.replace(path)
