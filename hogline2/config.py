import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    data: Path
    api_key: str = field(default="", repr=False)
    model: str = "gpt-6.1-terra"
    browser: str = ""

    @classmethod
    def load(cls, data=None):
        values = {}
        path = PROJECT / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key in {"OPENAI_API_KEY", "HOGLINE2_VISION_MODEL", "HOGLINE2_YOUTUBE_BROWSER"}:
                    values[key] = value.strip().strip("\"'")
        values.update(
            {
                k: v
                for k, v in os.environ.items()
                if k in {"OPENAI_API_KEY", "HOGLINE2_VISION_MODEL", "HOGLINE2_YOUTUBE_BROWSER"}
            }
        )
        return cls(
            Path(data or PROJECT / "data").resolve(),
            values.get("OPENAI_API_KEY", ""),
            values.get("HOGLINE2_VISION_MODEL", "gpt-6.1-terra"),
            values.get("HOGLINE2_YOUTUBE_BROWSER", ""),
        )
