"""Adapted YouTube URL validation, cookie options and verified local-media handling from Hogline."""

import hashlib
import json
import re
import subprocess
from bisect import bisect_left, bisect_right
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def youtube_id(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Paste a complete YouTube URL")
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        value = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        value = parse_qs(parsed.query).get("v", [""])[0]
        if parsed.path.startswith(("/live/", "/embed/", "/shorts/")):
            value = parsed.path.split("/")[2]
    else:
        raise ValueError("Use a YouTube recording URL")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError("Could not find a valid YouTube recording ID")
    return value


def command(args, timeout=120):
    try:
        return subprocess.run(args, capture_output=True, check=True, timeout=timeout).stdout
    except FileNotFoundError as exc:
        raise ValueError(f"Install {args[0]} first") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{args[0]} timed out; retry to resume") from exc
    except subprocess.CalledProcessError as exc:
        # Avoid persisting signed URLs, browser cookie details or API secrets.
        raise ValueError(f"{args[0]} failed (exit {exc.returncode})") from exc


def probe(path):
    data = json.loads(
        command(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    )
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("No video stream")
    return dict(
        duration=float(data["format"]["duration"]),
        width=video["width"],
        height=video["height"],
        fps=video["avg_frame_rate"],
        video_start=float(video.get("start_time", 0)),
        format_start=float(data["format"].get("start_time", 0)),
        bytes=Path(path).stat().st_size,
    )


def fingerprint(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(config, identifier, progress):
    root = config.data / "recordings" / identifier
    root.mkdir(parents=True, exist_ok=True)
    options = ["yt-dlp", "--ignore-config", "--no-playlist"]
    if config.browser:
        browser = config.browser
        if browser.lower() == "orion":
            browser = f"safari:{Path.home() / 'Library/HTTPStorages/com.kagi.kagimacOS.binarycookies'}"
        options += ["--cookies-from-browser", browser]
    url = f"https://www.youtube.com/watch?v={identifier}"
    progress("Reading YouTube metadata", 0.02)
    raw = json.loads(command([*options, "--skip-download", "--dump-single-json", url], 120))
    if raw.get("is_live") or raw.get("live_status") in {"is_live", "is_upcoming"}:
        raise ValueError("Wait for this stream to finish before splitting it")
    if not raw.get("duration"):
        raise ValueError("YouTube did not report a recording duration")
    progress("Downloading original video", 0.06)
    command(
        [
            *options,
            "--abort-on-unavailable-fragments",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "--socket-timeout",
            "20",
            "--concurrent-fragments",
            "2",
            "--no-progress",
            "-f",
            "bv*[height<=1080]+ba/b[height<=1080]",
            "--merge-output-format",
            "mp4",
            "-o",
            str(root / "source.%(ext)s"),
            url,
        ],
        14400,
    )
    paths = [p for p in root.glob("source.*") if p.suffix in {".mp4", ".webm", ".mkv"}]
    if len(paths) != 1:
        raise ValueError("Expected one complete downloaded video")
    media = probe(paths[0])
    if abs(media["duration"] - raw["duration"]) > 5 or abs(media["video_start"]) > 1:
        raise ValueError("Downloaded recording does not cover the expected timeline")
    progress("Verifying download", 0.12)
    return dict(
        title=raw.get("title", identifier),
        source_path=str(paths[0].resolve()),
        source_sha256=fingerprint(paths[0]),
        media=media,
    )


def frame(path, seconds, width=960):
    return command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-ss",
            str(seconds),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-threads",
            "2",
            "-f",
            "image2pipe",
            "-c:v",
            "mjpeg",
            "pipe:1",
        ],
        45,
    )


def keyframes(path, cache):
    if cache.is_file():
        return json.loads(cache.read_text())
    rows = command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "csv=p=0",
            str(path),
        ],
        180,
    )
    values = []
    for line in rows.decode().splitlines():
        fields = line.split(",")
        if len(fields) >= 2 and "K" in fields[1]:
            try:
                values.append(float(fields[0]))
            except ValueError:
                pass
    values = sorted(set(values))
    if not values:
        raise ValueError("Cannot locate safe video cut points")
    cache.write_text(json.dumps(values))
    return values


def enclosing_cut(start, end, keys, duration):
    lo = keys[max(0, bisect_right(keys, start) - 1)]
    idx = bisect_left(keys, end)
    hi = keys[idx] if idx < len(keys) else duration
    if lo > start + 0.05 or hi < end - 0.05:
        raise ValueError("Cannot safely include the requested end")
    return max(0.0, lo), min(duration, hi)


def first_video_packet(path, seek=None):
    interval = "%+#1" if seek is None else f"{seek}%+#1"
    data = json.loads(
        command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                interval,
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-show_entries",
                "packet=pts_time,flags,data_hash",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    packets = data.get("packets", [])
    if not packets or "K" not in packets[0].get("flags", ""):
        raise ValueError("Cannot verify the clip's initial keyframe")
    return packets[0]


def export_ends(config, record, progress, filename_prefix="end"):
    root = config.data / "recordings" / record["id"]
    path = Path(record["source_path"])
    keys = keyframes(path, root / f"keyframes-{record['source_sha256'][:16]}.json")
    clips = []
    for i, end in enumerate(record["ends"]):
        progress(f"Exporting end {i + 1} of {len(record['ends'])}", 0.85 + 0.14 * i / len(record["ends"]))
        lo, hi = enclosing_cut(end["start_seconds"], end["end_seconds"], keys, record["media"]["duration"])
        name = f"{filename_prefix}-{i + 1:02}-r{record['revision']}.mp4"
        target = root / name
        temp = root / (name + ".partial.mp4")
        command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(lo),
                "-i",
                str(path),
                "-t",
                str(hi - lo),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(temp),
            ],
            900,
        )
        info = probe(temp)
        # Demuxer seeking can retain the GOP BEFORE the requested keyframe. Verify the
        # actual copied packet, rather than assuming -ss established the source origin.
        source_packet = first_video_packet(path, lo)
        clip_packet = first_video_packet(temp)
        if source_packet["data_hash"] != clip_packet["data_hash"]:
            raise ValueError("Cannot verify the exported clip's source-time origin")
        actual_lo = float(source_packet["pts_time"]) - record["media"].get("format_start", 0)
        if actual_lo > end["start_seconds"] + 0.001 or actual_lo < -0.05:
            raise ValueError("Export omitted the beginning of the requested end")
        if abs(info["duration"] - (hi - actual_lo)) > 2:
            raise ValueError("Exported clip duration does not match the requested span")
        temp.replace(target)
        clips.append(
            dict(
                number=i + 1,
                filename=name,
                revision=record["revision"],
                requested_start=end["start_seconds"],
                requested_end=end["end_seconds"],
                source_start=max(0.0, actual_lo),
                requested_keyframe_start=lo,
                initial_packet_sha256=clip_packet["data_hash"],
                source_end=hi,
                clip_video_start=info["video_start"],
                duration=info["duration"],
                bytes=info["bytes"],
                sha256=fingerprint(target),
            )
        )
    return clips
