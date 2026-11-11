"""Portable, job-bound Colab bundles. No account credentials or server tunnel."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

from .media import frame, probe
from .tracking_jobs import atomic
from .tracking_sam2_specs import CHECKPOINTS, SAM_REVISION

SCHEMA = 1
MAX_UPLOAD = 2 * 1024**3
MAX_EXPANDED = 4 * 1024**3
SOURCES = (
    "__init__",
    "media",
    "segmentation",
    "tracking_runner",
    "tracking_geometry",
    "tracking_dimensions",
    "tracking_jobs",
    "tracking_proposals",
    "tracking_stone",
    "tracking_state",
    "tracking_motion",
    "tracking_smoothing",
    "tracking_calibration_fit",
    "tracking_sam",
    "tracking_sam2_granite",
    "tracking_sam2_cuda",
    "tracking_sam2_specs",
    "tracking_colab",
    "tracking_colab_worker",
)
RESULT_FILES = {
    "manifest.json",
    "status.json",
    "observations.jsonl",
    "events.json",
    "overlay.mp4",
    "trajectory.json",
    "trajectory-summary.json",
    "geometry-report.json",
}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text())


def prepare(job_path):
    folder = job_path.parent
    job = json_file(job_path)
    atomic(
        folder / "status.json",
        dict(
            state="running",
            stage="Preparing Colab video bundle",
            progress=0,
            pid=os.getpid(),
            started_at=time.time(),
        ),
    )
    if digest(job["source_path"]) != job["source_sha256"]:
        raise ValueError("Source recording changed; prepare a new run")
    with tempfile.TemporaryDirectory(prefix="colab-", dir=folder) as temporary:
        staging = Path(temporary)
        video = staging / "media.mp4"
        with (folder / "prepare.log").open("wb") as log:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-threads",
                    "2",
                    "-ss",
                    str(job["source_start"]),
                    "-i",
                    job["source_path"],
                    "-t",
                    str(job["duration"]),
                    "-vf",
                    f"fps={job['fps']}",
                    "-an",
                    "-c:v",
                    "libx264rgb",
                    "-crf",
                    "0",
                    "-preset",
                    "fast",
                    "-threads",
                    "2",
                    str(video),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        (staging / "background.jpg").write_bytes(
            frame(job["source_path"], job["calibration"]["source_seconds"], job["media"]["width"])
        )
        media = probe(video)
        if (media["width"], media["height"]) != (job["media"]["width"], job["media"]["height"]):
            raise ValueError("Prepared video dimensions changed")
        count = int(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nw=1:nk=1",
                    str(video),
                ],
                text=True,
            ).strip()
        )
        if count < 1 or abs(count - job["duration"] * job["fps"]) > 1:
            raise ValueError("Prepared video has an unexpected frame count")
        portable = {
            **job,
            "source_path": "media.mp4",
            "background_path": "background.jpg",
            "source_media_sha256": digest(video),
            "source_time_origin": job["source_start"],
            "expected_frames": count,
        }
        atomic(staging / "job.json", portable)
        code = staging / "hogline2"
        code.mkdir()
        for module in SOURCES:
            shutil.copyfile(Path(__file__).with_name(module + ".py"), code / (module + ".py"))
        files = {p.relative_to(staging).as_posix(): digest(p) for p in staging.rglob("*") if p.is_file()}
        envelope = dict(schema=SCHEMA, run_id=folder.name, job_sha256=digest(job_path), files=files)
        atomic(staging / "bundle.json", envelope)
        archive = folder / "colab-input.partial.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(staging).as_posix())
        archive.replace(folder / "colab-input.zip")
        atomic(
            folder / "colab-preparation.json",
            dict(
                schema=SCHEMA,
                job_sha256=envelope["job_sha256"],
                frames=count,
                bundle_sha256=digest(folder / "colab-input.zip"),
                source_media_sha256=portable["source_media_sha256"],
            ),
        )
    atomic(
        folder / "status.json",
        dict(
            state="awaiting_colab",
            stage="Ready for Colab — download notebook and video bundle",
            progress=0,
            started_at=time.time(),
        ),
    )


def notebook(folder):
    job = json_file(folder / "job.json")
    info = json_file(folder / "colab-preparation.json")
    size = "large" if "large" in job["backend"] else "base_plus"
    filename, _, sha = CHECKPOINTS[size]
    setup = f"""import sys, subprocess, pathlib, hashlib, urllib.request
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU, then rerun."
print("GPU:", torch.cuda.get_device_name())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "opencv-python-headless",
    "hydra-core", "iopath"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps",
    "git+https://github.com/facebookresearch/sam2.git@{SAM_REVISION}"], check=True)
checkpoints = pathlib.Path("/content/sam21-checkpoints")
checkpoints.mkdir(exist_ok=True)
checkpoint = checkpoints / {filename!r}
if not checkpoint.exists():
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/" + {filename!r}
    urllib.request.urlretrieve(url, checkpoint)
with checkpoint.open("rb") as source:
    assert hashlib.file_digest(source, "sha256").hexdigest() == {sha!r}, "Checkpoint checksum mismatch"
print("SAM2.1 {size} ready")
"""
    load_bundle = (
        """# @title Load the Hogline video bundle
drive_link_or_path = ""  # @param {type:"string"}
# Paste the Drive link for colab-input.zip, or a path under /content/drive/.
# Leave the field blank to upload the ZIP directly from your computer.

import hashlib
import pathlib
import re
import shutil
import tempfile
import zipfile
from urllib.parse import parse_qs, urlsplit

from google.colab import files

"""
        f"expected_bundle_sha256 = {info['bundle_sha256']!r}\n"
        f"workspace = pathlib.Path('/content/hogline-colab-{folder.name}')\n"
        r"""

def download_drive_bundle(link, destination):
    url = urlsplit(link)
    if url.scheme != "https" or url.netloc != "drive.google.com":
        raise ValueError("Paste a Google Drive file link or a path under /content/drive/.")
    query = parse_qs(url.query)
    match = re.match(r"/file/d/([^/]+)(?:/|$)", url.path)
    file_id = match.group(1) if match else ""
    if url.path in {"/open", "/uc"}:
        file_id = query.get("id", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError("Copy the link to colab-input.zip itself in Drive.")
    resource_key = query.get("resourcekey", [""])[0]
    if resource_key and not re.fullmatch(r"[A-Za-z0-9_-]+", resource_key):
        raise ValueError("The Drive file link has an invalid resource key.")

    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    auth.authenticate_user()
    with build("drive", "v3") as drive_service:
        request = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
        if resource_key:
            request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key}"
        with destination.open("wb") as output:
            download = MediaIoBaseDownload(output, request, chunksize=16 * 1024**2)
            done = False
            while not done:
                progress, done = download.next_chunk(num_retries=3)
                if progress:
                    print(f"Drive download: {progress.progress():.0%}", end="\r", flush=True)
    print("Drive download complete.")


if workspace.exists():
    raise FileExistsError("This bundle is already unpacked. Continue to the tracking cell.")

source = drive_link_or_path.strip()
with tempfile.TemporaryDirectory(prefix="hogline-input-", dir=workspace.parent) as temporary:
    scratch = pathlib.Path(temporary)
    archive = scratch / "colab-input.zip"
    if source.startswith("/content/drive/"):
        from google.colab import drive

        drive.mount("/content/drive", readonly=True)
        print("Copying the bundle from Drive…", flush=True)
        shutil.copyfile(source, archive)
    elif source:
        download_drive_bundle(source, archive)
    else:
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload one Hogline colab-input.zip bundle.")
        archive = pathlib.Path(next(iter(uploaded)))
        del uploaded

    with archive.open("rb") as bundle:
        if hashlib.file_digest(bundle, "sha256").hexdigest() != expected_bundle_sha256:
            raise ValueError("Use the complete colab-input.zip prepared for this notebook's run.")
    unpacked = scratch / "bundle"
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            path = pathlib.PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError("Invalid path in the video bundle.")
        package.extractall(unpacked)
    unpacked.rename(workspace)

print("Bundle verified and unpacked:", workspace)
"""
    )
    run = """import subprocess, sys
subprocess.run([sys.executable, "-u", "-m", "hogline2.tracking_colab_worker", str(workspace)],
    cwd=workspace, check=True)
files.download(str(workspace / "colab-results.zip"))
"""

    def cell(kind, source):
        value = dict(
            id=hashlib.sha256((kind + source).encode()).hexdigest()[:12],
            cell_type=kind,
            metadata={},
            source=source.splitlines(keepends=True),
        )
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value

    return dict(
        nbformat=4,
        nbformat_minor=5,
        metadata=dict(
            colab=dict(name=f"Hogline end {job['end']['number']} — SAM2.1 {size}"),
            accelerator="GPU",
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        ),
        cells=[
            cell(
                "markdown",
                f"# Hogline tracking — end {job['end']['number']}\n\n"
                f"{job['duration']:.1f} seconds at {job['fps']:g} fps, SAM2.1 **{size}**.\n\n"
                "1. Select **Runtime → Change runtime type → GPU**.\n"
                "2. In **Load the Hogline video bundle**, paste the Google Drive link for the matching "
                "**colab-input.zip** or its path under `/content/drive/`. Leave the field blank for a "
                "direct upload. Colab may ask you to sign in to access Drive.\n"
                "   Run the cells in order.\n"
                "3. Keep this tab open until **colab-results.zip** finishes downloading.\n"
                "4. Return to Hogline and import that ZIP into this run.\n\n"
                "The selected video interval and calibration frame are uploaded to your Colab runtime. "
                "The notebook uses the same delivery, identity, body validation and "
                "trajectory code as Hogline. "
                "GPU availability and Colab runtime limits apply.\n",
            ),
            cell("code", setup),
            cell("code", load_bundle),
            cell("code", run),
        ],
    )


def allowed_result(name):
    return name in RESULT_FILES or re.fullmatch(r"masks/[0-9]{6,}\.npz", name) is not None


def import_results(folder, archive, record):
    """Validate into a temporary directory; publish status last. Never accept returned code."""
    job = json_file(folder / "job.json")
    if json_file(folder / "status.json")["state"] != "awaiting_colab":
        raise ValueError("This run is not awaiting Colab results")
    if record["source_sha256"] != job["source_sha256"] or record["revision"] != job["boundary_revision"]:
        raise ValueError("Recording or end boundaries changed; prepare a new Colab run")
    info = json_file(folder / "colab-preparation.json")
    if digest(folder / "job.json") != info["job_sha256"]:
        raise ValueError("Local tracking job changed; prepare a new run")
    if Path(archive).stat().st_size > MAX_UPLOAD:
        raise ValueError("Result ZIP exceeds the 2 GiB upload limit")
    with tempfile.TemporaryDirectory(prefix="import-", dir=folder) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            members = package.infolist()
            names = [item.filename for item in members]
            if (
                len(set(names)) != len(names)
                or len(names) > info["frames"] + 20
                or sum(item.file_size for item in members) > MAX_EXPANDED
            ):
                raise ValueError("Invalid or oversized Colab result ZIP")
            if any(name != "result.json" and not allowed_result(name) for name in names):
                raise ValueError("Unexpected file in Colab results")
            if any(item.flag_bits & 1 for item in members):
                raise ValueError("Encrypted result ZIPs are not supported")
            for name, limit in (
                ("result.json", 16 * 1024**2),
                ("manifest.json", 4 * 1024**2),
                ("status.json", 1024**2),
            ):
                if package.getinfo(name).file_size > limit:
                    raise ValueError("Colab result metadata is too large")
            envelope = json.loads(package.read("result.json"))
            expected = set(envelope["files"]) | {"result.json"}
            if (
                set(names) != expected
                or envelope.get("schema") != SCHEMA
                or envelope.get("run_id") != folder.name
                or envelope.get("job_sha256") != info["job_sha256"]
            ):
                raise ValueError("Results belong to a different tracking run")
            if (
                not RESULT_FILES - {"trajectory.json", "trajectory-summary.json", "geometry-report.json"}
                <= expected
            ):
                raise ValueError("Colab results are incomplete")
            for item in members:
                target = staging / item.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if item.filename != "result.json" and digest(target) != envelope["files"][item.filename]:
                    raise ValueError("Colab result checksum mismatch")
        manifest, status = json_file(staging / "manifest.json"), json_file(staging / "status.json")
        for field in (
            "recording_id",
            "source_sha256",
            "media",
            "boundary_revision",
            "calibration",
            "end",
            "source_start",
            "duration",
            "fps",
            "delivery_view",
            "seeds",
            "backend",
            "execution",
        ):
            if manifest.get(field) != job.get(field):
                raise ValueError(f"Colab results changed the frozen job: {field}")
        size = "large" if "large" in job["backend"] else "base_plus"
        if (
            status.get("state") != "complete"
            or manifest.get("frames") != info["frames"]
            or manifest.get("weights_sha256") != CHECKPOINTS[size][2]
            or manifest.get("vendor_revision") != SAM_REVISION
            or manifest.get("source_media_sha256") != info["source_media_sha256"]
        ):
            raise ValueError("Colab run is incomplete or used a different model/video")
        count = 0
        with (staging / "observations.jsonl").open() as observations:
            for count, line in enumerate(observations, 1):
                row = json.loads(line)
                if row.get("frame") != count - 1 or not math.isclose(
                    row["source_seconds"], job["source_start"] + (count - 1) / job["fps"], abs_tol=1e-6
                ):
                    raise ValueError("Colab results changed source timestamps")
                mask = row.get("masks")
                if mask is not None and (
                    not allowed_result(mask) or not mask.startswith("masks/") or mask not in envelope["files"]
                ):
                    raise ValueError("Missing or invalid frame mask")
        if count != info["frames"]:
            raise ValueError("Colab observations are incomplete")
        if not isinstance(json_file(staging / "events.json"), list):
            raise ValueError("Invalid Colab event data")
        # Retain local source provenance and pending job; never import a remote job.json.
        manifest["source_path"] = job["source_path"]
        manifest.pop("background_path", None)
        manifest.update(colab_imported_at=time.time(), colab_bundle_sha256=info["bundle_sha256"])
        atomic(staging / "manifest.json", manifest)
        for name in envelope["files"]:
            if name == "status.json":
                continue
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / name), target)
        atomic(folder / "colab-result.json", envelope)
        status.pop("pid", None)
        status.update(state="complete", stage="Colab tracking ready — review identities", progress=1)
        atomic(folder / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    try:
        prepare(args.job)
    except Exception as exc:
        (args.job.parent / "error.log").write_text(traceback.format_exc())
        atomic(
            args.job.parent / "status.json",
            dict(state="failed", stage="Colab preparation failed", error=str(exc), progress=0),
        )
        raise


if __name__ == "__main__":
    main()
