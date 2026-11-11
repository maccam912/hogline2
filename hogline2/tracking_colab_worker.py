"""Entry point shipped in a Colab bundle; runs the actual Hogline tracking pipeline."""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from .tracking_colab import SCHEMA, allowed_result, digest, json_file
from .tracking_jobs import atomic


def execute(workspace):
    bundle = json_file(workspace / "bundle.json")
    if bundle["schema"] != SCHEMA:
        raise ValueError("Unsupported bundle version")
    for name, expected in bundle["files"].items():
        path = workspace / name
        if not path.resolve().is_relative_to(workspace.resolve()) or digest(path) != expected:
            raise ValueError("Tracking bundle checksum mismatch: " + name)
    run_id = bundle["run_id"]
    job = json_file(workspace / "job.json")
    folder = workspace / "recordings" / job["recording_id"] / "tracking" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    job.update(source_path=str(workspace / "media.mp4"), background_path=str(workspace / "background.jpg"))
    atomic(folder / "job.json", job)
    environment = {**os.environ, "PYTHONPATH": str(workspace)}
    subprocess.run(
        [sys.executable, "-u", "-m", "hogline2.tracking_runner", str(folder / "job.json")],
        cwd=workspace,
        env=environment,
        check=True,
    )
    if json_file(folder / "status.json")["state"] != "complete":
        raise ValueError("Tracking did not complete")
    paths = [p for p in folder.rglob("*") if p.is_file() and allowed_result(p.relative_to(folder).as_posix())]
    result = dict(
        schema=SCHEMA,
        run_id=run_id,
        job_sha256=bundle["job_sha256"],
        files={p.relative_to(folder).as_posix(): digest(p) for p in paths},
    )
    with zipfile.ZipFile(workspace / "colab-results.zip", "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr("result.json", json.dumps(result, indent=2))
        for path in paths:
            output.write(path, path.relative_to(folder).as_posix())
    print("Tracking complete. Download colab-results.zip and import it into Hogline.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    execute(parser.parse_args().workspace.resolve())
