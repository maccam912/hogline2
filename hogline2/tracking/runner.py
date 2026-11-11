"""The tracking process: `python -m hogline2.tracking.runner <job folder>/job.json`.

One process tracks the requested ends of one recording, one after another, with stonetrack. It
is separate from the web server so a tracking run survives a server restart and can be stopped
by signal. Progress goes to `status.json` beside the job; every end gets a run folder
`recordings/<id>/tracking/<run id>/` holding `job.json`, `status.json` and
`stonetrack-result.json`.

stonetrack keeps its own per-recording work (background, paint calibration, detections) under
`data/stonetrack/<id>/` and reuses it, so tracking an end again is quick.
"""

import argparse
import os
import signal
import sys
import time
import traceback
import uuid
from pathlib import Path

from ..files import read_json, write_json

FPS = 20.0  # stonetrack's tracking rules are written for this rate
DETECT_SLICE = 120.0  # seconds of video per detection call, so progress moves


class Stopped(Exception):
    pass


def prepare(work, source, layout, say):
    """Background, paint calibration and stone alignment for this recording (cached)."""
    from stonetrack import cli, frames

    work.mkdir(parents=True, exist_ok=True)
    previous = read_json(work / "run.json") or {}
    if previous.get("layout", {}).get("views") != layout["views"] or previous.get("source") != source:
        write_json(work / "run.json", dict(source=source, layout=layout, video=frames.probe(source)))
        for name in ("calibration.json", "stationary.jsonl"):
            (work / name).unlink(missing_ok=True)
    if not (work / "calibration.json").exists():
        say("Calibrating cameras from the paint (first time for this recording)", 0.02)
        cli.cmd_calibrate(argparse.Namespace(run=str(work), overhead_height=4.0))
    if "stone_alignment" not in read_json(work / "calibration.json"):
        say("Aligning cameras on stones at rest (first time for this recording)", 0.06)
        cli.cmd_refine(
            argparse.Namespace(run=str(work), frames=None, start=60, end=None, every=20, budget=1e9)
        )


def detect(work, start, end, say):
    from stonetrack import cli

    workers = max(2, min(8, (os.cpu_count() or 4) // 2))
    t = start
    while t < end:
        upto = min(end, t + DETECT_SLICE)
        cli.cmd_detect(
            argparse.Namespace(
                run=str(work), start=t, end=upto, ends=None, fps=FPS, workers=workers, budget=1e9, redo=False
            )
        )
        t = upto
        say("Finding stones", 0.1 + 0.8 * (t - start) / (end - start))


def track_end(work, start, end, delivery_end, say):
    """One end's stonetrack result: named stones with paths, split times and events."""
    from stonetrack.geometry import load_calibration
    from stonetrack.track import check_settled, track_range

    detect(work, start, end, say)
    say("Following stones", 0.92)
    cameras, _ = load_calibration(str(work / "calibration.json"))
    result, rows = track_range(str(work), cameras, start, end, FPS, delivery_end)
    if not rows:
        raise ValueError("No video frames were found in this end")
    result["check"] = check_settled(result, rows, cameras)
    return result


class Runner:
    def __init__(self, job_path):
        self.folder = Path(job_path).parent
        self.job = read_json(job_path)
        self.status = read_json(self.folder / "status.json") or {}
        self.data = Path(self.job["data_root"])
        self.tracking = self.data / "recordings" / self.job["recording_id"] / "tracking"
        self.work = self.data / "stonetrack" / self.job["recording_id"]

    def publish(self, **changes):
        self.status.update(changes, updated_at=time.time())
        write_json(self.folder / "status.json", self.status)

    def entry_progress(self, entry, stage, progress):
        entry.update(stage=stage, progress=round(float(progress), 3))
        done = sum(e["progress"] for e in self.status["ends"]) / len(self.status["ends"])
        self.publish(stage=f"End {entry['end_number']}: {stage}", progress=round(done, 3))

    def run_end(self, entry):
        end = entry["end"]
        run_id = uuid.uuid4().hex
        folder = self.tracking / run_id
        folder.mkdir(parents=True)
        started = time.time()
        write_json(
            folder / "job.json",
            dict(
                backend="stonetrack-0.1",
                recording_id=self.job["recording_id"],
                source_sha256=self.job["source_sha256"],
                boundary_revision=self.job["boundary_revision"],
                end=end,
                source_start=end["start_seconds"],
                duration=end["end_seconds"] - end["start_seconds"],
                fps=FPS,
                delivery_end=entry["delivery_end"],
                job_id=self.folder.name,
            ),
            indent=1,
        )

        def say(stage, progress, state="running", **extra):
            write_json(
                folder / "status.json",
                dict(state=state, stage=stage, progress=progress, started_at=started, **extra),
            )
            self.entry_progress(entry, stage, progress)

        entry.update(run_id=run_id, state="running", error=None)
        say("Starting", 0)
        try:
            prepare(self.work, self.job["source_path"], self.job["layout"], say)
            result = track_end(
                self.work, end["start_seconds"], end["end_seconds"], entry["delivery_end"], say
            )
        except Stopped:
            say("Stopped", 0, state="stopped")
            entry.update(state="stopped")
            raise
        except Exception as exc:
            traceback.print_exc()
            message = str(exc)[:300] or type(exc).__name__
            say("Failed", 0, state="failed", error=message)
            entry.update(state="failed", error=message)
            return
        write_json(folder / "stonetrack-result.json", result)
        entry.update(state="complete", stones=len(result["stones"]))
        say(
            "Tracked",
            1,
            state="complete",
            stones=len(result["stones"]),
            elapsed_seconds=time.time() - started,
        )

    def run(self):
        self.publish(state="running", stage="Starting", pid=os.getpid(), started_at=time.time(), error=None)
        try:
            for entry in self.status["ends"]:
                if entry["state"] != "complete":
                    self.run_end(entry)
            failed = [e["end_number"] for e in self.status["ends"] if e["state"] == "failed"]
            self.publish(
                state="failed" if failed else "complete",
                stage=f"Could not track end {', '.join(map(str, failed))}" if failed else "Tracking finished",
                progress=1,
            )
        except Stopped:
            for entry in self.status["ends"]:
                if entry["state"] in {"queued", "running"}:
                    entry["state"] = "stopped"
            self.publish(state="stopped", stage="Stopped")
        except Exception as exc:
            traceback.print_exc()
            self.publish(state="failed", stage="Tracking failed", error=str(exc)[:300] or type(exc).__name__)


def main():
    runner = Runner(sys.argv[1])

    def stop(*_):
        raise Stopped()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runner.run()


if __name__ == "__main__":
    main()
