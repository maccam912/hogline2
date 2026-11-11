"""Durable whole-game coordinator. Two isolated SAM workers, one physical ID set per end."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
import traceback
import uuid
from pathlib import Path

from .config import PROJECT, Config
from .media import export_ends, fingerprint
from .pipeline import Worker
from .store import Store
from .tracking_geometry import validate_calibration
from .tracking_jobs import atomic, full_end_job


class Cancelled(Exception):
    pass


class GameRunner:
    def __init__(self, path, popen=subprocess.Popen):
        self.path = Path(path)
        self.root = self.path.parent
        self.job = json.loads(self.path.read_text())
        self.config = Config.load(self.job["data_root"])
        self.store = Store(self.config.data)
        self.popen = popen
        self.children = {}
        self.cancelled = False
        self.status = json.loads((self.root / "status.json").read_text())

    def publish(self, **changes):
        self.status.update(changes)
        self.status["updated_at"] = time.time()
        atomic(self.root / "status.json", self.status)

    def stop(self, *_):
        self.cancelled = True

    def check(self):
        if self.cancelled or (self.root / "stop").exists():
            raise Cancelled()

    def prepare(self):
        identifier = self.job["recording_id"]
        record = self.store.get(identifier)
        self.check()
        self.publish(stage="Preparing trimmed recording and end clips", progress=0.02)
        current_clips = record.get("clips", [])
        valid = (
            bool(record.get("ends"))
            and len(current_clips) == len(record["ends"])
            and all(
                c["revision"] == record["revision"]
                and (self.config.data / "recordings" / identifier / c["filename"]).is_file()
                for c in current_clips
            )
        )
        if not record.get("source_path") or not Path(record["source_path"]).is_file() or not valid:
            Worker(self.config, self.store).run(identifier, "export" if record.get("ends") else "analyze")
            record = self.store.get(identifier)
        elif fingerprint(record["source_path"]) != record["source_sha256"]:
            raise ValueError("Source changed; re-import and recalibrate this recording")
        self.check()
        # Freeze the exact boundaries and camera setup for this campaign.
        if (
            self.job.get("boundary_revision") is not None
            and self.job["boundary_revision"] != record["revision"]
        ):
            raise ValueError("Boundaries changed; start a new whole-game job")
        calibration = validate_calibration(self.job["calibration"], record)
        self.job.update(boundary_revision=record["revision"], source_sha256=record["source_sha256"])
        atomic(self.path, self.job)
        game_clip = record.get("game_clip")
        if (
            not game_clip
            or game_clip["revision"] != record["revision"]
            or not (self.config.data / "recordings" / identifier / game_clip["filename"]).is_file()
        ):
            trimmed = {
                **record,
                "ends": [
                    dict(
                        number=1,
                        start_seconds=record["ends"][0]["start_seconds"],
                        end_seconds=record["ends"][-1]["end_seconds"],
                    )
                ],
            }

            def progress(stage, amount):
                self.check()
                self.publish(stage="Trimming game: " + stage, progress=0.1 * amount)

            game_clip = export_ends(self.config, trimmed, progress, filename_prefix="game")[0]
            record = self.store.update(identifier, game_clip=game_clip)
        self.check()
        existing = {e["end_number"]: e for e in self.status.get("ends", [])}
        ends = []
        tracking = self.config.data / "recordings" / identifier / "tracking"
        for end in record["ends"]:
            entry = existing.get(end["number"])
            if entry:
                previous = tracking / entry["run_id"] / "status.json"
                if previous.is_file() and json.loads(previous.read_text()).get("state") == "complete":
                    entry.update(state="complete", progress=1)
                    ends.append(entry)
                    continue
            run_id = uuid.uuid4().hex
            folder = tracking / run_id
            folder.mkdir(parents=True)
            task = full_end_job(record, calibration, end, self.job["fps"], self.root.name)
            atomic(folder / "job.json", task)
            atomic(
                folder / "status.json",
                dict(state="queued", stage="Queued for whole-game tracking", progress=0),
            )
            ends.append(
                dict(
                    end_number=end["number"],
                    run_id=run_id,
                    state="queued",
                    progress=0,
                    attempts=(entry.get("attempts", []) + [entry["run_id"]]) if entry else [],
                )
            )
        self.publish(ends=ends, stage="Tracking full ends", progress=0.1)

    def launch(self, entry):
        folder = self.config.data / "recordings" / self.job["recording_id"] / "tracking" / entry["run_id"]
        with (folder / "run.log").open("wb") as log:
            process = self.popen(
                [
                    str(PROJECT / ".venv-sam/bin/python"),
                    "-m",
                    "hogline2.tracking_runner",
                    str(folder / "job.json"),
                ],
                cwd=PROJECT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.children[entry["run_id"]] = process
        entry.update(state="running", pid=process.pid)

    def tick(self):
        self.check()
        tracking = self.config.data / "recordings" / self.job["recording_id"] / "tracking"
        for entry in self.status["ends"]:
            process = self.children.get(entry["run_id"])
            if process is None:
                continue
            path = tracking / entry["run_id"] / "status.json"
            value = json.loads(path.read_text())
            entry.update(
                progress=value.get("progress", 0), stage=value.get("stage"), stones=value.get("stones")
            )
            code = process.poll()
            if code is not None:
                self.children.pop(entry["run_id"])
                success = code == 0 and value.get("state") == "complete"
                entry.update(
                    state="complete" if success else "failed",
                    error=None
                    if success
                    else value.get("error") or f"Worker exited without completion (status {code})",
                )
        for entry in self.status["ends"]:
            if len(self.children) >= self.job["concurrency"]:
                break
            if entry["state"] == "queued":
                self.launch(entry)
        complete = sum(e["state"] == "complete" for e in self.status["ends"])
        progress = 0.1 + 0.9 * sum(e.get("progress", 0) for e in self.status["ends"]) / max(
            1, len(self.status["ends"])
        )
        self.publish(
            stage=f"Tracking full ends: {complete}/{len(self.status['ends'])} finished; "
            f"{len(self.children)} running",
            progress=progress,
        )
        return bool(self.children or any(e["state"] == "queued" for e in self.status["ends"]))

    def run(self):
        self.publish(
            state="running",
            pid=os.getpid(),
            started_at=self.status.get("started_at", time.time()),
            error=None,
        )
        try:
            self.prepare()
            while self.tick():
                time.sleep(1)
            failed = any(e["state"] == "failed" for e in self.status["ends"])
            self.publish(
                state="failed" if failed else "complete",
                progress=1,
                stage="Some ends failed — resume to retry"
                if failed
                else "Whole game processed — review stone identities",
            )
        except Cancelled:
            self.publish(state="stopped", stage="Stopped — resume keeps completed ends")
        except Exception as exc:
            (self.root / "error.log").write_text(traceback.format_exc())
            self.publish(
                state="stopped" if self.cancelled else "failed",
                stage="Stopped — resume keeps completed ends" if self.cancelled else "Needs attention",
                error=str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
            )
        finally:
            for run_id, process in self.children.items():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                folder = self.config.data / "recordings" / self.job["recording_id"] / "tracking" / run_id
                value = json.loads((folder / "status.json").read_text())
                if value.get("state") != "complete":
                    atomic(
                        folder / "status.json",
                        {**value, "state": "stopped", "stage": "Stopped by whole-game job"},
                    )
            for entry in self.status.get("ends", []):
                if entry["state"] in {"running", "queued"}:
                    entry["state"] = "stopped"
            self.publish(ends=self.status.get("ends", []))
            record = self.store.get(self.job["recording_id"])
            if record["state"] in {"running", "queued"}:
                self.store.update(
                    record["id"],
                    state="interrupted",
                    stage="Game preparation interrupted — resume whole game",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    runner = GameRunner(args.job)
    signal.signal(signal.SIGTERM, runner.stop)
    signal.signal(signal.SIGINT, runner.stop)
    # One campaign owns the catalog queue; its children share two GPU slots.
    with (runner.config.data / "game-pipeline.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            runner.publish(state="failed", stage="Another whole-game coordinator is active")
            return
        runner.run()


if __name__ == "__main__":
    main()
