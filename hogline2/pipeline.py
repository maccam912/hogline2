import fcntl
import queue
import threading
from pathlib import Path

from .detector import EndDetector
from .media import download, export_ends, fingerprint, probe
from .models import validate_ends
from .tracking_jobs import active_game

BUSY = {"queued", "running"}


class Worker:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.queue = queue.Queue()
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.lease = None
        self.thread = None

    def start(self):
        self.lease = (self.config.data / "worker.lock").open("a")
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lease.close()
            raise RuntimeError("Another Hogline 2 worker already uses this data directory") from None
        for record in self.store.all():
            if record["state"] in BUSY and not active_game(self.config.data, record["id"]):
                self.store.update(
                    record["id"], state="interrupted", stage="Interrupted — Retry to resume cached work"
                )
        self.thread = threading.Thread(target=self.loop, daemon=True, name="hogline2-worker")
        self.thread.start()

    def stop(self):
        self.closed.set()
        self.queue.put(None)
        # Active work finishes or the process exits; keep the process lease until then.
        if self.thread:
            self.thread.join(timeout=2)
        if self.lease and (not self.thread or not self.thread.is_alive()):
            self.lease.close()

    def submit(self, identifier, action="analyze"):
        with self.lock:
            record = self.store.get(identifier)
            if active_game(self.config.data, identifier):
                raise ValueError("Stop the whole-game job before changing recording preparation")
            if record["state"] in BUSY:
                raise ValueError("This recording already has a running or queued job")
            self.store.update(
                identifier, state="queued", stage="Queued", action=action, error=None, progress=0
            )
            self.queue.put((identifier, action))

    def loop(self):
        while not self.closed.is_set():
            task = self.queue.get()
            if task is None:
                break
            identifier, action = task
            try:
                self.run(identifier, action)
            except Exception as exc:
                # Only known validation messages are public; never leak raw provider/subprocess details.
                message = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    else f"{type(exc).__name__}; see local troubleshooting"
                )
                self.store.update(identifier, state="failed", stage="Needs attention", error=message)

    def run(self, identifier, action):
        def progress(stage, amount):
            self.store.update(identifier, state="running", stage=stage, progress=amount)

        progress("Starting", 0)
        record = self.store.get(identifier)
        if not record.get("source_path") or not Path(record["source_path"]).is_file():
            if not record.get("url"):
                raise ValueError("Local source is unavailable")
            fields = download(self.config, identifier, progress)
            record = self.store.update(identifier, **fields)
        else:
            progress("Verifying local recording", 0.1)
            if fingerprint(record["source_path"]) != record["source_sha256"]:
                raise ValueError("Source checksum changed; import this recording again")
            record = self.store.update(identifier, media=probe(record["source_path"]))
        if action == "analyze":
            ends, warnings = EndDetector(self.config).detect(record, progress)
            if record.get("expected_ends") and len(ends) != record["expected_ends"]:
                warnings.append(
                    f"Expected {record['expected_ends']} ends, detected {len(ends)}. Review the partition."
                )
            record = self.store.update(
                identifier,
                ends=ends,
                warnings=warnings,
                revision=record["revision"] + 1,
                boundary_origin="automatic_proposal",
                clips=[],
            )
        validate_ends(record["ends"], record["media"]["duration"])
        clips = export_ends(self.config, record, progress)
        self.store.update(
            identifier,
            clips=clips,
            state="ready",
            stage="Clips ready — review boundaries",
            progress=1,
            error=None,
            segmentation="deferred",
        )
