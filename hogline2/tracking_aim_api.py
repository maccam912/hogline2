"""Read local heading evidence and persist source-bound manual broom targets."""

import hashlib
import json
from collections import OrderedDict

from fastapi import HTTPException
from pydantic import Field

from .models import Model
from .tracking_aim import METHOD, build_report, compare_target


class BroomTarget(Model):
    view_id: str
    source_seconds: float = Field(ge=0)
    point: tuple[float, float]


class AimReview(Model):
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: int = Field(ge=0)
    target: BroomTarget | None = None


def mount_aim(app, store, run_root):
    cache = OrderedDict()

    def report(identifier, run_id):
        folder = run_root(identifier, run_id)
        paths = [folder / n for n in ("manifest.json", "observations.jsonl", "events.json")]
        if not all(p.is_file() for p in paths):
            raise HTTPException(409, "Tracking evidence is not ready")
        status = folder / "status.json"
        if status.is_file() and json.loads(status.read_text()).get("state") != "complete":
            raise HTTPException(409, "Choose a completed tracking run")
        key = (str(folder), tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
        with app.state.jobs_lock:
            if key not in cache:
                contents = [p.read_bytes() for p in paths]
                manifest = json.loads(contents[0])
                if manifest["source_sha256"] != store.get(identifier)["source_sha256"]:
                    raise HTTPException(409, "Tracking belongs to a different source recording")
                digest = hashlib.sha256(METHOD.encode() + b"\0".join(contents)).hexdigest()
                value = build_report(
                    manifest,
                    (json.loads(s) for s in contents[1].splitlines() if s),
                    json.loads(contents[2]),
                    digest,
                )
                cache[key] = value
                while len(cache) > 4:
                    cache.popitem(last=False)
            if cache[key]["source_sha256"] != store.get(identifier)["source_sha256"]:
                raise HTTPException(409, "Tracking belongs to a different source recording")
            return cache[key]

    def reviews(folder):
        path = folder / "aim-reviews.json"
        return json.loads(path.read_text()) if path.exists() else {}

    prefix = "/api/recordings/{identifier}/tracking/runs/{run_id}/aim"

    @app.get(prefix)
    def aim(identifier: str, run_id: str):
        data = report(identifier, run_id)
        saved = reviews(run_root(identifier, run_id))
        rays = []
        for ray in data["rays"]:
            review = saved.get(ray["stone_id"])
            stale = bool(review and review["evidence_sha256"] != ray.get("evidence_sha256"))
            rays.append(
                {
                    **ray,
                    "review": None if stale else review,
                    "review_stale": stale,
                    "review_revision": (review or {}).get("revision", 0),
                }
            )
        return {**data, "rays": rays}

    @app.put(prefix + "/{stone_id}")
    def save(identifier: str, run_id: str, stone_id: str, request: AimReview):
        data = report(identifier, run_id)
        ray = next((r for r in data["rays"] if r["stone_id"] == stone_id), None)
        if not ray:
            raise HTTPException(404, "Stone not found")
        if ray.get("evidence_sha256") != request.evidence_sha256:
            raise HTTPException(409, "Aim evidence changed; reload before marking the broom")
        value = request.model_dump(mode="json")
        if request.target:
            target = request.target
            if target.view_id not in data["receiving_views"]:
                raise HTTPException(400, "Choose an overhead camera at the receiving house")
            view = next(v for v in data["views"] if v["id"] == target.view_id)
            x0, y0, x1, y1 = view["crop"]
            if not x0 <= target.point[0] < x1 or not y0 <= target.point[1] < y1:
                raise HTTPException(400, "Broom point lies outside the selected camera")
            if not (
                max(0, ray["reference_seconds"] - 30) <= target.source_seconds <= ray["reference_seconds"] + 2
            ):
                raise HTTPException(400, "Choose a broom frame within 30 seconds before the crossing")
            if target.source_seconds >= store.get(identifier)["media"]["duration"]:
                raise HTTPException(400, "Broom frame lies outside the recording")
            if ray["status"] == "estimated":
                value["comparison"] = compare_target(ray, view, target.point)
        with app.state.jobs_lock:
            folder = run_root(identifier, run_id)
            saved = reviews(folder)
            previous = saved.get(stone_id, {})
            if previous.get("revision", 0) != request.revision:
                raise HTTPException(409, "Broom review changed in another tab; reload before saving")
            value["revision"] += 1
            saved[stone_id] = value
            path = folder / "aim-reviews.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(saved, indent=2))
            temp.replace(path)
        return value
