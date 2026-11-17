import sys
import time
from pathlib import Path
import traceback

from .jobs import Jobs
from .supervisor import exclusive
from .errors import Cancelled, JobError


def check_cancel(jobs, job_id):
    if jobs.cancelled(job_id):
        raise Cancelled


def run_task(jobs, job):
    if job["kind"] == "diagnostic":
        for item in range(1, 5):
            check_cancel(jobs, job["id"])
            time.sleep(0.35)
            jobs.progress(job["id"], {"completed": item, "total": 4, "message": "任务系统诊断"})
        return {"diagnostic": True}
    from .settings import Settings

    settings = Settings.from_environment()
    kind = job["kind"]
    payload = job["payload"]
    if kind in {"dataset_scan", "dataset_prepare"}:
        from .datasets import process

        return process(jobs, job)
    if kind == "asset_base":
        from dataclasses import replace
        from .environment import fetch_base

        try:
            path = fetch_base(
                replace(settings, root=jobs.root), source=Path(payload["source"]) if payload["source"] else None,
                check=lambda: check_cancel(jobs, job["id"]), progress=lambda data: jobs.progress(job["id"], data),
            )
        except ValueError as exc:
            raise JobError(str(exc)) from None
        return {"path": str(path)}
    if kind == "cover":
        from .covers import generate

        return generate(jobs, job, settings)
    if kind == "remix":
        from .covers import remix

        return remix(jobs, job)
    from .engine import invoke

    actions = {"features": "features", "dataset_append": "append_range", "train": "train", "model_import": "inspect",
               "model_check": "check_model", "export": "export", "test": "infer"}
    if kind in actions:
        if kind == "test":
            from .audio import RATE, atomic_json, ffmpeg
            from .models import record_smoke

            area = jobs.root / "runtime" / "tests" / job["id"]
            area.mkdir(parents=True)
            source = area / "input.wav"
            ffmpeg("-ss", payload["start"], "-i", payload["input"], "-t", payload["duration"],
                   "-vn", "-ac", 1, "-ar", RATE, "-c:a", "pcm_f32le", source)
            result = invoke(jobs, job, settings, "rift_worker", "infer",
                            {**payload, "input": str(source), "output": str(area / "converted.wav")})
            atomic_json(area / "manifest.json", result)
            record_smoke(jobs, job["id"], payload, result)
            return result
        return invoke(jobs, job, settings, "rift_worker", actions[kind], payload)
    raise JobError("该任务类型尚未接通")


def main():
    jobs = Jobs(Path(sys.argv[1]))
    job_id, instance = sys.argv[2:4]
    for _ in range(100):
        try:
            with exclusive(jobs.state / "worker.lock"):
                job = jobs.get(job_id)
                if job["state"] != "RUNNING" or job["instance"] != instance:
                    return
                try:
                    result = run_task(jobs, job)
                    jobs.status(job_id, result.get("outcome", "COMPLETED"), result=result, instance=instance)
                except Cancelled:
                    jobs.status(job_id, "CANCELLED", instance=instance)
                except JobError as exc:
                    jobs.status(job_id, "FAILED", error=str(exc), instance=instance)
                except Exception as exc:
                    frames = traceback.extract_tb(exc.__traceback__)
                    place = frames[-1] if frames else None
                    error = f"{type(exc).__name__} ({Path(place.filename).name}:{place.lineno})" if place else type(exc).__name__
                    jobs.status(job_id, "FAILED", error=error, instance=instance)
                return
        except OSError:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
