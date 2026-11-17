from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .audio import managed_child
from .features import dataset
from .jobs import Conflict
from .models import model_version
from .worker import JobError


def trash_directory(jobs):
    return managed_child(managed_child(jobs.root, "state"), "deleted")


@contextmanager
def discard(jobs):
    """Move managed folders out of their visible namespace before committing changes."""
    moved = []
    with jobs.connect(write=True) as db:
        def move(path):
            try:
                path = managed_child(path.parent, path.name)
                if path.exists():
                    trash = trash_directory(jobs)
                    trash.mkdir(exist_ok=True)
                    target = managed_child(trash, uuid4().hex)
                    path.replace(target)
                    moved.append((path, target))
            except (OSError, ValueError):
                raise Conflict("产物目录正在使用或无法访问，请关闭播放或占用文件的程序后重试") from None

        try:
            yield db, move
            db.commit()
        except BaseException:
            for path, target in reversed(moved):
                target.replace(path)
            raise


def cleanup_deleted(jobs):
    trash = trash_directory(jobs)
    if not trash.is_dir() or not any(trash.iterdir()):
        return
    with jobs.connect(write=True):
        paths = [path.name for path in trash.iterdir() if re.fullmatch(r"[a-f0-9]{32}", path.name)]
    for name in paths:
        try:
            shutil.rmtree(managed_child(trash, name))
        except (OSError, ValueError):
            pass  # A Windows audio stream can still hold a handle; retry later.


def active_payloads(db):
    for row in db.execute("SELECT kind,payload FROM jobs WHERE state IN ('QUEUED','RUNNING')"):
        yield row["kind"], json.loads(row["payload"])


def delete_dataset(jobs, dataset_id, version_id):
    spec = {"dataset_id": dataset_id, "version_id": version_id}
    with discard(jobs) as (db, move):
        path, _ = dataset(jobs.root, spec)
        for kind, payload in active_payloads(db):
            selection = payload.get("config", payload)
            if ((kind in {"train", "features"} and all(selection.get(k) == v for k, v in spec.items()))
                    or (kind == "dataset_append" and payload["dataset"]["dataset_id"] == dataset_id)):
                raise Conflict("此数据集正被排队或运行中的任务使用")
        move(path)
    return {"deleted": True}


def delete_model(jobs, model_id, version_id):
    def matches(spec):
        return spec and spec.get("model_id") == model_id and spec.get("version_id") == version_id

    with discard(jobs) as (db, move):
        model = model_version(jobs.root, model_id, version_id)
        if model.get("source") == "external":
            raise JobError("只能删除项目训练并发布的模型")
        for kind, payload in active_payloads(db):
            if ((kind in {"model_check", "test"} and matches(payload))
                    or (kind in {"cover", "dataset_append"} and matches(payload.get("model")))
                    or (kind == "train" and matches(payload.get("initial_model")))
                    or (kind == "export" and payload.get("version_id") == version_id
                        and payload.get("summary", {}).get("dataset_id") == model_id)):
                raise Conflict("此模型正被排队或运行中的任务使用")
        selection = db.execute("SELECT j.payload FROM model_selection s JOIN jobs j ON j.id=s.job_id").fetchone()
        if selection and matches(json.loads(selection["payload"])):
            db.execute("DELETE FROM model_selection")
        path = jobs.root
        for name in ("models", "characters", model_id, version_id):
            path = managed_child(path, name)
        move(path)
    return {"deleted": True}


def delete_saved(jobs, ids):
    count = 0
    with discard(jobs) as (db, move):
        root = managed_child(jobs.root, "saved_covers")
        for item in dict.fromkeys(ids):
            if db.execute("DELETE FROM saved_covers WHERE id=?", (item,)).rowcount:
                move(managed_child(root, item))
                count += 1
    return {"deleted": count}


def saved_entries(jobs):
    with jobs.connect() as db:
        for row in db.execute("""SELECT s.id, json_extract(metadata,'$.title') AS title,
            json_extract(metadata,'$.model') AS model, json_extract(metadata,'$.model_role') AS model_role,
            COALESCE(json_extract(metadata,'$.song_title'),json_extract(metadata,'$.song.title')) AS song_title,
            json_extract(j.payload,'$.local_source') AS local_source,
            json_extract(metadata,'$.stems') AS stems,
            COALESCE(json_extract(metadata,'$.format'),'wav') AS format
            FROM saved_covers s LEFT JOIN jobs j ON j.id=json_extract(metadata,'$.job_id') ORDER BY s.rowid DESC"""):
            yield {"artifact_id": row["id"], "title": row["title"], "model": json.loads(row["model"] or "{}"),
                   "model_role": row["model_role"], "stems": bool(row["stems"]), "format": row["format"],
                   "song_title": row["song_title"] or (Path(row["local_source"]).stem if row["local_source"] else None)}
