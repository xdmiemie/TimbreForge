import json
from pathlib import Path
import time
from typing import Literal
from uuid import UUID, uuid4

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .audio import RATE, digest, managed_child, probe, stage_file
from .augmentation import available_dataset, model_dataset
from .covers import ORIGINAL_TRACK, SAVED_FORMATS, STEM_FILES, accept_cover, cover_result, cover_stem, current, save_cover
from .datasets import manifests
from .environment import assets, doctor, gpu_snapshot, lock
from .features import dataset, train_preflight
from .folder_picker import pick_folder
from .inference import InferenceConfig
from .jobs import Conflict
from .library import delete_dataset, delete_model, delete_saved, saved_entries
from .models import model_version, list_models
from .music import bridge
from .rift_training import checkpoint_index
from .training_config import TrainConfig, TrainSubmission
from .worker import JobError


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: UUID


class PathBody(RequestBody):
    path: str = Field(min_length=1, max_length=2048)


class SourceBody(RequestBody):
    source: str | None = Field(None, max_length=2048)


class DatasetBody(RequestBody):
    source: str = Field(min_length=1, max_length=2048)
    role: str = Field(min_length=1, max_length=100)
    language: str = Field("zh", min_length=1, max_length=30)
    recursive: bool = True
    excluded_paths: list[str] = Field(default_factory=list, max_length=10000)


class VersionBody(RequestBody):
    dataset_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    version_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class DeriveBody(RequestBody):
    file: str = Field(pattern=r"^[0-9]{9}-[a-f0-9]{8}\.ckpt$")


class ModelBody(RequestBody):
    model_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    version_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    speaker: str = Field(min_length=1, max_length=500)


class TestBody(ModelBody):
    path: str = Field(min_length=1, max_length=2048)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    start_seconds: float = Field(0, ge=0, le=1800)
    end_seconds: float | None = Field(None, gt=0, le=1800)


class CoverBody(RequestBody):
    local_source: str | None = Field(None, max_length=2048)
    search_id: UUID | None = None
    song_id: int | None = Field(None, ge=1, lt=10**12)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    instrumental_gain_db: float = Field(0, ge=-40, le=12)
    vocal_gain_db: float = Field(0, ge=-40, le=12)


class RemixBody(RequestBody):
    instrumental_gain_db: float = Field(0, ge=-40, le=12)
    vocal_gain_db: float = Field(0, ge=-40, le=12)


class SaveBody(RequestBody):
    title: str = Field(min_length=1, max_length=180, pattern=r"^[^\x00-\x1f\x7f]+$")
    stems: bool = False
    format: Literal["wav", "mp3"] = "wav"


class AppendBody(RequestBody):
    first_seconds: float = Field(ge=0, le=1800)
    second_seconds: float = Field(ge=0, le=1800)


class SavedDeleteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[UUID] = Field(min_length=1, max_length=10000)


def initial_model(root, cfg):
    _, metadata = dataset(root, cfg)
    initial = metadata.get("initial_model")
    if initial and not cfg.get("use_base_model"):
        try:
            model = model_version(root, initial["model_id"], initial["version_id"])
        except JobError:
            raise JobError("用于微调的来源模型已删除，请改用官方底模") from None
        if (model["architecture"] != cfg["architecture"] or len(model["speakers"]) != 1
                or model["sha256"] != initial["sha256"] or not Path(model["checkpoint"]).is_file()):
            raise JobError("来源模型不可用于此配置，请改用官方底模")
        return {**initial, "checkpoint": model["checkpoint"]}
    return None


def store(req):
    return req.app.state.jobs


def submit(req, kind, body, payload=None, **kwargs):
    snapshot = body.model_dump(mode="json", exclude={"request_id"}) if payload is None else payload
    return store(req).submit(kind, snapshot, str(body.request_id), **kwargs)


def validated_model(jobs, body):
    row = model_version(jobs.root, body.model_id, body.version_id)
    if body.speaker not in row["speakers"] or not Path(row["checkpoint"]).is_file():
        raise JobError("模型或所选说话人不可用")
    fields = ("model_id", "version_id", "checkpoint", "sha256", "architecture", "role")
    return {**{name: row[name] for name in fields}, "speaker": body.speaker}


def enabled(jobs):
    with jobs.connect() as db:
        row = db.execute("SELECT job_id FROM model_selection WHERE id=1").fetchone()
    if row and row["job_id"]:
        job = jobs.get(row["job_id"])
        return {"job": job, "model": job["payload"] if job["state"] == "COMPLETED" else None}
    return {"job": None, "model": None}


def register(app):
    settings = app.state.settings

    @app.get("/api/environment")
    def environment():
        return {"assets": assets(settings), "rift_source_present": settings.rift.is_dir()}

    @app.get("/api/paths")
    def directories(path: str = Query("", max_length=2048),
                    kind: Literal["directory", "model", "audio"] = "directory"):
        from .datasets import ACCEPTED

        directory = Path(path).expanduser().resolve() if path else Path.home()
        if directory.is_file():
            directory = directory.parent
        if not directory.is_dir():
            raise JobError("找不到可浏览的目录")
        suffixes = {".ckpt"} if kind == "model" else ACCEPTED if kind == "audio" else set()
        try:
            entries = [{"path": str(item), "name": item.name, "directory": item.is_dir()}
                       for item in directory.iterdir()
                       if item.is_dir() or item.suffix.lower() in suffixes and item.is_file()]
        except PermissionError:
            raise JobError("当前目录不可读") from None
        return {"path": str(directory), "parent": str(directory.parent),
                "entries": sorted(entries, key=lambda entry: (not entry["directory"], entry["name"].casefold()))[:300]}

    @app.post("/api/paths/pick-directory")
    def native_directory(body: SourceBody):
        return {"path": pick_folder(body.source)}

    @app.post("/api/environment/doctor")
    def diagnostics():
        return doctor(settings)

    @app.post("/api/assets/base", status_code=202)
    def asset(req: Request, body: SourceBody):
        return submit(req, "asset_base", body)

    @app.post("/api/datasets/{operation}", status_code=202)
    def prepare_dataset(req: Request, operation: Literal["scan", "prepare"], body: DatasetBody):
        if not Path(body.source).is_dir():
            raise JobError("数据源目录不存在")
        return submit(req, "dataset_" + operation, body)

    @app.get("/api/datasets")
    def dataset_list(req: Request):
        root = store(req).root
        return [{**{key: entry.get(key) for key in ("role", "language", "dataset_id", "version_id",
                                                    "stats", "created", "initial_model")},
                 "features_ready": (root / "datasets" / entry["dataset_id"] / entry["version_id"] / "features.json").is_file()}
                for entry in sorted(manifests(root), key=lambda row: row["created"], reverse=True)]

    @app.delete("/api/datasets/{dataset_id}/{version_id}")
    def remove_dataset(req: Request, version_id: UUID, dataset_id: str):
        if len(dataset_id) != 24 or any(ch not in "0123456789abcdef" for ch in dataset_id):
            raise KeyError("数据集不存在")
        return delete_dataset(store(req), dataset_id, version_id.hex)

    @app.get("/api/datasets/previews/{job_id}/{index}")
    def preview_source(req: Request, job_id: UUID, index: int):
        job = store(req).get(job_id.hex)
        if job["kind"] not in {"dataset_scan", "dataset_prepare"} or not job["result"]:
            raise KeyError("无可播放的扫描记录")
        sources = job["result"]["sources"]
        if not 0 <= index < len(sources) or "seconds" not in sources[index]:
            raise KeyError("此录音无法解码")
        root = Path(job["result"]["source"]).resolve()
        path = (root / sources[index]["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise KeyError("源录音不可用")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.post("/api/features", status_code=202)
    def features(req: Request, body: VersionBody):
        return submit(req, "features", body, on_accept=lambda *_: dataset(store(req).root, body.model_dump()))

    @app.post("/api/training/preflight")
    def preflight(req: Request, body: TrainConfig):
        cfg = body.model_dump()
        return {"config": cfg, "optimizer": "adamwsf", "crop_frames": 256, "gpu_status": gpu_snapshot(),
                "initial_model": initial_model(store(req).root, cfg),
                **train_preflight(store(req).root, cfg)}

    @app.post("/api/training", status_code=202)
    def train(req: Request, body: TrainSubmission):
        cfg = body.model_dump(exclude={"request_id"})
        result = train_preflight(store(req).root, cfg)
        if not result["feature_manifest_present"]:
            raise JobError("请先准备该数据版本的 RIFT 特征")
        base = settings.root / "assets" / lock()["assets"]["base"]["path"]
        initial = initial_model(store(req).root, cfg)
        if initial is None and not base.is_file():
            raise JobError("缺少官方 1024-16 底模")

        def accept(db, _):
            dataset(store(req).root, cfg)
            if initial:
                initial_model(store(req).root, cfg)

        return submit(req, "train", body, {"config": cfg, "fingerprint": result["dataset_fingerprint"],
                                            "feature_key": result["feature_key"],
                                            "feature_digest": result["feature_digest"], "initial_model": initial},
                      on_accept=accept)

    @app.get("/api/training/{job_id}/checkpoints")
    def checkpoints(req: Request, job_id: UUID):
        job = store(req).get(job_id.hex)
        if job["kind"] != "train":
            raise KeyError("训练任务不存在")
        return checkpoint_index(store(req).root, job["payload"].get("run_id", job["id"]))

    @app.post("/api/training/{job_id}/resume", status_code=202)
    def resume(req: Request, job_id: UUID, body: RequestBody):
        jobs = store(req)
        original = jobs.get(job_id.hex)
        if original["kind"] != "train" or original["state"] not in {"PAUSED", "INTERRUPTED", "FAILED"}:
            raise Conflict("该训练当前不能继续")
        snapshot = {key: value for key, value in original["payload"].items()
                    if key not in {"parent_run_id", "checkpoint_file"}}
        snapshot["run_id"] = original["payload"].get("run_id", original["id"])
        run = snapshot["run_id"]
        points = checkpoint_index(jobs.root, run)
        if not points:
            raise JobError("该训练没有已提交的恢复点")
        if points[-1]["reason"] == "final":
            raise Conflict("该运行已经达到累计训练目标")

        def accept(db, new_id):
            dataset(jobs.root, snapshot["config"])
            other = db.execute(
                """SELECT 1 FROM jobs WHERE id!=? AND kind='train' AND state IN ('QUEUED','RUNNING')
                   AND (id=? OR json_extract(payload,'$.run_id')=?)""", (new_id, run, run),
            ).fetchone()
            if other:
                raise Conflict("该运行已有活跃训练")
            if checkpoint_index(jobs.root, run)[-1]["reason"] == "final":
                raise Conflict("该运行已经达到累计训练目标")

        return submit(req, "train", body, snapshot, on_accept=accept)

    @app.post("/api/training/{job_id}/derive", status_code=202)
    def derive(req: Request, job_id: UUID, body: DeriveBody):
        jobs = store(req)
        original = jobs.get(job_id.hex)
        if original["kind"] != "train":
            raise KeyError("训练任务不存在")
        source = original["payload"].get("run_id", original["id"])
        points = checkpoint_index(jobs.root, source)
        point = next((row for row in points if row["file"] == body.file), None)
        if point is None:
            raise JobError("找不到该恢复点")
        snapshot = {key: original["payload"][key] for key in
                    ("config", "fingerprint", "feature_key", "feature_digest", "initial_model") if key in original["payload"]}
        target = json.loads((jobs.root / "runs" / source / "config.json").read_text(encoding="utf-8"))["target"]
        if point["step"] >= target:
            raise JobError("请选择早于训练目标的恢复点")
        return submit(req, "train", body, {**snapshot, "parent_run_id": source, "checkpoint_file": body.file},
                      on_accept=lambda *_: dataset(jobs.root, snapshot["config"]))

    @app.get("/api/training/{job_id}/exports")
    def exports(req: Request, job_id: UUID):
        job = store(req).get(job_id.hex)
        if job["kind"] != "train":
            raise KeyError("训练任务不存在")
        directory = settings.root / "runs" / job["payload"].get("run_id", job["id"])
        return [{"version_id": path.stem.removeprefix("export-"), "summary": json.loads(path.read_text(encoding="utf-8"))["summary"]}
                for path in sorted(directory.glob("export-*.json"))]

    @app.post("/api/training/{job_id}/exports/{version_id}", status_code=202)
    def retry_export(req: Request, body: RequestBody, job_id: UUID, version_id: UUID):
        job = store(req).get(job_id.hex)
        if job["kind"] != "train":
            raise KeyError("训练任务不存在")
        path = settings.root / "runs" / job["payload"].get("run_id", job["id"]) / f"export-{version_id.hex}.json"
        if not path.is_file():
            raise KeyError("导出快照不存在")
        return submit(req, "export", body, json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/models")
    def models(req: Request):
        return list_models(store(req).root)

    @app.delete("/api/models/{model_id}/{version_id}")
    def remove_model(req: Request, version_id: UUID, model_id: str):
        if len(model_id) != 24 or any(ch not in "0123456789abcdef" for ch in model_id):
            raise KeyError("模型不存在")
        return delete_model(store(req), model_id, version_id.hex)

    @app.post("/api/models/import", status_code=202)
    def import_model(req: Request, body: PathBody):
        return submit(req, "model_import", body)

    @app.get("/api/models/enabled")
    def active_model(req: Request):
        return enabled(store(req))

    @app.delete("/api/models/enabled")
    def close_model(req: Request):
        with store(req).connect(write=True) as db:
            db.execute("DELETE FROM model_selection")
        return {"enabled": False}

    @app.post("/api/models/enabled", status_code=202)
    def enable_model(req: Request, body: ModelBody):
        spec = validated_model(store(req), body)

        def accept(db, job_id):
            validated_model(store(req), body)
            db.execute("INSERT INTO model_selection(id,job_id) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET job_id=excluded.job_id",
                       (job_id,))

        return submit(req, "model_check", body, spec, on_accept=accept)

    @app.post("/api/model-tests", status_code=202)
    def model_test(req: Request, body: TestBody):
        spec = validated_model(store(req), body)
        path = Path(body.path)
        if not path.is_file():
            raise JobError("干声文件不存在")
        try:
            seconds = probe(path)["seconds"]
        except Exception:
            raise JobError("无法解码干声文件") from None
        end = seconds if body.end_seconds is None else body.end_seconds
        if end > seconds + 0.05 or not 0.5 <= end - body.start_seconds <= 60:
            raise JobError("请选择音频内 0.5 至 60 秒的试听范围")
        return submit(req, "test", body, {**spec, "input": str(path.resolve()),
                                         "start": body.start_seconds, "duration": end-body.start_seconds,
                                         "settings": body.inference.model_dump()},
                      on_accept=lambda *_: validated_model(store(req), body))

    @app.get("/api/model-tests/{job_id}/{kind}")
    def play_test(req: Request, job_id: UUID, kind: Literal["input", "converted"]):
        row = store(req).get(job_id.hex)
        if row["kind"] != "test" or row["state"] != "COMPLETED":
            raise KeyError("试听结果不可用")
        path = settings.root / "runtime" / "tests" / job_id.hex / f"{kind}.wav"
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/music/status")
    def music_status(verify: bool = False):
        return bridge(settings, "account" if verify else "status")

    @app.get("/api/music/search")
    def search(req: Request, query: str = Query(min_length=1, max_length=120), page: int = Query(1, ge=1, le=5)):
        response = bridge(settings, "search", {"query": query.strip(), "page": page})
        receipt = str(uuid4())
        with store(req).connect(write=True) as db:
            db.execute("DELETE FROM search_receipts WHERE expires<?", (time.time(),))
            db.execute("INSERT INTO search_receipts VALUES(?,?,?)",
                       (receipt, json.dumps(response["candidates"], ensure_ascii=False), time.time() + 900))
        return {**response, "search_id": receipt, "page": page}

    @app.get("/api/covers/current")
    def cover_current(req: Request):
        jobs = store(req)
        job = current(jobs)
        if job and job["state"] == "COMPLETED":
            path, meta = cover_result(jobs, job["id"])
            tracks = [track for track in STEM_FILES if cover_stem(path.parent, track).is_file()]
            try:
                target = model_dataset(jobs.root, job["payload"]["model"])
                _, source = available_dataset(jobs.root, target)
                training = {"role": source["role"]} if "vocals" in tracks else None
            except (JobError, KeyError):
                training = None
            job["result"] = {**meta, "tracks": tracks, "training_dataset": training,
                             "original_available": stage_file(path.parent, ORIGINAL_TRACK).is_file()}
        with jobs.connect() as db:
            pending = db.execute("SELECT COUNT(*) FROM cover_cleanup").fetchone()[0]
        return {"job": job, "pending_cleanup": pending}

    @app.post("/api/covers", status_code=202)
    def create_cover(req: Request, body: CoverBody):
        jobs = store(req)
        selection = enabled(jobs)
        model = selection["model"]
        if not model:
            raise Conflict("请先完成模型开启检查")
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.is_file() or digest(checkpoint) != model["sha256"]:
            raise JobError("已开启模型的内容发生变化，请重新导入和开启")
        spec = {"model": model, "inference": body.inference.model_dump(),
                "instrumental_gain_db": body.instrumental_gain_db, "vocal_gain_db": body.vocal_gain_db}
        if body.local_source and body.search_id is None and body.song_id is None:
            path = Path(body.local_source)
            if not path.is_file():
                raise JobError("本地歌曲文件不存在")
            try:
                duration = probe(path)["seconds"]
            except Exception:
                raise JobError("无法读取本地歌曲") from None
            if not 0.5 <= duration <= 1800:
                raise JobError("本地歌曲时长须介于 0.5 秒和 30 分钟")
            spec["local_source"] = str(path.resolve())
            spec["local_sha256"] = digest(path)
        elif body.search_id and body.song_id and body.local_source is None:
            with jobs.connect() as db:
                receipt = db.execute("SELECT candidates FROM search_receipts WHERE id=? AND expires>?",
                                     (str(body.search_id), time.time())).fetchone()
            if not receipt:
                raise JobError("搜索凭据已过期，请重新搜索")
            song = next((item for item in json.loads(receipt["candidates"]) if item["song_id"] == body.song_id), None)
            if not song or song["available"] is not True:
                raise JobError("选定版本不可下载完整音源")
            spec["song"] = song
        else:
            raise JobError("请选择一个在线歌曲版本或一个本地歌曲文件")
        def accept(db, job_id):
            current_selection = db.execute("SELECT job_id FROM model_selection WHERE id=1").fetchone()
            if not current_selection or current_selection["job_id"] != selection["job"]["id"]:
                raise Conflict("模型已切换，请确认后重新提交")
            accept_cover(db, job_id)

        return submit(req, "cover", body, spec, on_accept=accept)

    @app.post("/api/covers/{job_id}/remix", status_code=202)
    def remix_cover(req: Request, job_id: UUID, body: RemixBody):
        def accept(db, _):
            row = db.execute("SELECT job_id FROM current_cover WHERE id=1").fetchone()
            if not row or row["job_id"] != job_id.hex:
                raise Conflict("翻唱结果已过期")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE kind='remix' AND state IN ('QUEUED','RUNNING')").fetchone()[0] > 1:
                raise Conflict("重混音正在运行")

        cover_result(store(req), job_id.hex)
        return submit(req, "remix", body, {**body.model_dump(exclude={"request_id"}), "cover_id": job_id.hex},
                      on_accept=accept)

    @app.post("/api/covers/{job_id}/retry", status_code=202)
    def retry_cover(req: Request, job_id: UUID, body: RequestBody):
        return store(req).retry_cover(job_id.hex, str(body.request_id))

    @app.get("/api/covers/{job_id}/audio")
    def play_cover(req: Request, job_id: UUID, download: bool = False, version: str = "",
                   track: Literal["mix", "original", "vocals", "instrumental"] = "mix"):
        try:
            path, _ = cover_result(store(req), job_id.hex)
        except KeyError:
            raise HTTPException(410, "临时翻唱已过期或尚未完成") from None
        if track == "original":
            path = stage_file(path.parent, ORIGINAL_TRACK)
            filename = "original.wav"
        elif track in STEM_FILES:
            path = cover_stem(path.parent, track)
            filename = STEM_FILES[track]
        else:
            filename = "cover.wav"
        if not path.is_file():
            raise KeyError("音轨不存在")
        return FileResponse(path, media_type="audio/wav", filename=filename if download else None,
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/covers/{job_id}/dataset-fragment", status_code=202)
    def add_fragment(req: Request, job_id: UUID, body: AppendBody):
        jobs = store(req)
        cover = jobs.get(job_id.hex)
        if cover["kind"] != "cover" or cover["state"] != "COMPLETED":
            raise JobError("翻唱尚未完成")
        model = {key: cover["payload"]["model"][key] for key in ("model_id", "version_id", "sha256")}
        target = model_dataset(jobs.root, model)
        start, end = sorted(round(point * RATE) for point in (body.first_seconds, body.second_seconds))
        if end - start < RATE / 2:
            raise JobError("请选择至少 0.5 秒的人声")

        def accept(db, _):
            path, meta = cover_result(jobs, job_id.hex)
            if not stage_file(path.parent, STEM_FILES["vocals"]).is_file() or end > meta["samples"]:
                raise JobError("人声轨不可用或选区超出音频范围")
            model_dataset(jobs.root, model)
            available_dataset(jobs.root, target)

        return submit(req, "dataset_append", body, {"cover_id": job_id.hex, "model": model, "dataset": target,
                                                     "start_sample": start, "end_sample": end}, on_accept=accept)

    @app.post("/api/covers/{job_id}/save")
    def save(req: Request, job_id: UUID, body: SaveBody):
        return save_cover(store(req), job_id.hex, body.request_id.hex, body.title, body.stems, body.format)

    @app.get("/api/saved-covers")
    def saved(req: Request):
        return list(saved_entries(store(req)))

    @app.post("/api/saved-covers/bulk-delete")
    def bulk_remove_saved(req: Request, body: SavedDeleteBody):
        return delete_saved(store(req), [item.hex for item in body.ids])

    @app.delete("/api/saved-covers/{artifact_id}")
    def remove_saved(req: Request, artifact_id: UUID):
        return delete_saved(store(req), [artifact_id.hex])

    @app.get("/api/saved-covers/{artifact_id}/audio")
    def play_saved(req: Request, artifact_id: UUID, track: Literal["mix", "vocals", "instrumental"] = "mix",
                   download: bool = False):
        with store(req).connect() as db:
            row = db.execute("SELECT json_extract(metadata,'$.format') AS format FROM saved_covers WHERE id=?",
                             (artifact_id.hex,)).fetchone()
            if row is None:
                raise KeyError("已保存作品不存在")
        directory = managed_child(managed_child(settings.root, "saved_covers"), artifact_id.hex)
        if track == "mix":
            filename, media_type = SAVED_FORMATS[row["format"] or "wav"]
        else:
            filename, media_type = STEM_FILES[track], "audio/wav"
        path = stage_file(directory, filename)
        if not path.is_file():
            raise KeyError("作品音频不存在")
        return FileResponse(path, media_type=media_type, filename=filename if download else None)
