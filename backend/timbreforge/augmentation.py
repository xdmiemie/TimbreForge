import hashlib
import json
import math
import re
import shutil
import time

from .audio import RATE, atomic_json, digest, managed_child, stage_file
from .covers import cover_directory
from .datasets import MIN_SECONDS, clip_bounds, normalize, write_clip
from .features import dataset, prepare_features, reuse_file
from .models import model_version
from .worker import JobError, check_cancel


def model_dataset(root, spec):
    model_id, version_id = spec["model_id"], spec["version_id"]
    model = model_version(root, model_id, version_id)
    if model.get("source") == "external":
        raise JobError("外部模型没有可追加的原训练数据集")
    if spec.get("sha256") and spec["sha256"] != model["sha256"]:
        raise JobError("翻唱所用的模型版本已变化")
    version = managed_child(managed_child(managed_child(root, "models"), "characters"), model_id)
    summary_path = managed_child(version, version_id) / "training_summary.json"
    if not summary_path.is_file():
        raise JobError("此模型缺少训练数据来源记录")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_id") != model_id or not re.fullmatch(r"[a-f0-9]{32}", summary.get("version_id", "")):
        raise JobError("此模型的训练数据来源记录无效")
    return {"dataset_id": model_id, "version_id": summary["version_id"]}


def append_parent(root, base):
    original = base["version_id"]
    group = managed_child(managed_child(root, "datasets"), base["dataset_id"])
    parent = None
    for path in group.glob("*/manifest.json"):
        if not re.fullmatch(r"[a-f0-9]{32}", path.parent.name):
            continue
        candidate = dataset(root, {**base, "version_id": path.parent.name})
        if path.parent.name != original and original not in candidate[1].get("ancestors", []):
            continue
        if parent is None or (candidate[1]["created"], candidate[1]["version_id"]) > (
                parent[1]["created"], parent[1]["version_id"]):
            parent = candidate
    if parent is None:
        raise JobError("找不到此模型的训练数据版本")
    return parent


def available_dataset(root, base):
    try:
        return dataset(root, base)
    except JobError:
        return append_parent(root, base)


def append_range(jobs, job_id, spec):
    import numpy as np
    import soundfile as sf

    parent_dir, parent = append_parent(jobs.root, spec["dataset"])
    vocal = stage_file(cover_directory(jobs.root, spec["cover_id"]), "converted.wav")
    if not vocal.is_file():
        raise JobError("转换人声已不可用")
    start, end = spec["start_sample"], spec["end_sample"]
    with sf.SoundFile(vocal) as stream:
        if stream.samplerate != RATE or stream.channels != 1 or not 0 <= start < end <= stream.frames:
            raise JobError("裁剪范围与人声轨时间轴不匹配")
        stream.seek(start)
        samples = stream.read(end - start, dtype="float32")
    if len(samples) != end - start or not np.isfinite(samples).all():
        raise JobError("裁剪范围的人声音频不完整")
    bounds, _ = clip_bounds(samples)
    if not bounds:
        raise JobError("所选范围没有足够长的人声，请重新选取")
    group = hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()
    if any(clip["group"] == group for clip in parent["clips"]):
        raise JobError("该人声片段已加入数据集")
    directory = managed_child(parent_dir.parent, "." + job_id + ".part")
    count = len(bounds)
    jobs.progress(job_id, {"phase": "CLIPPING", "completed": 0, "total": count})
    try:
        (directory / "speaker").mkdir(parents=True)
        clips = list(parent["clips"])
        for clip in parent["clips"]:
            check_cancel(jobs, job_id)
            name = f"{clip['file_name']}.wav"
            reuse_file(managed_child(parent_dir / "speaker", name), directory / "speaker" / name)
        for index, (first, last) in enumerate(bounds, 1):
            check_cancel(jobs, job_id)
            name = f"{group}-{index - 1:03d}"
            path = directory / "speaker" / (name + ".wav")
            audio = normalize(samples[first:last])
            if len(audio) != (last - first) * 2:
                raise JobError("整理音频的采样长度发生变化")
            write_clip(path, audio)
            clips.append({"file_name": name, "group": group, "start_sample": start + first,
                          "end_sample": start + last, "seconds": (last - first) / RATE,
                          "sha256": digest(path), "split": "train"})
            jobs.progress(job_id, {"phase": "CLIPPING", "completed": index, "total": count})
        seconds = len(samples) / RATE
        added = math.fsum((last - first) / RATE for first, last in bounds)
        stats = dict(parent["stats"])
        stats["train_clips"] += count
        stats["decoded_seconds"] += seconds
        stats["retained_seconds"] += added
        stats["excluded_seconds"] += max(0, seconds - added)
        stats["default_train_batches"] = stats["train_clips"] // 16
        origin = {key: spec[key] for key in ("cover_id", "model", "start_sample", "end_sample")}
        manifest = {**parent, "version_id": job_id, "clips": clips, "stats": stats, "created": time.time(),
                    "ancestors": [*parent.get("ancestors", []), parent["version_id"]],
                    "initial_model": spec["model"],
                    "generated_ranges": [*parent.get("generated_ranges", []), {**origin, "group": group}]}
        snapshot = {key: manifest[key] for key in ("sources", "clips", "config", "role", "language", "generated_ranges")}
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        prepare_features(jobs, job_id, spec["dataset"], staged=(directory, manifest), parent=(parent_dir, parent))
        check_cancel(jobs, job_id)
        atomic_json(directory / "manifest.json", manifest)
        directory.replace(managed_child(parent_dir.parent, job_id))
    finally:
        if directory.exists():
            shutil.rmtree(directory)
    return {"dataset_id": parent["dataset_id"], "version_id": job_id, "role": parent["role"],
            "added_clips": count, "previous_train_clips": parent["stats"]["train_clips"],
            "train_clips": stats["train_clips"]}
