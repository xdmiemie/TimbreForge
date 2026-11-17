import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .audio import atomic_json, digest, managed_child
from .environment import lock
from .worker import JobError, check_cancel


def dataset(root, payload):
    try:
        base = managed_child(root, "datasets")
        path = managed_child(managed_child(base, payload["dataset_id"]), payload["version_id"])
    except ValueError:
        raise JobError("数据版本路径无效") from None
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise JobError("数据版本不存在")
    return path, json.loads(manifest.read_text(encoding="utf-8"))


def train_preflight(root, payload):
    path, metadata = dataset(root, payload)
    stats = metadata["stats"]
    batch = payload["train_batch_size"]
    train_batches = stats["train_clips"] // batch
    if not train_batches or not stats["validation_clips"]:
        raise JobError(f"训练片段 {stats['train_clips']}，验证片段 {stats['validation_clips']}，"
                       f"物理 batch {batch}。训练和验证均须至少有一个批次")
    pointer = path / "features.json"
    feature = json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else None
    return {"dataset_fingerprint": metadata["fingerprint"], "train_batches": train_batches,
            "effective_batch": batch * payload["gradient_accumulation_steps"],
            "feature_manifest_present": feature is not None,
            "feature_key": feature["key"] if feature else None,
            "feature_digest": feature.get("content_sha256") if feature else None}


def finite_feature(base, frames, name):
    import torch

    path = Path(f"{base}.{name}.pt")
    value = torch.load(path, weights_only=True, map_location="cpu")
    if not value.is_floating_point() or not value.numel() or not torch.isfinite(value).all():
        raise ValueError(f"{name} 特征数值无效")
    if name == "mel" and (value.ndim != 3 or value.shape[:2] != (1, 128)
                          or abs(value.shape[-1] - frames) > 1):
        raise ValueError("mel 特征形状无效")
    if name in {"rms", "f0"} and (value.squeeze().ndim != 1 or abs(value.numel() - frames) > 2):
        raise ValueError(f"{name} 帧长无效")
    if name == "cvec" and (value.ndim != 2 or value.shape[1] != 768):
        raise ValueError("ContentVec 特征形状无效")


def feature_name(index):
    return f"clip-{index:08d}"


def reuse_file(source, destination):
    # Published data is immutable; the two versions retain independent directory entries.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def previous_features(parent, key_data):
    if parent:
        directory, manifest = parent
        try:
            value = json.loads((directory / "features.json").read_text(encoding="utf-8"))
            previous_key = {**key_data, "dataset": manifest["fingerprint"]}
            key = hashlib.sha256(json.dumps(previous_key, sort_keys=True).encode()).hexdigest()
            if value.get("key") == key and all(value.get(k) == v for k, v in previous_key.items()):
                return directory / "features" / key / "speaker"
        except (OSError, ValueError):
            pass
    return None


def prepare_features(jobs, job_id, payload, *, staged=None, parent=None):
    import torch

    from .rift_worker import require_cuda, require_module_assets

    require_cuda()
    module_assets = require_module_assets("features")
    sys.path.insert(0, str(Path("scripts").resolve()))
    from prepare_cvec import process_cvec
    from prepare_f0 import process_f0
    from prepare_mel import process_audio
    from prepare_rms import process_rms

    directory, source = staged if staged is not None else dataset(jobs.root, payload)
    key_data = {"adapter": 2, "commit": lock()["rift"]["commit"],
                "assets": {str(path): digest(path) for path in module_assets},
                "dataset": source["fingerprint"]}
    cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    feature_root = directory / "features" / cache_key
    target = feature_root / "speaker"
    target.mkdir(parents=True, exist_ok=True)
    metadata = {"speakers": ["speaker"], "train_audios": [], "test_audios": []}
    feature_hashes = {}
    previous = previous_features(parent, key_data)
    torch.cuda.reset_peak_memory_stats()
    for index, clip in enumerate(source["clips"], 1):
        check_cancel(jobs, job_id)
        name = feature_name(index)
        wav = directory / "speaker" / f"{clip['file_name']}.wav"
        if digest(wav) != clip["sha256"]:
            raise JobError(f"冻结音频已修改（片段 {index}）")
        frames = (clip["end_sample"] - clip["start_sample"]) // 512 + 1
        item = {"speaker": "speaker", "file_name": name, "frame_len": frames}
        done = target / f"{name}.json"
        try:
            recorded = json.loads(done.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = {}
        inherited = previous is not None and index <= len(parent[1]["clips"])
        if inherited:
            try:
                recorded = json.loads((previous / done.name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                inherited = False
        valid, missing = {}, []
        for suffix in ("mel", "rms", "f0", "cvec"):
            file = f"{name}.{suffix}.pt"
            path = target / file
            if inherited and file in recorded:
                old = previous / file
                if old.is_file() and digest(old) == recorded[file]:
                    reuse_file(old, path)
                    valid[file] = recorded[file]
                    continue
            if file in recorded and path.is_file() and digest(path) == recorded[file]:
                valid[file] = recorded[file]
            else:
                missing.append(suffix)
        if missing:
            with tempfile.TemporaryDirectory(prefix="ft-", dir=jobs.state) as temp:
                scratch = Path(temp)
                (scratch / "speaker").mkdir()
                shutil.copyfile(wav, scratch / "speaker" / f"{name}.wav")
                operations = {
                    "mel": (process_audio, (512, 128, 44100, False, True, "cuda")),
                    "rms": (process_rms, (512, False)),
                    "f0": (process_f0, ("pretrained/rmvpe/model.pt", 512, 44100, True, False)),
                    "cvec": (process_cvec, ("pretrained/content-vec-best", True, False)),
                }
                for suffix in missing:
                    check_cancel(jobs, job_id)
                    operation, args = operations[suffix]
                    operation(item, scratch, *args)
                    try:
                        finite_feature(scratch / "speaker" / name, frames, suffix)
                    except ValueError as exc:
                        raise JobError(f"片段 {index} 的 {exc}") from None
                    except (OSError, RuntimeError):
                        raise JobError(f"{suffix} 特征提取或读取失败（片段 {index}）") from None
                    part = scratch / "speaker" / f"{name}.{suffix}.pt"
                    valid[part.name] = digest(part)
                    part.replace(target / part.name)
        if missing or inherited:
            atomic_json(done, valid)
        feature_hashes.update(valid)
        metadata["train_audios" if clip["split"] == "train" else "test_audios"].append(item)
        jobs.progress(job_id, {"phase": "FEATURES", "completed": index, "total": len(source["clips"])})
    atomic_json(feature_root / "meta_info.json", metadata)
    content_hash = hashlib.sha256(json.dumps(feature_hashes, sort_keys=True).encode()).hexdigest()
    result = {**key_data, "key": cache_key, "content_sha256": content_hash,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(directory / "features.json", result)
    return result
