import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from .audio import atomic_json, digest, managed_child
from .environment import lock
from .worker import JobError


def checkpoint_path(path):
    if path.is_dir():
        path = path / "model.ckpt"
    if not path.is_file() or path.suffix.lower() != ".ckpt":
        raise JobError("请选择 RIFT V3 .ckpt 或含 model.ckpt 的模型包")
    return path.resolve()


def validate_model(path, device="cpu"):
    import torch
    from rift_svc import DiT, RF

    try:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        cfg = checkpoint["hyper_parameters"]["cfg"]
        model_cfg, spk2idx, dataset_cfg = cfg["model"], cfg["spk2idx"], cfg["dataset"]
        if (not isinstance(spk2idx, dict) or not spk2idx
                or any(not isinstance(name, str) for name in spk2idx)
                or sorted(spk2idx.values()) != list(range(len(spk2idx)))):
            raise ValueError("invalid speakers")
        if (model_cfg["dim"], model_cfg["depth"]) not in {(512, 8), (768, 12), (1024, 16)}:
            raise ValueError("unsupported RIFT V3 model")
        if model_cfg["n_mel_channels"] != 128:
            raise ValueError("unsupported mel channels")
        state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()
                 if key.startswith("model.")}
        model = RF(transformer=DiT(num_speaker=len(spk2idx), **model_cfg))
        model.load_state_dict(state, strict=True)
        model.eval()
        if device != "cpu":
            model = model.to(device=device, dtype=torch.float16)
        metadata = {"architecture": f"dit-{model_cfg['dim']}-{model_cfg['depth']}",
                    "speakers": list(spk2idx), "defaults": {"inference_steps": 64}}
        return model, spk2idx, dataset_cfg, metadata
    except (ValueError, RuntimeError, KeyError, TypeError, AttributeError, OSError, EOFError, __import__("pickle").UnpicklingError):
        raise JobError("此检查点未通过 RIFT V3 结构和 CPU 权重严格加载检查") from None


def list_models(root):
    base = root / "models" / "characters"
    result = []
    for path in sorted(base.glob("*/*/manifest.json")):
        if not path.parent.name.startswith("."):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                value["present"] = Path(value["checkpoint"]).is_file()
                smoke = path.with_name("gpu_smoke.json")
                if smoke.is_file():
                    receipt = json.loads(smoke.read_text(encoding="utf-8"))
                    if receipt["sha256"] == value["sha256"]:
                        value["gpu_smoke_test_status"] = receipt["status"]
                result.append(value)
            except FileNotFoundError:
                continue
    return sorted(result, key=lambda item: item["created"], reverse=True)


def model_version(root, model_id, version_id):
    try:
        root = managed_child(managed_child(root, "models"), "characters")
        path = managed_child(managed_child(root, model_id), version_id) / "manifest.json"
    except ValueError:
        raise JobError("模型版本路径无效") from None
    if not path.is_file():
        raise JobError("模型版本不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def record_smoke(jobs, job_id, spec, result):
    path = jobs.root / "models" / "characters" / spec["model_id"] / spec["version_id"] / "gpu_smoke.json"
    atomic_json(path, {"status": "PASSED", "job_id": job_id, "sha256": spec["sha256"],
                       "settings": result["settings"], "samples": result["samples"]})


def import_external(jobs, job_id, spec):
    path = checkpoint_path(Path(spec["path"]))
    sha = digest(path)
    _, _, _, metadata = validate_model(path)
    if digest(path) != sha:
        raise JobError("检查期间外部模型已修改")
    model_id = hashlib.sha256(str(path).encode()).hexdigest()[:24]
    root = jobs.root / "models" / "characters" / model_id
    version = root / sha[:32]
    version.mkdir(parents=True, exist_ok=True)
    manifest = {**metadata, "model_id": model_id, "version_id": sha[:32],
                "checkpoint": str(path), "sha256": sha, "role": spec.get("role") or path.stem,
                "source": "external", "created": time.time(), "gpu_smoke_test_status": "NOT_RUN"}
    atomic_json(version / "manifest.json", manifest)
    jobs.progress(job_id, {"phase": "MODEL_PUBLISHED", "model_id": model_id, "version_id": sha[:32]})
    return manifest


def publish(jobs, job_id, spec):
    import torch

    src = Path(spec["snapshot"])
    summary = spec["summary"]
    run_id = summary["run_id"]
    root = jobs.root / "models" / "characters" / summary["dataset_id"]
    version_id = spec["version_id"]
    root.mkdir(parents=True, exist_ok=True)
    final = managed_child(root, version_id)
    if (final / "manifest.json").is_file():
        return json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    tmp = managed_child(root, f".{version_id}.{uuid.uuid4().hex}.part")
    tmp.mkdir()
    try:
        path = tmp / "model.ckpt"
        if spec.get("snapshot_sha256") and digest(src) != spec["snapshot_sha256"]:
            raise JobError("导出恢复点的校验值发生变化")
        source = torch.load(src, weights_only=True, map_location="cpu", mmap=True)
        torch.save({"state_dict": source["state_dict"], "hyper_parameters": source["hyper_parameters"]}, path)
        del source
        _, _, _, metadata = validate_model(path)
        manifest = {**metadata, "model_id": summary["dataset_id"], "version_id": version_id,
                    "checkpoint": str(final / "model.ckpt"), "sha256": digest(path),
                    "role": summary["role"], "language": summary["language"], "step": summary["step"],
                    "run_id": run_id, "source": summary["reason"], "created": time.time(),
                    "upstream": lock()["rift"]["commit"], "gpu_smoke_test_status": "NOT_RUN"}
        atomic_json(tmp / "training_summary.json", summary)
        atomic_json(tmp / "manifest.json", manifest)
        tmp.replace(final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    jobs.progress(job_id, {"phase": "MODEL_PUBLISHED", "model_id": summary["dataset_id"],
                           "version_id": version_id, "step": summary["step"]})
    return manifest
