from pathlib import Path

from .audio import digest
from .engine import worker_entry
from .environment import lock
from .worker import JobError


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise JobError("RIFT 工作环境无法使用 CUDA")
    return torch


def require_module_assets(operation):
    module_files = {
        "features": ("pretrained/content-vec-best/config.json", "pretrained/content-vec-best/pytorch_model.bin",
                     "pretrained/rmvpe/model.pt"),
        "inference": ("pretrained/content-vec-best/config.json", "pretrained/content-vec-best/pytorch_model.bin",
                      "pretrained/rmvpe/model.pt", "pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt",
                      "pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/config.json"),
    }
    paths = [Path(path) for path in module_files[operation]]
    hashes = lock()["rift_modules"]["files"]
    for path in paths:
        if not path.is_file():
            raise JobError(f"缺少 RIFT 资产 {path}")
        expected = hashes[path.relative_to("pretrained").as_posix()]["sha256"]
        if digest(path) != expected:
            raise JobError(f"RIFT 资产校验不符 {path}")
    return paths


def check_model(jobs, job_id, spec):
    from .models import validate_model

    require_cuda()
    require_module_assets("inference")
    path = Path(spec["checkpoint"])
    if digest(path) != spec["sha256"]:
        raise JobError("模型文件已修改，请重新验证")
    _, speakers, _, metadata = validate_model(path, "cuda:0")
    if spec["speaker"] not in speakers:
        raise JobError("所选说话人不存在")
    return metadata


def main():
    from .augmentation import append_range
    from .features import prepare_features
    from .inference import infer_file
    from .models import import_external, publish
    from .rift_training import train

    worker_entry({"features": prepare_features, "append_range": append_range, "train": train, "inspect": import_external,
                  "export": publish, "check_model": check_model, "infer": infer_file}, needs_rift=True)


if __name__ == "__main__":
    main()
