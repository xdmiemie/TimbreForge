from importlib.metadata import version
from pathlib import Path
import time

from .audio import digest
from .engine import worker_entry
from .environment import lock
from .worker import JobError, check_cancel


def separate(jobs, job_id, spec):
    import torch
    from audio_separator.separator import Separator

    model = lock()["assets"]["separation"]
    paths = [jobs.root / "assets" / model["path"], jobs.root / "assets" / model.get("config", "missing.yaml")]
    if not model["sha256"] or not model.get("config_sha256"):
        raise JobError("BS-RoFormer 权重或 YAML 的 SHA256 尚未锁定")
    for path, expected in zip(paths, (model["sha256"], model["config_sha256"])):
        if not path.is_file() or digest(path) != expected:
            raise JobError(f"缺少校验通过的分离资产 {path.name}")
    if version("audio-separator") != model["version"]:
        raise JobError("audio-separator 软件版本与锁定清单不一致")
    if not torch.cuda.is_available():
        raise JobError("分离环境无法使用 CUDA")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    directory = Path(spec["directory"])

    class LocalRoFormer(Separator):
        def download_model_files(self, model_filename):
            if model_filename != model["path"]:
                raise JobError("请求的分离模型与已校验的资产不一致")
            self.model_is_uvr_vip = False
            self.model_friendly_name = "BS-RoFormer 317"
            return model_filename, "MDXC", self.model_friendly_name, str(paths[0]), paths[1].name

    separator = LocalRoFormer(
        model_file_dir=str(paths[0].parent), output_dir=str(directory),
        output_format="WAV", use_soundfile=True, sample_rate=44100,
        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                     "batch_size": 1, "overlap": None, "pitch_shift": 0},
        use_torch_compile=False,
    )
    if getattr(separator.torch_device, "type", None) != "cuda":
        raise JobError("分离器未选用 CUDA 设备")
    separator.load_model(model_filename=paths[0].name)
    jobs.progress(job_id, {"phase": "SEPARATING", "message": "BS-RoFormer 已加载"})
    paths = separator.separate(spec["input"], custom_output_names={"Vocals": "vocals", "Instrumental": "instrumental"})
    check_cancel(jobs, job_id)
    if {Path(path).name.casefold() for path in paths} != {"vocals.wav", "instrumental.wav"}:
        raise JobError("分离器未返回明确的人声与伴奏两轨")
    engine = separator.model_instance
    return {"model": model["path"], "backend": model["version"], "batch_size": engine.batch_size,
            "overlap": engine.overlap, "precision": str(separator.effective_precision),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.monotonic() - started}


if __name__ == "__main__":
    worker_entry({"separate": separate})
