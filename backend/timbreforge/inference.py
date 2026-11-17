from pathlib import Path
import random
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .audio import RATE, digest, ffmpeg
from .datasets import clip_bounds
from .worker import JobError, check_cancel


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    inference_steps: int = Field(64, ge=1, le=256)
    ds_cfg_strength: float = Field(0.2, ge=0, le=5)
    spk_cfg_strength: float = Field(0.8, ge=0, le=5)
    cfg_rescale: float = Field(0.7, ge=0, le=1)
    robust_f0: Literal[0, 1, 2] = 0
    global_transpose_semitones: int = Field(0, ge=-12, le=12)
    vocal_octave_offset: Literal[-12, 0, 12] = 0
    seed: int = Field(42, ge=0, le=2**32 - 1)


def infer_file(jobs, job_id, spec):
    import numpy as np
    import soundfile as sf

    from .models import validate_model
    from .rift_worker import require_cuda, require_module_assets
    from infer import process_segment
    from rift_svc.feature_extractors import HubertModelWithFinalProj, RMSExtractor
    from rift_svc.nsf_hifigan import NsfHifiGAN
    from rift_svc.rmvpe import RMVPE

    torch = require_cuda()
    require_module_assets("inference")
    started = time.monotonic()
    cfg = InferenceConfig.model_validate(spec["settings"])
    device = torch.device("cuda:0")
    if digest(Path(spec["checkpoint"])) != spec["sha256"]:
        raise JobError("模型文件已修改，请重新验证")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    model, speakers, _, metadata = validate_model(Path(spec["checkpoint"]), device)
    if spec["speaker"] not in speakers:
        raise JobError("所选说话人不存在")
    vocoder = NsfHifiGAN("pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt").to(device).half().eval()
    f0 = RMVPE(model_path="pretrained/rmvpe/model.pt", hop_length=160, device=device)
    hubert = HubertModelWithFinalProj.from_pretrained("pretrained/content-vec-best").to(device).half().eval()
    rms = RMSExtractor().to(device).half().eval()
    decoded = ffmpeg("-i", spec["input"], "-vn", "-ac", 1, "-ar", RATE, "-f", "f32le", "pipe:1")
    source = np.frombuffer(decoded, dtype="<f4")
    if not len(source) or not np.isfinite(source).all():
        raise JobError("人声轨为空或包含非有限数值")
    bounds, _ = clip_bounds(source)
    if not bounds:
        raise JobError("人声轨中未找到可转换片段")
    output = np.zeros_like(source)
    layers = 6
    completed = []
    for index, (start, end) in enumerate(bounds, 1):
        check_cancel(jobs, job_id)
        segment = source[start:end].copy()
        converted = process_segment(
            segment, model, vocoder, f0, hubert, rms, speakers[spec["speaker"]], RATE, 512, device,
            key_shift=cfg.global_transpose_semitones + cfg.vocal_octave_offset,
            infer_steps=cfg.inference_steps, ds_cfg_strength=cfg.ds_cfg_strength,
            spk_cfg_strength=cfg.spk_cfg_strength, skip_cfg_strength=0.0, cfg_skip_layers=layers,
            cfg_rescale=cfg.cfg_rescale, cvec_downsample_rate=2, robust_f0=cfg.robust_f0, use_fp16=True,
        )
        if converted.shape != segment.shape or not np.isfinite(converted).all():
            raise JobError(f"RIFT 第 {index} 段长度或数值不匹配")
        fade = min(RATE // 100, len(converted) // 2)
        if fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            converted[:fade] *= ramp
            converted[-fade:] *= ramp[::-1]
        output[start:end] = converted
        completed.append({"start_sample": start, "end_sample": end, "fade_samples": fade})
        jobs.progress(job_id, {"phase": "CONVERTING", "completed": index, "total": len(bounds),
                               "inference_steps": cfg.inference_steps})
    peak = float(np.max(np.abs(output)))
    if peak == 0:
        raise JobError("转换结果为静音")
    sf.write(spec["output"], output, RATE, subtype="FLOAT")
    return {**metadata, "settings": cfg.model_dump(), "samples": len(output),
            "segments": completed, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.monotonic() - started}
