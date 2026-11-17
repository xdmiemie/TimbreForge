import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
import wave

from .audio import RATE, atomic_json, digest, ffmpeg, managed_child, probe
from .worker import JobError, check_cancel


ACCEPTED = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".aiff", ".wma"}
MIN_SECONDS = 0.5
MAX_SECONDS = 1800
SILENCE_DB = -48
SILENCE_SECONDS = 0.45
MAX_CLIP_SECONDS = 15


def source_files(source, recurse):
    if not source.is_dir():
        raise JobError("数据源目录不存在或不可读")
    candidates = source.rglob("*") if recurse else source.glob("*")
    for path in sorted(candidates):
        if path.suffix.lower() in ACCEPTED and path.is_file() and not path.is_symlink():
            if path.resolve().is_relative_to(source):
                yield path


def decode(path):
    import numpy as np

    pcm = ffmpeg("-i", path, "-vn", "-sn", "-dn", "-t", MAX_SECONDS + 1,
                 "-ac", 1, "-ar", RATE, "-f", "f32le", "pipe:1")
    samples = np.frombuffer(pcm, dtype="<f4")
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("empty or invalid PCM")
    return samples


def clip_bounds(samples):
    import numpy as np

    frame = RATE // 50
    padded = np.pad(samples, (0, (-len(samples)) % frame))
    rms = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    active = np.flatnonzero(rms >= 10 ** (SILENCE_DB / 20))
    if not len(active):
        return [], 1.0
    boundaries = np.flatnonzero(np.diff(active) * frame >= SILENCE_SECONDS * RATE) + 1
    ranges = []
    for group in np.split(active, boundaries):
        start = max(0, int(group[0]) * frame - RATE // 10)
        stop = min(len(samples), (int(group[-1]) + 1) * frame + RATE // 10)
        if stop - start < RATE * MIN_SECONDS:
            continue
        chunks = math.ceil((stop - start) / (RATE * MAX_CLIP_SECONDS))
        points = [start + (stop - start) * n // chunks for n in range(chunks + 1)]
        ranges.extend(zip(points, points[1:]))
    return ranges, 1 - len(active) / len(rms)


def write_clip(path, clip):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(clip)


def normalize(samples):
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ac", "1", "-ar", str(RATE),
         "-i", "pipe:0", "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
         "-ar", str(RATE), "-ac", "1", "-f", "s16le", "pipe:1"],
        input=samples.tobytes(), capture_output=True, check=True, timeout=45,
    ).stdout


def process(jobs, job):
    import numpy as np

    job_id = job["id"]
    payload = job["payload"]
    source = Path(payload["source"]).resolve()
    generated = (jobs.root / "datasets").resolve()
    if generated.is_relative_to(source) or source.is_relative_to(generated):
        raise JobError("数据源不能包含项目生成的数据集目录")
    paths = list(source_files(source, payload["recursive"]))
    if not paths:
        raise JobError("目录内未找到受支持的音频")

    def check():
        check_cancel(jobs, job_id)

    prepare = job["kind"] == "dataset_prepare"
    dataset_id = hashlib.sha256((str(source) + "\0" + payload["role"]).encode()).hexdigest()[:24]
    dataset_root = jobs.root / "datasets" / dataset_id
    temporary = None
    records, clips, seen = [], [], set()
    excluded = set(payload.get("excluded_paths", []))
    try:
        if prepare:
            dataset_root.mkdir(parents=True, exist_ok=True)
            temporary = managed_child(dataset_root, "." + job_id + ".part")
            (temporary / "speaker").mkdir(parents=True)
        for index, path in enumerate(paths, 1):
            check()
            record = {"path": str(path.relative_to(source)), "retained_seconds": 0}
            try:
                sha = digest(path, check)
                record["sha256"] = sha
                meta = probe(path)
                if not math.isfinite(meta["seconds"]) or not 0 < meta["seconds"] <= MAX_SECONDS:
                    record.update(reason="duration_out_of_range")
                    records.append(record)
                    continue
                samples = decode(path)
                bounds, silence = clip_bounds(samples)
            except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
                record.update(reason="cannot_decode")
                records.append(record)
                continue
            seconds = len(samples) / RATE
            peak = float(np.max(np.abs(samples)))
            metadata = {**meta, "seconds": seconds, "peak": peak, "silence_ratio": silence,
                        "clipped_ratio": float(np.mean(np.abs(samples) >= 0.999))}
            record.update(metadata)
            if record["clipped_ratio"] >= 0.01:
                record["warnings"] = ["severe_clipping"]
            if record["path"] in excluded:
                record["reason"] = "user_excluded"
            elif sha in seen or not bounds:
                record["reason"] = "duplicate" if sha in seen else "silent_or_too_short"
            else:
                seen.add(sha)
                for number, (start, end) in enumerate(bounds):
                    check()
                    file_name = f"{sha}-{number:03d}"
                    retained_seconds = (end - start) / RATE
                    if prepare:
                        out = temporary / "speaker" / f"{file_name}.wav"
                        audio = normalize(samples[start:end])
                        if len(audio) != (end - start) * 2:
                            raise JobError("整理音频的采样长度发生变化")
                        write_clip(out, audio)
                        if probe(out)["sample_rate"] != RATE:
                            raise JobError("整理音频的采样率无效")
                    clips.append({"file_name": file_name, "group": sha, "start_sample": start,
                                  "end_sample": end, "seconds": retained_seconds,
                                  "sha256": digest(out) if prepare else None})
                    record["retained_seconds"] += retained_seconds
            if prepare and digest(path, check) != sha:
                raise JobError(f"整理期间源文件发生变化：{record['path']}")
            records.append(record)
            jobs.progress(job_id, {"phase": "DATASET_PREPARE" if prepare else "DATASET_SCAN",
                                   "completed": index, "total": len(paths)})
        groups = sorted({clip["group"] for clip in clips})
        random.Random(42).shuffle(groups)
        count = min(len(groups) - 1, max(1, round(0.1 * len(groups)))) if len(groups) >= 2 else 0
        validation = set(groups[:count])
        for clip in clips:
            clip["split"] = "validation" if clip["group"] in validation else "train"
        train_count = sum(clip["split"] == "train" for clip in clips)
        original = math.fsum(item.get("seconds", 0) for item in records)
        retained = math.fsum(clip["seconds"] for clip in clips)
        stats = {"decoded_seconds": original, "retained_seconds": retained,
                 "excluded_seconds": max(0, original - retained), "train_clips": train_count,
                 "validation_clips": len(clips) - train_count, "default_train_batches": train_count // 16}
        manifest = {"role": payload["role"], "language": payload["language"],
                    "dataset_id": dataset_id, "version_id": job_id, "source": str(source),
                    "created": time.time(), "sources": records, "clips": clips, "stats": stats,
                    "config": {"sample_rate": RATE, "lufs": -18, "peak_db": -1.5, "silence_db": SILENCE_DB,
                               "silence_seconds": SILENCE_SECONDS, "min_seconds": MIN_SECONDS,
                               "max_clip_seconds": MAX_CLIP_SECONDS, "split_seed": 42,
                               "recursive": payload["recursive"], "excluded_paths": sorted(excluded)}}
        if prepare:
            if not clips:
                raise JobError("没有可整理的音频片段")
            snapshot = {key: manifest[key] for key in ("sources", "clips", "config", "role", "language")}
            manifest["fingerprint"] = hashlib.sha256(
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            check()
            atomic_json(temporary / "manifest.json", manifest)
            temporary.replace(managed_child(dataset_root, job_id))
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def manifests(root):
    for path in sorted((root / "datasets").glob("*/*/manifest.json")):
        if not path.parent.name.startswith("."):
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue    # The version was deleted during enumeration.
