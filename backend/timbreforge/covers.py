import json
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .audio import RATE, atomic_json, digest, ffmpeg, managed_child, probe, stage_file
from .engine import invoke
from .environment import lock
from .jobs import ACTIVE, Conflict
from .music import bridge
from .worker import JobError, check_cancel


STEM_FILES = {"vocals": "converted.wav", "instrumental": "input-instrumental.wav"}
ORIGINAL_TRACK = "decoded.wav"
SAVED_FORMATS = {"wav": ("cover.wav", "audio/wav"), "mp3": ("cover.mp3", "audio/mpeg")}


def cover_stem(directory, track):
    if track == "instrumental":
        backing = stage_file(directory, "backing.wav")
        if backing.is_file():
            return backing
    return stage_file(directory, STEM_FILES[track])


def cover_directory(root, job_id):
    runtime = managed_child(root, "runtime")
    return managed_child(managed_child(runtime, "covers"), job_id)


def accept_cover(db, job_id):
    entry = db.execute("SELECT job_id FROM current_cover WHERE id=1").fetchone()
    if entry:
        old = db.execute("SELECT state FROM jobs WHERE id=?", (entry["job_id"],)).fetchone()
        mixing = db.execute("SELECT 1 FROM jobs WHERE kind='remix' AND state IN ('QUEUED','RUNNING')").fetchone()
        if old["state"] in ACTIVE or mixing:
            raise Conflict("当前翻唱或重混音正在执行")
        db.execute("INSERT OR IGNORE INTO cover_cleanup(job_id) VALUES(?)", (entry["job_id"],))
    db.execute("INSERT INTO current_cover(id,job_id) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET job_id=excluded.job_id",
               (job_id,))


def current(jobs):
    with jobs.connect() as db:
        row = db.execute("SELECT job_id FROM current_cover WHERE id=1").fetchone()
    if row:
        return jobs.get(row["job_id"])
    return None


def cleanup(jobs):
    with jobs.connect(write=True) as db:
        rows = db.execute("""SELECT job_id,attempts FROM cover_cleanup
            WHERE attempts<8 AND retry_at<=? AND NOT EXISTS (
                SELECT 1 FROM jobs WHERE kind='dataset_append' AND state IN ('QUEUED','RUNNING')
                AND json_extract(payload,'$.cover_id')=cover_cleanup.job_id)""", (time.time(),)).fetchall()
        for row in rows:
            job_id = row["job_id"]
            try:
                path = cover_directory(jobs.root, job_id)
                if path.exists():
                    shutil.rmtree(path)
            except (OSError, ValueError):
                db.execute("UPDATE cover_cleanup SET attempts=attempts+1,retry_at=? WHERE job_id=?",
                           (time.time() + 2 ** (row["attempts"] + 1), job_id))
            else:
                db.execute("DELETE FROM cover_cleanup WHERE job_id=?", (job_id,))


@lru_cache(maxsize=24)
def pitch_filter(semitones):
    import numpy as np

    pad = RATE // 2
    ratio = 2 ** (semitones / 12)
    core = f"adelay={pad}S:all=1,apad=pad_len={pad},rubberband=pitch={ratio:.10f}"
    calibration = ffmpeg("-f", "lavfi", "-i",
                         r"aevalsrc=if(eq(n\,44100)\,0.8\,0):s=44100:d=2",
                         "-af", core, "-ac", 1, "-f", "f32le", "pipe:1")
    wave = np.frombuffer(calibration, dtype="<f4")
    offset = int(np.argmax(np.abs(wave))) - RATE if len(wave) else -1
    if not pad - RATE // 10 <= offset <= pad + RATE // 10:
        raise JobError("无法校准伴奏变调延迟")
    return core, offset


def mix_audio(instrumental, vocals, destination, samples, instrumental_gain, vocal_gain, transpose=0):
    import numpy as np
    import soundfile as sf

    if transpose:
        shifted = destination.with_suffix(".shifted.wav")
        try:
            shift_backing(instrumental, shifted, samples, transpose)
            return mix_audio(shifted, vocals, destination, samples, instrumental_gain, vocal_gain)
        finally:
            shifted.unlink(missing_ok=True)
    stem, sample_rate = sf.read(instrumental, dtype="float32", always_2d=True)
    vocal, vocal_rate = sf.read(vocals, dtype="float32", always_2d=True)
    if sample_rate != RATE or vocal_rate != RATE or stem.shape[1] != 2 or vocal.shape[1] != 1:
        raise JobError("混音轨道采样率或声道数不匹配")
    if len(vocal) != samples or abs(len(stem) - samples) > 512:
        raise JobError("伴奏和转换人声的时长不匹配")
    if len(stem) < samples:
        stem = np.pad(stem, ((0, samples - len(stem)), (0, 0)))
    mixture = stem[:samples] * 10 ** (instrumental_gain / 20)
    mixture += vocal * 10 ** (vocal_gain / 20)
    if not len(mixture) or not np.isfinite(mixture).all():
        raise JobError("混音结果为空或包含非有限数值")
    peak = float(np.max(np.abs(mixture)))
    if not peak:
        raise JobError("混音结果为静音")
    mix_scale = min(1, 0.98 / peak)
    part = destination.with_suffix(".part.wav")
    try:
        sf.write(part, mixture * mix_scale, RATE, subtype="PCM_24")
        if sf.info(part).frames != samples:
            raise JobError("最终文件未覆盖完整时间轴")
        part.replace(destination)
    finally:
        part.unlink(missing_ok=True)
    return {"samples": samples, "sample_rate": RATE, "channels": 2,
            "peak": peak * mix_scale, "scale": mix_scale, "sha256": digest(destination)}


def shift_backing(instrumental, destination, samples, transpose):
    import soundfile as sf

    if abs(sf.info(instrumental).frames - samples) > 512:
        raise JobError("变调前伴奏时长与歌曲不匹配")
    filters, offset = pitch_filter(transpose)
    ffmpeg("-y", "-i", instrumental, "-af",
           f"{filters},atrim=start_sample={offset}:end_sample={offset + samples}",
           "-ar", RATE, "-ac", 2, "-c:a", "pcm_f32le", destination)


def prepare_track(source, output, channels):
    ffmpeg("-y", "-i", source, "-vn", "-sn", "-dn", "-ac", channels, "-ar", RATE, "-c:a", "pcm_f32le", output)


def cached_stage(directory, stage, key, outputs, execute):
    receipt = directory / f"{stage}.json"
    try:
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if data["inputs"] == key and all(digest(path) == data["files"][path.name] for path in outputs):
            return data["result"]
    except (OSError, ValueError, KeyError):
        pass
    result = execute()
    atomic_json(receipt, {"inputs": key, "files": {path.name: digest(path) for path in outputs}, "result": result})
    return result


def generate(jobs, job, settings):
    import soundfile as sf

    started = time.monotonic()
    job_id = job["id"]
    directory = cover_directory(jobs.root, job_id)
    directory.mkdir(parents=True, exist_ok=True)
    spec = job["payload"]

    def phase(name):
        jobs.progress(job_id, {"phase": name})

    phase("RESOLVING")
    check_cancel(jobs, job_id)
    if "song" in spec and spec["song"] is not None:
        phase("DOWNLOADING")
        source = directory / "source.audio"
        source_receipt = cached_stage(
            directory, "download", spec["song"], [source],
            lambda: bridge(settings, "download", {"song": spec["song"], "destination": str(source)}),
        )
    else:
        source = Path(spec["local_source"])
        if not source.is_file():
            raise JobError("本地歌曲文件不可读")
        if digest(source) != spec["local_sha256"]:
            raise JobError("所选本地歌曲文件已修改")
        source_receipt = {"sha256": spec["local_sha256"]}
    check_cancel(jobs, job_id)
    phase("DECODING")
    info = probe(source)
    if not 0.5 <= info["seconds"] <= 1800:
        raise JobError("歌曲时长须介于 0.5 秒和 30 分钟之间")
    normalized = directory / ORIGINAL_TRACK
    cached_stage(directory, "decoded", source_receipt["sha256"], [normalized],
                 lambda: prepare_track(source, normalized, 2))
    source_receipt["decoded_sha256"] = digest(normalized)
    if digest(source) != source_receipt["sha256"]:
        raise JobError("解码期间原始音频已改变")
    length = sf.info(normalized).frames
    check_cancel(jobs, job_id)
    phase("SEPARATING")
    stems = [directory / name for name in ("vocals.wav", "instrumental.wav")]
    separation = cached_stage(
        directory, "separation", {"source": source_receipt["decoded_sha256"], "engine": lock()["assets"]["separation"]},
        stems, lambda: invoke(jobs, job, settings, "separator_worker", "separate",
                              {"input": str(normalized), "directory": str(directory)}),
    )
    phase("PREPARING_VOCALS")
    input_vocals = directory / "input-vocals.wav"
    instrumental = directory / "input-instrumental.wav"
    prepare_track(directory / "vocals.wav", input_vocals, 1)
    prepare_track(directory / "instrumental.wav", instrumental, 2)
    if abs(sf.info(input_vocals).frames - length) > 512 or abs(sf.info(instrumental).frames - length) > 512:
        raise JobError("分离输出未覆盖原歌曲的时间轴")
    phase("LOADING_RIFT")
    converted = directory / "converted.wav"
    def convert():
        report = invoke(jobs, job, settings, "rift_worker", "infer",
                        {**spec["model"], "settings": spec["inference"], "input": str(input_vocals),
                         "output": str(converted)})
        if sf.info(converted).frames != length:
            import numpy as np

            data, _ = sf.read(converted, dtype="float32")
            if abs(len(data) - length) > 512:
                raise JobError("转换结果时长超出可解释的补偿范围")
            data = np.pad(data, (0, max(0, length - len(data))))[:length]
            sf.write(converted, data, RATE, subtype="FLOAT")
        return report

    inference = cached_stage(directory, "inference",
                             {"voice": digest(input_vocals), "rift": lock()["rift"]["commit"],
                              "model": spec["model"], "settings": spec["inference"]},
                             [converted], convert)
    phase("MIXING")
    transpose = spec["inference"]["global_transpose_semitones"]
    if transpose:
        backing = directory / "backing.wav"
        shift_backing(instrumental, backing, length, transpose)
        instrumental = backing
    mix = mix_audio(instrumental, converted, directory / "cover.wav", length,
                    spec["instrumental_gain_db"], spec["vocal_gain_db"])
    phase("VALIDATING")
    receipt = {**mix, "job_id": job_id, "song": spec.get("song"), "input": source_receipt,
               "song_title": spec["song"]["title"] if spec.get("song") else source.stem,
               "model_role": spec["model"].get("role"),
               "model": {key: spec["model"][key] for key in ("model_id", "version_id", "sha256", "speaker")},
               "inference": inference, "separation": separation, "elapsed_seconds": time.monotonic() - started}
    atomic_json(directory / "manifest.json", receipt)
    return receipt


def remix(jobs, job):
    source = job["payload"]
    directory = cover_directory(jobs.root, source["cover_id"])
    meta_file = directory / "manifest.json"
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    check_cancel(jobs, job["id"])
    destination = directory / f"mix-{job['id']}.wav"
    backing = cover_stem(directory, "instrumental")
    mix = mix_audio(backing, directory / "converted.wav", destination,
                    metadata["samples"], source["instrumental_gain_db"], source["vocal_gain_db"],
                    metadata["inference"]["settings"]["global_transpose_semitones"]
                    if backing.name == "input-instrumental.wav" else 0)
    metadata.update(mix, playback=destination.name, remix_id=job["id"])
    atomic_json(meta_file, metadata)
    return mix


def cover_result(jobs, job_id):
    job = current(jobs)
    if not job or job["id"] != job_id or job["state"] != "COMPLETED":
        raise KeyError("翻唱结果已过期或不可用")
    directory = cover_directory(jobs.root, job_id)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = stage_file(directory, manifest.get("playback", "cover.wav"))
    if not path.is_file():
        raise KeyError("翻唱音频不存在")
    return path, manifest


def save_cover(jobs, job_id, request, title, stems, audio_format="wav"):
    filename, _ = SAVED_FORMATS[audio_format]
    permanent = managed_child(jobs.root, "saved_covers")
    final = managed_child(permanent, request)
    temporary = managed_child(permanent, f".{uuid.uuid4().hex}.part")

    def matches(metadata):
        return (metadata["job_id"] == job_id and metadata["title"] == title and metadata["stems"] == stems
                and metadata.get("format", "wav") == audio_format)

    def register(db, metadata):
        db.execute("INSERT INTO saved_covers(id,metadata) VALUES(?,?)",
                   (request, json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
        return metadata

    def existing(db):
        old = db.execute("SELECT metadata FROM saved_covers WHERE id=?", (request,)).fetchone()
        if old:
            previous = json.loads(old["metadata"])
            if not matches(previous):
                raise Conflict("保存请求 ID 已使用")
            return previous
        if final.is_dir():
            try:
                recovered = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
                if matches(recovered) and digest(stage_file(final, filename)) == recovered["sha256"]:
                    return register(db, recovered)
            except (OSError, ValueError, KeyError):
                pass
            raise Conflict("保存路径已存在，请先检查作品目录")
        return None

    try:
        with jobs.connect(write=True) as db:
            previous = existing(db)
            if previous is not None:
                return previous
            path, manifest = cover_result(jobs, job_id)
            permanent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            source = temporary / "cover.wav"
            shutil.copyfile(path, source)
            if stems:
                for track, name in STEM_FILES.items():
                    shutil.copyfile(cover_stem(path.parent, track), temporary / name)

        # The snapshot outlives the temporary cover, so encoding need not hold the database lock.
        mix_sha = digest(source)
        info = probe(source)
        if mix_sha != manifest["sha256"] or info["sample_rate"] != RATE:
            raise JobError("作品复制校验失败")
        saved_sha = mix_sha
        if audio_format == "mp3":
            output = temporary / filename
            try:
                ffmpeg("-i", source, "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "320k", output)
                encoded = probe(output)
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
                raise JobError("MP3 编码失败，请检查 FFmpeg 的 MP3 编码支持") from None
            if (encoded["sample_rate"] != info["sample_rate"] or encoded["channels"] != info["channels"]
                    or not abs(encoded["seconds"] - info["seconds"]) < 0.15):
                raise JobError("MP3 音频校验失败")
            saved_sha = digest(output)
            source.unlink()
        metadata = {**manifest, "artifact_id": request, "title": title, "stems": stems, "saved_at": time.time(),
                    "format": audio_format, "playback": filename, "sha256": saved_sha, "source_mix_sha256": mix_sha}
        atomic_json(temporary / "manifest.json", metadata)
        with jobs.connect(write=True) as db:
            previous = existing(db)
            if previous is not None:
                return previous
            if final.exists():
                raise Conflict("保存路径已存在，请先检查作品目录")
            temporary.replace(final)
            return register(db, metadata)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
