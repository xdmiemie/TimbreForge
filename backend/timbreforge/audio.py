import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import uuid


RATE = 44100


def ffmpeg_dll_directory(astrbot):
    if os.name != "nt":
        return nullcontext()
    override = os.environ.get("TIMBREFORGE_FFMPEG_DLL_DIR")
    executable = shutil.which("ffmpeg")
    candidates = [Path(override)] if override else []
    if executable:
        candidates.append(Path(executable).parent)
    candidates.append(astrbot / "runtime_env" / "Library" / "bin")
    for path in candidates:
        if path.is_dir() and list(path.glob("avutil-*.dll")):
            return os.add_dll_directory(str(path.resolve()))
    return nullcontext()


def ffmpeg(*args, capture=True):
    return subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *map(str, args)],
                          capture_output=capture, check=True, timeout=900).stdout


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams", "-show_format",
         "-of", "json", str(path)], capture_output=True, check=True, timeout=30,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    seconds = float(stream.get("duration") or data["format"]["duration"])
    return {"seconds": seconds, "sample_rate": int(stream["sample_rate"]), "channels": stream["channels"]}


def digest(path, check=lambda: None):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            check()
            sha.update(chunk)
    return sha.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def managed_child(root, child):
    resolved_root = root.resolve()
    candidate = root / child
    if candidate.resolve().parent != resolved_root or candidate.is_symlink():
        raise ValueError("产物路径越界")
    if candidate.exists() and getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("产物目录不能是重解析点")
    return candidate


def stage_file(root, name):
    return managed_child(root, name)
