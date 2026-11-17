import json
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .audio import digest, ffmpeg, probe
from .settings import ROOT


def lock():
    return json.loads((ROOT / "configs/upstreams.lock.json").read_text(encoding="utf-8"))


def assets(settings):
    result = {}
    for key, spec in lock()["assets"].items():
        path = settings.root / "assets" / spec["path"]
        result[key] = {"path": str(path), "present": path.is_file(),
                       "locked": bool(spec["sha256"])}
    return result


def gpu_snapshot():
    def query(option):
        output = subprocess.check_output(
            ["nvidia-smi", option, "--format=csv,noheader,nounits"], timeout=10,
            text=True, encoding="utf-8", errors="replace",
        )
        return list(csv.reader(output.splitlines(), skipinitialspace=True))

    try:
        devices = [{"name": name, "driver": driver, "total_mib": int(total), "free_mib": int(free)}
                   for name, driver, total, free in query("--query-gpu=name,driver_version,memory.total,memory.free")]
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"devices": [], "compute_processes": [], "process_details_available": False}
    try:
        processes = [{"pid": int(pid), "name": Path(name).name, "used_mib": int(used) if used.isdigit() else None}
                     for pid, name, used in query("--query-compute-apps=pid,process_name,used_gpu_memory") if pid.isdigit()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"devices": devices, "compute_processes": [], "process_details_available": False}
    return {"devices": devices, "compute_processes": processes, "process_details_available": True}


def fetch_base(settings, *, progress=lambda _: None, check=lambda: None, source=None):
    import requests

    spec = lock()["assets"]["base"]
    path = settings.root / "assets" / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and digest(path, check) == spec["sha256"]:
        return path
    partial = path.with_name(path.name + ".part")
    if source is not None:
        if digest(source, check) != spec["sha256"]:
            raise ValueError("缓存底模的 SHA256 校验失败")
        with source.open("rb") as stream, partial.open("wb") as sink:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                check()
                sink.write(chunk)
    else:
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(spec["url"], headers=headers, stream=True, timeout=(10, 30)) as response:
                    if response.status_code == 416 and offset and digest(partial, check) == spec["sha256"]:
                        break
                    if response.status_code == 416:
                        partial.unlink(missing_ok=True)
                        continue
                    response.raise_for_status()
                    resume = response.status_code == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("下载返回的断点位置不匹配")
                    received = offset if resume else 0
                    last = 0.0
                    with partial.open("ab" if resume else "wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            check()
                            output.write(chunk)
                            received += len(chunk)
                            if time.monotonic() - last >= 1:
                                progress({"phase": "ASSET_DOWNLOAD", "bytes": received})
                                last = time.monotonic()
                    break
            except requests.RequestException:
                if attempt == 2:
                    raise ValueError("底模下载失败，可重新执行以续传") from None
        if not partial.is_file():
            raise ValueError("底模下载未产生文件")
    if digest(partial, check) != spec["sha256"]:
        partial.unlink(missing_ok=True)
        raise ValueError("底模 SHA256 校验失败")
    partial.replace(path)
    return path


def doctor(settings):
    import psutil

    settings.root.mkdir(parents=True, exist_ok=True)
    result = {"python": sys.version.split()[0], "ram_bytes": psutil.virtual_memory().total,
              "disk_free_bytes": shutil.disk_usage(settings.root).free, "assets": assets(settings),
              "gpu_status": gpu_snapshot()}
    try:
        result["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader"], timeout=10, text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["gpu"] = "nvidia-smi 不可用"
    try:
        with tempfile.TemporaryDirectory(prefix="诊断 音频 ", dir=settings.root) as temp:
            path = Path(temp) / "重采样.wav"
            ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=0.25", "-ar", "44100", path)
            result["ffmpeg"] = probe(path)
            program = (
                "import json,sys,torch; "
                "x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); "
                "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
                "device=torch.cuda.get_device_name(0),bf16=torch.cuda.is_bf16_supported(),"
                "result=(x@x).sum().item(),memory=torch.cuda.mem_get_info()))); "
                "from pathlib import Path; from timbreforge.audio import ffmpeg_dll_directory; "
                "dll=ffmpeg_dll_directory(Path(sys.argv[2])); "
                "import torchaudio; print(torchaudio.load(sys.argv[1])[1])"
            )
            process = subprocess.run([settings.rift_python, "-B", "-c", program, str(path), str(settings.astrbot)],
                                     capture_output=True, timeout=60, text=True)
            lines = process.stdout.splitlines()
            result["cuda"] = json.loads(lines[0]) if lines else {"ready": False}
            result["torch_audio_ready"] = process.returncode == 0 and lines[-1] == "44100"
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        result["diagnostic_error"] = "音频或 CUDA 诊断未完成，请确认 FFmpeg 和 RIFT Python"
    return result
