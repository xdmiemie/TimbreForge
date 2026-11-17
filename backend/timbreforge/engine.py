import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .audio import atomic_json, ffmpeg_dll_directory
from .environment import lock
from .worker import JobError, check_cancel


def rift_source(settings):
    if not all((settings.rift / name).is_file() for name in ("train.py", "infer.py", "config/finetune.yaml")):
        raise JobError("缺少固定版本的 RIFT-SVC 源码")
    revision = lock()["rift"]["commit"]
    marker = settings.rift / ".timbreforge-revision"
    if marker.is_file():
        if marker.read_text(encoding="ascii").strip() != revision:
            raise JobError("RIFT-SVC 源码版本与清单不符")
    else:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=settings.rift, capture_output=True,
                              text=True, timeout=10)
        if head.returncode != 0 or head.stdout.strip() != revision:
            raise JobError("无法确认 RIFT-SVC 源码版本")
    return settings.rift


def invoke(jobs, job, settings, module, action, payload, *, timeout=None):
    python = settings.separator_python if module == "separator_worker" else settings.rift_python
    cwd = None if module == "separator_worker" else rift_source(settings)
    check_cancel(jobs, job["id"])
    with tempfile.TemporaryDirectory(prefix="engine-", dir=jobs.state) as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        atomic_json(request, payload)
        env = {key: value for key, value in os.environ.items()
               if key not in ("ASTRBOT_CHAT_NETEASE_COOKIE", "ALYSIA_NETEASE_COOKIE")}
        backend = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join([backend] + ([str(cwd)] if cwd else []))
        result = subprocess.run(
            [python, "-B", "-m", f"timbreforge.{module}", action,
             str(jobs.root), job["id"], str(request), str(response)],
            cwd=cwd, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        if not response.is_file():
            raise JobError(f"{action} 工作进程退出（代码 {result.returncode}）")
        data = json.loads(response.read_text(encoding="utf-8"))
        if data.get("cancelled"):
            check_cancel(jobs, job["id"])
        if "error" in data:
            raise JobError(data["error"])
        if result.returncode:
            raise JobError(f"{action} 工作进程失败（代码 {result.returncode}）")
        return data


def worker_entry(operations, *, needs_rift=False):
    import traceback

    from .jobs import Jobs
    from .settings import Settings
    from .supervisor import exclusive
    from .worker import Cancelled

    action, root, job_id, request, response = sys.argv[1:]
    output = Path(response)
    jobs = Jobs(Path(root))
    try:
        if needs_rift and sys.version_info < (3, 11):
            raise JobError("固定 RIFT-SVC 版本要求 Python 3.11 或更新版本")
        spec = json.loads(Path(request).read_text(encoding="utf-8"))
        if action in {"inspect", "export"}:
            with exclusive(jobs.state / "export.lock"):
                value = operations[action](jobs, job_id, spec)
        else:
            with exclusive(jobs.state / "gpu.lock"):
                with ffmpeg_dll_directory(Settings.from_environment().astrbot):
                    value = operations[action](jobs, job_id, spec)
        atomic_json(output, value)
    except Cancelled:
        atomic_json(output, {"cancelled": True})
    except JobError as exc:
        atomic_json(output, {"error": str(exc)})
    except ModuleNotFoundError as exc:
        atomic_json(output, {"error": f"{action} 工作环境缺少依赖 {exc.name}"})
    except Exception as exc:
        if type(exc).__name__ == "OutOfMemoryError":
            atomic_json(output, {"error": f"{action} 显存不足，请检查当前 RTX GPU 占用和运行配置"})
            return
        place = traceback.extract_tb(exc.__traceback__)[-1]
        atomic_json(output, {"error": f"{action}: {type(exc).__name__} ({Path(place.filename).name}:{place.lineno})"})
