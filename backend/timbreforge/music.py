import json
import os
from pathlib import Path
import subprocess

from .worker import JobError


def bridge(settings, action, payload=None):
    if not (settings.astrbot / "services/media_api/song_source.py").is_file():
        raise JobError("未找到相邻 astrbot-chat WeAPI 源码")
    entry = Path(__file__).with_name("music_bridge.py")
    try:
        result = subprocess.run(
            [settings.music_python, "-B", str(entry), str(settings.astrbot), action],
            input=json.dumps(payload or {}, ensure_ascii=False).encode(),
            capture_output=True, timeout=240 if action == "download" else 55,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        response = json.loads(result.stdout)
    except (ValueError, OSError, subprocess.SubprocessError):
        raise JobError("music_bridge_unavailable") from None
    if result.returncode or "error" in response:
        raise JobError(response.get("error", "music_bridge_failed"))
    return response
