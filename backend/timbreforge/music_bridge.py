"""Isolated JSON bridge to the sibling project's current authenticated WeAPI."""

import json
from http.cookies import SimpleCookie
from pathlib import Path
import re
import subprocess
import sys
import time


def load(repo):
    from importlib import import_module

    sys.path.insert(0, str(repo))
    modules = [import_module("services.media_api." + name) for name in ("netease_auth", "song_source")]
    for module, name in zip(modules, ("netease_auth", "song_source")):
        expected = repo / "services" / "media_api" / (name + ".py")
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError("unexpected bridge import")
    return modules


def download(source, song, destination, classify):
    temporary = destination.with_suffix(".candidate")
    for attempt in range(3):
        try:
            sha = source.download(song["song_id"], temporary)
            # Decode the whole stream and count samples to reject previews and partial files.
            probe = subprocess.run(
                ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(temporary), "-vn", "-sn",
                 "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"], capture_output=True, timeout=120,
            )
            seconds = len(probe.stdout) / 16000
            if probe.returncode or seconds <= 0 or abs(seconds - song["duration_seconds"]) > 2:
                return {"error": "source_is_not_full_song"}
            temporary.replace(destination)
            return {"song_id": song["song_id"], "sha256": sha, "seconds": seconds}
        except PermissionError:
            return {"error": "source_full_track_unavailable"}
        except Exception as exc:
            info = classify(exc)
            if info is None:
                return {"error": "source_download_failed"}
            if not (info.retryable or info.code == "source_http_403") or attempt == 2:
                return {"error": info.code}
            time.sleep(0.5 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
    return {"error": "source_download_failed"}


def run(repo, action, spec):
    import os

    auth, music = load(repo)
    override = os.environ.get("ASTRBOT_CHAT_NETEASE_COOKIE") or os.environ.get("ALYSIA_NETEASE_COOKIE")
    source_type = "environment" if override else "dpapi"
    try:
        cookie = auth.resolve_cookie()
    except (auth.LoginError, OSError):
        return {"error": "netease_session_unreadable"}
    parsed = SimpleCookie()
    parsed.load(cookie)
    if not parsed.get("MUSIC_U") or not parsed["MUSIC_U"].value:
        if action in {"status", "account"}:
            return {"configured": False, "source": source_type}
        return {"error": "netease_login_required"}
    if action == "status":
        return {"configured": True, "source": source_type, "online_verified": False}
    if action == "account":
        login = auth.NetEaseLogin(cookie)
        try:
            login.account()
        except auth.LoginError:
            return {"error": "netease_login_invalid"}
        finally:
            login.close()
        return {"configured": True, "source": source_type, "online_verified": True}
    source = music.NetEaseSource(cookie)
    try:
        if action == "search":
            results = source.search(spec["query"], spec["page"])
            return {"candidates": [item.public() for item in results], "quality": "standard/mp3"}
        if action == "download":
            return download(source, spec["song"], Path(spec["destination"]), music.classify_search_error)
        return {"error": "unsupported_music_operation"}
    except auth.LoginError:
        return {"error": "netease_login_invalid"}
    except Exception as exc:
        info = music.classify_search_error(exc)
        code = info.code if info else "source_request_failed"
        return {"error": code if re.fullmatch(r"[a-z0-9_]+", code or "") else "source_request_failed"}
    finally:
        source.http.close()


if __name__ == "__main__":
    try:
        result = run(Path(sys.argv[1]).resolve(), sys.argv[2], json.load(sys.stdin.buffer))
    except Exception:
        result = {"error": "music_bridge_unavailable"}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
