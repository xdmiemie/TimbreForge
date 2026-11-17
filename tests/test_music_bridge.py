import hashlib
import sys

import pytest

from timbreforge.audio import ffmpeg
from timbreforge.music_bridge import download, load
from timbreforge.settings import ROOT


repo = ROOT.parent / "astrbot-chat"


def test_cdn_request_has_no_login_cookie_and_retries_same_id(tmp_path, monkeypatch):
    if not (repo / "services/media_api/song_source.py").is_file():
        pytest.skip("requires the audited sibling modules")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    _, upstream = load(repo.resolve())
    import requests

    pcm = tmp_path / "fixture.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=1", pcm)
    audio = pcm.read_bytes()
    ids, headers_seen = [], []

    class Response:
        status_code = 200
        headers = {"Content-Type": "audio/wav"}
        is_redirect = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            if len(headers_seen) == 1:
                response = requests.Response()
                response.status_code = 403
                raise requests.HTTPError("expired", response=response)

        def iter_content(self, chunk_size):
            yield audio

    def mock_get(url, **kwargs):
        headers_seen.append(kwargs["headers"])
        assert kwargs["allow_redirects"] is False
        return Response()

    source = upstream.NetEaseSource("MUSIC_U=fixture-cookie")
    monkeypatch.setattr(upstream.requests, "get", mock_get)
    monkeypatch.setattr(source, "playable_url", lambda song_id: ids.append(song_id) or "https://m1.music.126.net/song")
    result = download(source, {"song_id": 9, "duration_seconds": 1}, tmp_path / "selected.audio",
                      upstream.classify_search_error)
    assert ids == [9, 9]
    assert all("cookie" not in {key.casefold() for key in headers} for headers in headers_seen)
    assert result["song_id"] == 9 and result["sha256"] == hashlib.sha256(audio).hexdigest()
    assert "fixture-cookie" not in repr(result)
