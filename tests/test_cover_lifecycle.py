from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from fastapi.testclient import TestClient
import soundfile as sf

from timbreforge.api import create_app
from timbreforge.audio import RATE, atomic_json, digest
from timbreforge import covers
from timbreforge.covers import accept_cover, cleanup, cover_directory, save_cover
from timbreforge.jobs import Conflict, Jobs
from timbreforge.settings import Settings
import pytest


def test_replace_invalidates_temporary_audio_and_keeps_saved(tmp_path):
    app = create_app(Settings(tmp_path))
    store = Jobs(tmp_path)
    first = store.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    with pytest.raises(Conflict):
        store.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    assert store.get(first["id"])["state"] == "QUEUED"
    assert store.claim()["id"] == first["id"]
    area = cover_directory(tmp_path, first["id"])
    area.mkdir(parents=True)
    path = area / "cover.wav"
    sf.write(path, np.ones((RATE, 2), dtype=np.float32) / 10, RATE)
    atomic_json(area / "manifest.json", {"job_id": first["id"], "sha256": digest(path), "samples": RATE})
    store.status(first["id"], "COMPLETED")
    saved = save_cover(store, first["id"], uuid4().hex, "保存作品", False)
    # Use the actual FastAPI route and Range response without a cover GPU worker.
    app.state.jobs = store
    client = TestClient(app)
    audio = f"/api/covers/{first['id']}/audio"
    response = client.get(audio, headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert response.content[:4] == b"RIFF"
    original = audio + "?track=original"
    assert not client.get("/api/covers/current").json()["job"]["result"]["original_available"]
    assert client.get(original).status_code == 404
    source = area / "decoded.wav"
    sf.write(source, np.full((RATE, 2), .2, dtype=np.float32), RATE, subtype="FLOAT")
    assert client.get("/api/covers/current").json()["job"]["result"]["original_available"]
    response = client.get(original, headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert response.content == source.read_bytes()[:100]
    store.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    assert client.get(audio).status_code == 410
    assert client.get(original).status_code == 410
    cleanup(store)
    assert not area.exists()
    saved_url = f"/api/saved-covers/{saved['artifact_id']}/audio"
    assert client.get(saved_url, headers={"Range": "bytes=0-99"}).status_code == 206


def test_save_and_next_submission_serialize_then_gc_retries(tmp_path, monkeypatch):
    store = Jobs(tmp_path)
    original = store.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    store.claim()
    cover_id = original["id"]
    area = cover_directory(tmp_path, cover_id)
    area.mkdir(parents=True)
    path = area / "cover.wav"
    sf.write(path, np.full((RATE, 2), .1, dtype=np.float32), RATE)
    atomic_json(area / "manifest.json", {"job_id": cover_id, "sha256": digest(path), "samples": RATE})
    store.status(cover_id, "COMPLETED")
    copying, resume_copy = Event(), Event()
    copyfile = covers.shutil.copyfile

    def held_copy(*args, **kwargs):
        copying.set()
        if not resume_copy.wait(4):
            raise TimeoutError
        return copyfile(*args, **kwargs)

    monkeypatch.setattr(covers.shutil, "copyfile", held_copy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        saving = executor.submit(save_cover, store, cover_id, uuid4().hex, "永久作品", False)
        try:
            assert copying.wait(3)
            submission = executor.submit(store.submit, "cover", {}, str(uuid4()), on_accept=accept_cover)
            assert not submission.done()
        finally:
            resume_copy.set()
        saved = saving.result(4)
        submission.result(4)
    real_remove = covers.shutil.rmtree

    def sharing_violation(*args, **kwargs):
        raise PermissionError("simulated Windows sharing violation")

    with monkeypatch.context() as patch:
        patch.setattr(covers.shutil, "rmtree", sharing_violation)
        cleanup(store)
        assert area.is_dir()
        with store.connect(write=True) as db:
            row = db.execute("SELECT attempts FROM cover_cleanup WHERE job_id=?", (cover_id,)).fetchone()
            assert row["attempts"] == 1
            db.execute("UPDATE cover_cleanup SET retry_at=0 WHERE job_id=?", (cover_id,))
    assert covers.shutil.rmtree is real_remove
    cleanup(store)
    assert not area.exists()
    assert (tmp_path / "saved_covers" / saved["artifact_id"] / "cover.wav").is_file()


def test_save_mp3_and_play_from_library_after_cover_cleanup(tmp_path):
    jobs = Jobs(tmp_path)
    app = create_app(Settings(tmp_path))
    app.state.jobs = jobs
    client = TestClient(app)
    cover = jobs.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    jobs.claim()
    area = cover_directory(tmp_path, cover["id"])
    area.mkdir(parents=True)
    path = area / "cover.wav"
    tone = .1 * np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE)
    stereo = np.column_stack((tone, tone))
    sf.write(path, stereo, RATE)
    sf.write(area / "converted.wav", tone, RATE)
    sf.write(area / "input-instrumental.wav", stereo, RATE)
    atomic_json(area / "manifest.json", {"job_id": cover["id"], "sha256": digest(path), "samples": RATE})
    jobs.status(cover["id"], "COMPLETED")

    endpoint = f"/api/covers/{cover['id']}/save"
    request = {"request_id": str(uuid4()), "title": "保存为 MP3", "stems": True, "format": "mp3"}
    response = client.post(endpoint, json=request)
    assert response.status_code == 200, response.text
    saved = response.json()
    artifact_id = saved["artifact_id"]
    final = tmp_path / "saved_covers" / artifact_id
    assert saved["format"] == "mp3"
    assert digest(final / "cover.mp3") == saved["sha256"]
    assert not (final / "cover.wav").exists()
    assert client.post(endpoint, json={**request, "format": "wav"}).status_code == 409
    jobs.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    cleanup(jobs)
    assert not area.exists()
    assert client.post(endpoint, json=request).json() == saved
    assert client.get("/api/saved-covers").json()[0]["format"] == "mp3"
    url = f"/api/saved-covers/{artifact_id}/audio"
    response = client.get(url + "?download=true", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert response.headers["content-type"] == "audio/mpeg"
    assert "cover.mp3" in response.headers["content-disposition"]
    assert response.content == (final / "cover.mp3").read_bytes()[:100]
    assert client.get(url + "?track=vocals").headers["content-type"] == "audio/wav"
