import json
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from timbreforge import augmentation
from timbreforge.api import create_app
from timbreforge.audio import RATE, atomic_json
from timbreforge.covers import accept_cover, cleanup, cover_directory
from timbreforge.jobs import Jobs
from timbreforge.library import cleanup_deleted
from timbreforge.settings import Settings


def setup_cover(root):
    jobs = Jobs(root)
    app = create_app(Settings(root))
    app.state.jobs = jobs
    model_id, version_id, dataset_version = "a" * 24, "b" * 32, "c" * 32
    model_dir = root / "models" / "characters" / model_id / version_id
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    model = {"model_id": model_id, "version_id": version_id, "source": "final", "sha256": "hash",
             "checkpoint": str(checkpoint), "architecture": "dit-1024-16", "speakers": ["speaker"]}
    atomic_json(model_dir / "manifest.json", model)
    atomic_json(model_dir / "training_summary.json", {"dataset_id": model_id, "version_id": dataset_version})
    source = root / "original-source"
    source.mkdir()
    (source / "source.wav").write_bytes(b"original")
    directory = root / "datasets" / model_id / dataset_version
    (directory / "speaker").mkdir(parents=True)
    manifest = {"dataset_id": model_id, "version_id": dataset_version, "role": "测试", "language": "zh",
                "created": 0, "fingerprint": "original", "sources": [], "config": {}, "source": str(source),
                "stats": {"train_clips": 1, "validation_clips": 1, "decoded_seconds": 2,
                          "retained_seconds": 2, "excluded_seconds": 0},
                "clips": [{"file_name": split, "group": split, "split": split} for split in ("train", "validation")]}
    for clip in manifest["clips"]:
        (directory / "speaker" / (clip["file_name"] + ".wav")).write_bytes(b"existing")
    atomic_json(directory / "manifest.json", manifest)
    cover = jobs.submit("cover", {"model": model}, str(uuid4()), on_accept=accept_cover)
    jobs.claim()
    area = cover_directory(root, cover["id"])
    area.mkdir(parents=True)
    # The mix is silent; the converted track contains three distinct voiced intervals.
    phrase = (0.2 * np.sin(np.arange(2 * RATE) * 2 * np.pi * 220 / RATE)).astype("float32")
    vocals = np.tile(np.concatenate([phrase, np.zeros(RATE)]), 3)
    sf.write(area / "converted.wav", vocals, RATE, subtype="FLOAT")
    sf.write(area / "cover.wav", np.zeros((len(vocals), 2)), RATE)
    atomic_json(area / "manifest.json", {"samples": len(vocals)})
    jobs.status(cover["id"], "COMPLETED")
    return jobs, TestClient(app), area, directory, model_dir


def test_clip_uses_preserved_vocals_and_publishes_only_after_features(tmp_path, monkeypatch):
    jobs, client, area, original, model_dir = setup_cover(tmp_path)
    endpoint = f"/api/covers/{area.name}"
    assert client.get(endpoint + "/audio?track=vocals", headers={"Range": "bytes=0-7"}).status_code == 206
    assert client.get("/api/covers/current").json()["job"]["result"]["training_dataset"]["role"] == "测试"
    body = {"request_id": str(uuid4()), "first_seconds": 9, "second_seconds": 0}
    assert client.post(endpoint + "/dataset-fragment", json={**body, "first_seconds": 10}).status_code == 422
    queued = client.post(endpoint + "/dataset-fragment", json=body)
    assert queued.status_code == 202
    job = jobs.claim()
    assert job["id"] == queued.json()["id"]
    assert client.delete(f"/api/datasets/{original.parent.name}/{original.name}").status_code == 409
    assert client.delete(f"/api/models/{model_dir.parent.name}/{model_dir.name}").status_code == 409
    jobs.submit("cover", {}, str(uuid4()), on_accept=accept_cover)
    cleanup(jobs)
    assert area.is_dir()

    final = original.with_name(job["id"])

    def simulated_features(jobs, job_id, spec, *, staged, parent):
        scratch, manifest = staged
        assert not final.exists()
        assert parent[0] == original
        assert [clip["split"] for clip in manifest["clips"]] == ["train", "validation", "train", "train", "train"]
        atomic_json(scratch / "features.json", {"key": "test"})

    monkeypatch.setattr(augmentation, "prepare_features", simulated_features)
    result = augmentation.append_range(jobs, job["id"], job["payload"])
    assert (result["added_clips"], result["previous_train_clips"], result["train_clips"]) == (3, 1, 4)
    stored = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert stored["ancestors"] == [original.name]
    assert augmentation.append_parent(tmp_path, job["payload"]["dataset"])[0] == final
    jobs.status(job["id"], "COMPLETED", result=result)
    cleanup(jobs)
    assert not area.exists()
    training = client.post("/api/training", json={"request_id": str(uuid4()), "train_batch_size": 1,
                                                 "dataset_id": original.parent.name, "version_id": final.name})
    assert training.status_code == 202
    assert training.json()["payload"]["initial_model"]["checkpoint"] == str(model_dir / "model.ckpt")
    assert client.post(f'/api/jobs/{training.json()["id"]}/control',
                       json={"request_id": str(uuid4()), "action": "cancel"}).status_code == 202
    assert client.delete(f"/api/datasets/{original.parent.name}/{original.name}").status_code == 200
    assert augmentation.append_parent(tmp_path, job["payload"]["dataset"])[0] == final
    assert client.delete(f"/api/models/{model_dir.parent.name}/{model_dir.name}").status_code == 200
    cleanup_deleted(jobs)
    assert not original.exists() and not model_dir.exists()
    assert (final / "speaker" / "validation.wav").read_bytes() == b"existing"
    assert (tmp_path / "original-source" / "source.wav").read_bytes() == b"original"


def test_saved_delete_targets_only_requested_registered_works(tmp_path):
    jobs = Jobs(tmp_path)
    app = create_app(Settings(tmp_path))
    app.state.jobs = jobs
    client = TestClient(app)
    first, second, unregistered = [uuid4().hex for _ in range(3)]
    for artifact_id in (first, second, unregistered):
        area = tmp_path / "saved_covers" / artifact_id
        area.mkdir(parents=True)
        (area / "cover.wav").write_bytes(b"audio")
    with jobs.connect(write=True) as db:
        for item in (first, second):
            db.execute("INSERT INTO saved_covers VALUES(?,?)", (item, json.dumps(
                {"title": "作品", "song": {"title": "原曲"}, "model": {"model_id": "test"}, "stems": True})))
    assert {(row["song_title"], row["format"]) for row in client.get("/api/saved-covers").json()} == {("原曲", "wav")}
    assert client.post("/api/saved-covers/bulk-delete", json={"ids": [first, first, unregistered]}).json() == {"deleted": 1}
    assert client.get(f"/api/saved-covers/{first}/audio").status_code == 404
    assert [row["artifact_id"] for row in client.get("/api/saved-covers").json()] == [second]
    assert client.delete(f"/api/saved-covers/{second}").json() == {"deleted": 1}
    cleanup_deleted(jobs)
    assert (tmp_path / "saved_covers" / unregistered / "cover.wav").is_file()
