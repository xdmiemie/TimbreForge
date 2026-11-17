from types import SimpleNamespace

import pytest

from timbreforge.audio import digest, ffmpeg, probe
from timbreforge.datasets import process


def test_dataset_reads_mixed_inputs_without_mutating_source(tmp_path):
    source = tmp_path / "角色 原音"
    source.mkdir()
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=2", "-ac", 2, source / "正常.wav")
    ffmpeg("-f", "lavfi", "-i", "sine=f=620:d=1.5", source / "长音.flac")
    ffmpeg("-f", "lavfi", "-i", "anullsrc", "-t", 1, source / "全静音.wav")
    (source / "副本.wav").write_bytes((source / "正常.wav").read_bytes())
    (source / "损坏.mp3").write_bytes(b"broken")
    before = {p.name: digest(p) for p in source.iterdir()}
    jobs = SimpleNamespace(root=tmp_path / "data", cancelled=lambda _: False, progress=lambda *_: None)
    payload = {"source": str(source), "recursive": True, "role": "测试", "language": "zh"}
    result = process(jobs, {"id": "a" * 32, "kind": "dataset_prepare", "payload": payload})
    assert {"duplicate", "cannot_decode", "silent_or_too_short"} == {
        row["reason"] for row in result["sources"] if "reason" in row}
    assert result["stats"]["retained_seconds"] == pytest.approx(3.5)
    for clip in result["clips"]:
        path = jobs.root / "datasets" / result["dataset_id"] / result["version_id"] / "speaker" / (clip["file_name"] + ".wav")
        assert probe(path)["seconds"] == pytest.approx(clip["seconds"])
        assert digest(path) == clip["sha256"]
    assert {clip["split"] for clip in result["clips"]} == {"train", "validation"}
    excluded = process(jobs, {"id": "b" * 32, "kind": "dataset_scan",
                              "payload": {**payload, "excluded_paths": ["长音.flac"]}})
    rejected = next(row for row in excluded["sources"] if row["path"] == "长音.flac")
    assert rejected["reason"] == "user_excluded" and rejected["retained_seconds"] == 0
    assert {p.name: digest(p) for p in source.iterdir()} == before
