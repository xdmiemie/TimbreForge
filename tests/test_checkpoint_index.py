from timbreforge.audio import atomic_json, digest
from timbreforge.environment import lock
from timbreforge.rift_training import validate_resume


def test_recovery_uses_committed_hash_and_ignores_partial_file(tmp_path):
    run = tmp_path / "runs" / ("a" * 32)
    run.mkdir(parents=True)
    old = run / "000000001-00000001.ckpt"
    new = run / "000000002-00000002.ckpt"
    old.write_bytes(b"older complete artifact")
    new.write_bytes(b"candidate checkpoint file")
    (run / "000000003-00000003.ckpt.part").write_bytes(b"uncommitted")
    snapshot = {"fingerprint": "fixture"}
    records = [{**snapshot, "file": path.name, "sha256": digest(path), "commit": lock()["rift"]["commit"]}
               for path in (old, new)]
    atomic_json(run / "checkpoints.json", records)
    new.write_bytes(b"interrupted or corrupt")
    assert validate_resume(tmp_path, run.name, snapshot) == old
