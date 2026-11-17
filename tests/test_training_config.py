import pytest

from timbreforge.audio import atomic_json
from timbreforge.features import train_preflight
from timbreforge.inference import InferenceConfig
from timbreforge.training_config import TrainConfig
from timbreforge.worker import JobError


def test_defaults_and_incomplete_batches(tmp_path):
    flags = TrainConfig(dataset_id="a" * 24, version_id="b" * 32)
    assert flags.train_batch_size == 16 and flags.architecture == "dit-1024-16"
    assert flags.lora_training is False and InferenceConfig().inference_steps == 64
    area = tmp_path / "datasets" / flags.dataset_id / flags.version_id
    area.mkdir(parents=True)
    settings = {"fingerprint": "asset-fingerprint", "stats": {"train_clips": 15, "validation_clips": 2}}
    atomic_json(area / "manifest.json", settings)
    with pytest.raises(JobError, match="物理 batch 16"):
        train_preflight(tmp_path, flags.model_dump())
    settings["stats"]["train_clips"] = 16
    atomic_json(area / "manifest.json", settings)
    assert train_preflight(tmp_path, flags.model_dump())["train_batches"] == 1
