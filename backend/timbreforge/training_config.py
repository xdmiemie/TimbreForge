from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    dataset_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    version_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    architecture: Literal["dit-1024-16"] = "dit-1024-16"
    lora_training: Literal[False] = False
    freeze_adaln_and_tembed: bool = True
    use_base_model: bool = False
    train_batch_size: int = Field(16, ge=1, le=128)
    gradient_accumulation_steps: int = Field(1, ge=1, le=64)
    precision: Literal["bf16-mixed"] = "bf16-mixed"
    validation_batch_size: Literal[1] = 1
    learning_rate: float = Field(2e-5, gt=0, le=0.01)
    weight_decay: float = Field(0.01, ge=0, le=1)
    stop_mode: Literal["steps", "epochs"] = "steps"
    max_train_steps: int | None = Field(30000, ge=1, le=1_000_000)
    max_train_epochs: int | None = Field(None, ge=1, le=10000)
    resume_checkpoint_interval_steps: int = Field(1000, ge=1)
    validation_interval_steps: int = Field(1000, ge=1)
    num_workers: int = Field(4, ge=0, le=8)
    seed: int = Field(42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_target(self):
        if self.stop_mode == "steps" and (self.max_train_steps is None or self.max_train_epochs is not None):
            raise ValueError("步数模式只接受累计步数目标")
        if self.stop_mode == "epochs" and (self.max_train_epochs is None or self.max_train_steps is not None):
            raise ValueError("轮次模式只接受累计轮次目标")
        return self


class TrainSubmission(TrainConfig):
    request_id: UUID
