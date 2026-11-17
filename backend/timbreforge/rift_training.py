import ast
import hashlib
import json
import inspect
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

from .audio import atomic_json, digest
from .environment import lock
from .features import dataset, feature_name, train_preflight
from .rift_worker import require_cuda
from .worker import JobError, check_cancel


def checkpoint_index(root, run_id):
    path = root / "runs" / run_id / "checkpoints.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def validate_resume(root, run_id, expected, selected=None):
    points = checkpoint_index(root, run_id)
    for entry in reversed(points):
        if selected is not None and entry["file"] != selected:
            continue
        path = root / "runs" / run_id / entry["file"]
        if (path.is_file() and entry["fingerprint"] == expected["fingerprint"]
                and entry["commit"] == lock()["rift"]["commit"] and digest(path) == entry["sha256"]):
            return path
    raise JobError("没有可校验的完整恢复点")


def feature_dataset(root, spec, expected_digest=None):
    directory, source = dataset(root, spec)
    paths = directory / "features.json"
    if not paths.is_file():
        raise JobError("请先准备该版本的 RIFT 特征")
    features = json.loads(paths.read_text(encoding="utf-8"))
    if features["commit"] != lock()["rift"]["commit"] or features["dataset"] != source["fingerprint"]:
        raise JobError("RIFT 特征版本已失效")
    if expected_digest and features.get("content_sha256") != expected_digest:
        raise JobError("冻结的训练特征内容发生变化")
    data_dir = directory / "features" / features["key"]
    all_hashes = {}
    for index, _ in enumerate(source["clips"], 1):
        name = feature_name(index)
        row = json.loads((data_dir / "speaker" / f"{name}.json").read_text(encoding="utf-8"))
        for suffix in ("mel", "rms", "f0", "cvec"):
            file = f"{name}.{suffix}.pt"
            if digest(data_dir / "speaker" / file) != row.get(file):
                raise JobError("训练特征损坏，请重新运行特征准备")
            all_hashes[file] = row[file]
    if expected_digest and hashlib.sha256(json.dumps(all_hashes, sort_keys=True).encode()).hexdigest() != expected_digest:
        raise JobError("训练特征内容与已完成的数据版本不一致")
    return data_dir, source


def train(jobs, job_id, spec):
    torch = require_cuda()
    if not torch.cuda.is_bf16_supported():
        raise JobError("当前 CUDA 设备不支持 BF16")
    import numpy as np
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader
    from rift_svc import DiT, RF
    from rift_svc.dataset import SVCDataset, collate_fn
    from rift_svc.lightning_module import RIFTSVCLightningModule
    from rift_svc.optim import get_optimizer

    flags = spec["config"]
    train_preflight(jobs.root, flags)
    data_dir, source = feature_dataset(jobs.root, flags, spec.get("feature_digest"))
    if source["fingerprint"] != spec["fingerprint"]:
        raise JobError("数据版本指纹发生变化")
    if data_dir.name != spec["feature_key"]:
        raise JobError("训练所绑定的特征版本发生变化")
    run_id = spec.get("run_id", job_id)
    run = jobs.root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    source_run = spec.get("parent_run_id") or spec.get("run_id")
    resume = validate_resume(jobs.root, source_run, spec, spec.get("checkpoint_file")) if source_run else None
    base = lock()["assets"]["base"]
    initial = spec.get("initial_model")
    base_path = Path(initial["checkpoint"]) if initial else jobs.root / "assets" / base["path"]
    expected_sha = initial["sha256"] if initial else base["sha256"]
    if resume is None and (not base_path.is_file() or digest(base_path) != expected_sha):
        raise JobError("来源模型已删除或发生变化" if initial else "缺少 SHA256 校验通过的官方 1024-16 底模")
    pl.seed_everything(flags["seed"], workers=True)
    torch.set_float32_matmul_precision("high")
    with initialize_config_dir(config_dir=str(Path("config").resolve()), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name="finetune", overrides=["model=dit-1024-16"]), resolve=True)
    cfg["dataset"].update(data_dir=str(data_dir), meta_info_path=str(data_dir / "meta_info.json"))
    train_set = SVCDataset(**cfg["dataset"], split="train")
    val_set = SVCDataset(**cfg["dataset"], split="test")
    worker_count = flags["num_workers"]
    batch_size = flags["train_batch_size"]
    accum = flags["gradient_accumulation_steps"]
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=worker_count, persistent_workers=bool(worker_count), collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, collate_fn=collate_fn)
    if not len(loader) or not len(val_loader):
        raise JobError(f"DataLoader 为空，物理 batch {batch_size}，训练片段 {len(train_set)}")
    epochs = flags["max_train_epochs"]
    target = flags["max_train_steps"] or epochs * math.ceil(len(loader) / accum)
    training = cfg["training"]
    training.update(max_steps=target, lora_training=False, logger="tensorboard", save_weights_only=False,
                    batch_size_per_gpu=batch_size, grad_accumulation_steps=accum, num_workers=worker_count,
                    learning_rate=flags["learning_rate"], weight_decay=flags["weight_decay"], drop_spk_prob=0.0)
    cfg["spk2idx"] = train_set.spk2idx
    cfg["seed"] = flags["seed"]
    rf = RF(transformer=DiT(**cfg["model"], num_speaker=train_set.num_speakers), time_schedule=training["time_schedule"])
    if resume is None:
        stored = torch.load(base_path, weights_only=True, map_location="cpu", mmap=bool(initial))
        stored = stored.get("state_dict", stored)
        state = {key.removeprefix("model."): value for key, value in stored.items()}
        own = rf.state_dict()
        speaker = {key for key in own if "spk" in key or "speaker" in key}
        filtered = {key: value for key, value in state.items()
                    if key not in speaker or (key in own and own[key].shape == value.shape)}
        load = rf.load_state_dict(filtered, strict=initial is not None)
        if set(load.missing_keys) - speaker or load.unexpected_keys:
            raise JobError("初始模型参数与锁定的 1024-16 架构不匹配")
        del stored, state, own, filtered
    if flags["freeze_adaln_and_tembed"]:
        rf.transformer.freeze_adaln_and_tembed()
    optimizer, scheduler = get_optimizer(
        training["optimizer_type"], rf, training["learning_rate"], ast.literal_eval(training["betas"]),
        training["weight_decay"], int(target * training["warmup_ratio"]), max_steps=target,
        min_lr=training.get("min_lr", 0), lora_training=False,
    )

    class Module(RIFTSVCLightningModule):
        def validation_step(self, batch, batch_idx):
            loss, _ = self.model(batch["mel"], spk_id=batch["spk_id"], f0=batch["f0"],
                                 rms=batch["rms"], cvec=batch["cvec"], frame_len=batch["frame_len"])
            self.log("val/loss", loss, on_epoch=True, batch_size=1)

        def on_validation_start(self):
            torch.cuda.reset_peak_memory_stats()
            if hasattr(optimizer, "eval"):
                optimizer.eval()

        def on_validation_end(self):
            if hasattr(optimizer, "train"):
                optimizer.train()
            value = self.trainer.callback_metrics.get("val/loss")
            jobs.progress(job_id, {"phase": "VALIDATING", "step": self.global_step,
                                   "loss": float(value) if value is not None and torch.isfinite(value) else None,
                                   "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
            torch.cuda.reset_peak_memory_stats()

        def on_save_checkpoint(self, state):
            state["timbreforge"] = {"fingerprint": source["fingerprint"], "commit": lock()["rift"]["commit"],
                                   "run_id": run_id, "rng_cpu": torch.get_rng_state(),
                                   "rng_gpu": torch.cuda.get_rng_state(), "rng_python": random.getstate(),
                                   "rng_numpy": np.random.get_state()[1].tolist(),
                                   "rng_numpy_info": list(np.random.get_state()[2:])}

        def on_load_checkpoint(self, state):
            rng = state["timbreforge"]
            if rng["fingerprint"] != source["fingerprint"] or rng["run_id"] != source_run:
                raise JobError("恢复点与本次运行不匹配")
            torch.set_rng_state(rng["rng_cpu"].cpu())
            torch.cuda.set_rng_state(rng["rng_gpu"].cpu())
            random.setstate(rng["rng_python"])
            np.random.set_state(("MT19937", np.array(rng["rng_numpy"], dtype=np.uint32), *rng["rng_numpy_info"]))

    resume_step = checkpoint_index(jobs.root, source_run) if resume else []
    initial = next((point["step"] for point in resume_step if resume and point["file"] == resume.name), 0)
    started = time.monotonic()

    class Control(pl.Callback):
        step = initial
        outcome = "COMPLETED"
        export_process = None
        export_response = None
        export_status = "NOT_RUN"

        def on_train_start(self, trainer, module):
            if hasattr(optimizer, "train"):
                optimizer.train()
            torch.cuda.reset_peak_memory_stats()

        def save(self, trainer, reason):
            step = trainer.global_step
            points = checkpoint_index(jobs.root, run_id)
            last = run / points[-1]["file"] if points else None
            needed = max(last.stat().st_size * 2 if last and last.is_file() else 0, 5 * 1024**3)
            if shutil.disk_usage(run).free < needed:
                raise JobError("磁盘余量不足以安全写入完整恢复点")
            file = f"{step:09d}-{uuid.uuid4().hex[:8]}.ckpt"
            final = run / file
            temp = run / (file + ".part")
            try:
                torch.cuda.synchronize()
                if hasattr(optimizer, "eval"):
                    optimizer.eval()
                trainer.save_checkpoint(temp, weights_only=False)
                with temp.open("rb+") as stream:
                    import os
                    os.fsync(stream.fileno())
                sha = digest(temp)
                temp.replace(final)
            finally:
                if hasattr(optimizer, "train"):
                    optimizer.train()
                temp.unlink(missing_ok=True)
            entry = {"step": step, "epoch": trainer.current_epoch, "file": file, "sha256": sha,
                     "fingerprint": source["fingerprint"], "commit": lock()["rift"]["commit"], "reason": reason}
            points.append(entry)
            discard = [point for point in points if point["reason"] == "periodic"][:-3]
            atomic_json(run / "checkpoints.json", [point for point in points if point not in discard])
            for point in discard:
                (run / point["file"]).unlink(missing_ok=True)
            jobs.progress(job_id, {"phase": "CHECKPOINT", **entry})
            return entry

        def close_export(self, wait=False):
            if self.export_process is None:
                return True
            if self.export_process.poll() is None and not wait:
                return False
            self.export_process.wait()
            response = json.loads(self.export_response.read_text(encoding="utf-8")) if self.export_response.is_file() else {}
            if not response or "error" in response:
                self.export_status = "FAILED"
                jobs.progress(job_id, {"phase": "EXPORT_FAILED", "message": response.get("error", "导出工作进程退出")})
            else:
                self.export_status = "PUBLISHED"
            self.export_process = None
            return True

        def export(self, trainer, entry, reason, version_id):
            self.close_export(wait=True)
            export_spec = {"snapshot": str(run / entry["file"]), "snapshot_sha256": entry["sha256"],
                           "version_id": version_id,
                           "summary": {"run_id": run_id, "dataset_id": flags["dataset_id"],
                                       "version_id": flags["version_id"], "role": source["role"],
                                       "language": source["language"], "step": trainer.global_step,
                                       "epoch": trainer.current_epoch, "reason": reason, "config": flags,
                                       "initial_model": initial}}
            request = run / f"export-{version_id}.json"
            atomic_json(request, export_spec)
            self.export_response = run / f"publication-{version_id}.json"
            try:
                self.export_process = subprocess.Popen(
                    [sys.executable, "-B", "-m", "timbreforge.rift_worker", "export", str(jobs.root),
                     job_id, str(request), str(self.export_response)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self.export_status = "RUNNING"
            except OSError:
                self.export_status = "FAILED"
                jobs.progress(job_id, {"phase": "EXPORT_FAILED", "message": "未启动导出工作进程", "version": version_id})

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_index):
            step = trainer.global_step
            if step <= self.step:
                return
            self.step = step
            elapsed = time.monotonic() - started
            loss = float(outputs["loss"].detach())
            if not math.isfinite(loss):
                raise JobError("训练损失为非有限值")
            jobs.progress(job_id, {"phase": "TRAINING", "step": step, "target": target,
                                   "epoch": trainer.current_epoch, "batch_index": batch_index, "loss": loss,
                                   "physical_batch": int(batch["spk_id"].shape[0]),
                                   "optimizer_updates": optimizer.param_groups[0].get("k"),
                                   "learning_rate": optimizer.param_groups[0]["lr"],
                                   "elapsed_seconds": elapsed, "eta_seconds": (target-step)*elapsed/(step-initial),
                                   "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
            check_cancel(jobs, job_id)
            cmd = jobs.pending_control(job_id)
            self.close_export()
            if cmd and step < target and (cmd["action"] != "export" or self.export_process is None):
                action = cmd["action"]
                entry = self.save(trainer, action)
                if action in {"export", "stop_export"}:
                    self.export(trainer, entry, action, uuid.UUID(cmd["request"]).hex)
                if action in {"pause", "stop_export"}:
                    self.outcome = "PAUSED" if action == "pause" else "EARLY_COMPLETED"
                    trainer.should_stop = True
                jobs.complete_control(job_id, cmd["request"])
            elif step % flags["resume_checkpoint_interval_steps"] == 0 and step < target:
                self.save(trainer, "periodic")

    module = Module(model=rf, optimizer=optimizer, lr_scheduler=scheduler, cfg=cfg)
    callbacks = Control()
    trainer = pl.Trainer(
        accelerator="gpu", devices=[0], precision="bf16-mixed", max_steps=target, max_epochs=epochs if epochs else -1,
        accumulate_grad_batches=accum, val_check_interval=1,
        check_val_every_n_epoch=None, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, callbacks=[callbacks], logger=TensorBoardLogger(str(run), name="tensorboard"),
    )
    last_validation = initial

    def validate_at_update(_data_fetcher):
        nonlocal last_validation
        step = trainer.global_step
        should_validate = (step > last_validation and not trainer.should_stop
                           and (step == target or step % flags["validation_interval_steps"] == 0))
        if should_validate:
            last_validation = step
        return should_validate

    # Lightning 2.5.5 normally schedules this hook in batches, including partial accumulation cycles.
    trainer.fit_loop.epoch_loop._should_check_val_fx = validate_at_update
    atomic_json(run / "config.json", {**spec, "resolved": cfg, "fingerprint": source["fingerprint"],
                                     "run_id": run_id, "target": target, "upstream": lock()["rift"]["commit"]})
    jobs.progress(job_id, {"phase": "PREPARING", "target": target, "physical_batch": batch_size,
                           "validation_batch": 1, "optimizer": training["optimizer_type"], "train_batches": len(loader),
                           "cuda_free_bytes": torch.cuda.mem_get_info()[0]})
    fit_options = {"weights_only": True} if "weights_only" in inspect.signature(trainer.fit).parameters else {}
    trainer.fit(module, loader, val_loader, ckpt_path=resume, **fit_options)
    if callbacks.outcome == "COMPLETED":
        entry = callbacks.save(trainer, "final")
        callbacks.export(trainer, entry, "final", uuid.uuid4().hex)
    callbacks.close_export(wait=True)
    return {"outcome": callbacks.outcome, "run_id": run_id, "step": trainer.global_step,
            "target": target, "export_status": callbacks.export_status,
            "checkpoints": checkpoint_index(jobs.root, run_id)}
