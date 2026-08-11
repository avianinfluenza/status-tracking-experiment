import os
import sys
import json
import yaml
import time
import random
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _format_float(x: Any) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return str(x)


def make_run_id(cfg: Dict[str, Any]) -> str:
    """
    Example:

    20260810-204400__looped__rec=basic__d=128__bs=64__lr=1e-4__seed=42

    또는

    20260810-204400__basic__layers=4__d=128__bs=64__lr=1e-4__seed=42
    """
    ts = _now_run_id()

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    model_name = cfg.get("run_name") or model_cfg.get("name", "model")

    d_model = model_cfg.get("d_model", "NA")
    batch_size = train_cfg.get("batch_size", "NA")
    lr = train_cfg.get("learning_rate", "NA")
    seed = train_cfg.get("seed", "NA")

    parts = [
        ts,
        model_name,
    ]

    # BasicTransformer / LoopedTransformer 구분용
    if model_name.lower() in {"looped", "looped_transformer"}:
        recurrence_type = model_cfg.get("recurrence_type", "basic")
        parts.append(f"rec={recurrence_type}")

        if "n_recurrence" in model_cfg:
            parts.append(f"T={model_cfg['n_recurrence']}")

    elif model_name.lower() in {"basic", "basic_transformer"}:
        n_layers = model_cfg.get("n_transformer_layer", "NA")
        parts.append(f"layers={n_layers}")

    parts.extend(
        [
            f"d={d_model}",
            f"bs={batch_size}",
            f"lr={_format_float(lr)}",
            f"seed={seed}",
        ]
    )

    return "__".join(map(str, parts))


def ensure_run_dir(base_dir: str, run_id: str) -> str:
    run_dir = os.path.join(base_dir, run_id)

    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    return run_dir


def _get_git_info() -> str:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )

        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )

        dirty = "dirty" if status else "clean"

        return f"commit: {commit}\n" f"status: {dirty}\n\n" f"porcelain:\n{status}\n"

    except Exception as e:
        return f"git info unavailable: {e}\n"


def _get_command_line() -> str:
    return " ".join([sys.executable] + sys.argv)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_yaml(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class CheckpointPolicy:
    save_ckpt: bool = True
    save_every: int = 0

    save_best: bool = True

    # 이 프로젝트의 핵심 metric:
    # 모든 사람의 상태를 전부 맞춘 sample 비율
    monitor: str = "valid_state_acc"
    mode: str = "max"

    keep_last: bool = True


class ExperimentLogger:
    """
    State-tracking Transformer experiment logger.

    Saves:
    - config.final.yaml
    - command.txt
    - git.txt
    - metrics.jsonl
    - checkpoints/
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        run_dir: str,
    ):
        self.cfg = cfg
        self.run_dir = run_dir

        self.metrics_path = os.path.join(
            run_dir,
            "metrics.jsonl",
        )

        self.ckpt_dir = os.path.join(
            run_dir,
            "checkpoints",
        )

        train_cfg = cfg.get("train", {})

        self.policy = CheckpointPolicy(
            save_ckpt=bool(train_cfg.get("save_ckpt", True)),
            save_every=int(train_cfg.get("save_every", 0)),
            save_best=bool(train_cfg.get("save_best", True)),
            monitor=str(
                train_cfg.get(
                    "monitor",
                    "valid_state_acc",
                )
            ),
            mode=str(
                train_cfg.get(
                    "monitor_mode",
                    "max",
                )
            ).lower(),
            keep_last=bool(train_cfg.get("keep_last", True)),
        )

        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

        # 학습이 죽어도 실험 조건은 남도록 즉시 저장
        save_yaml(
            os.path.join(
                run_dir,
                "config.final.yaml",
            ),
            cfg,
        )

        save_text(
            os.path.join(
                run_dir,
                "command.txt",
            ),
            _get_command_line() + "\n",
        )

        save_text(
            os.path.join(
                run_dir,
                "git.txt",
            ),
            _get_git_info(),
        )

    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        권장 metrics:

        {
            "train_loss": ...,
            "train_slot_acc": ...,
            "train_state_acc": ...,

            "valid_loss": ...,
            "valid_slot_acc": ...,
            "valid_state_acc": ...
        }

        OOD 평가 시:
        {
            "ood_swap_20_state_acc": ...,
            "ood_swap_40_state_acc": ...,
            ...
        }
        """

        record = {
            "epoch": epoch,
            **metrics,
        }

        append_jsonl(
            self.metrics_path,
            record,
        )

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.policy.mode == "min":
            return value < self.best_value

        if self.policy.mode == "max":
            return value > self.best_value

        raise ValueError(f"Unknown monitor mode: {self.policy.mode}")

    def _make_state(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "metrics": metrics,
            "cfg": self.cfg,
            # best 정보도 checkpoint에 직접 저장
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            # reproducibility
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        if extra:
            state.update(extra)

        return state

    def maybe_save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not self.policy.save_ckpt:
            return

        # best 여부를 checkpoint 생성 전에 먼저 판단
        improved = False

        if self.policy.save_best and self.policy.monitor in metrics:
            try:
                cur = float(metrics[self.policy.monitor])

                if self._is_better(cur):
                    improved = True
                    self.best_value = cur
                    self.best_epoch = epoch

            except (TypeError, ValueError):
                pass

        state = self._make_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            extra=extra,
        )

        # 마지막 epoch
        if self.policy.keep_last:
            torch.save(
                state,
                os.path.join(
                    self.ckpt_dir,
                    "last.pt",
                ),
            )

        # periodic
        if self.policy.save_every > 0 and epoch % self.policy.save_every == 0:
            torch.save(
                state,
                os.path.join(
                    self.ckpt_dir,
                    f"epoch_{epoch}.pt",
                ),
            )

        # best
        if improved:
            torch.save(
                state,
                os.path.join(
                    self.ckpt_dir,
                    "best.pt",
                ),
            )

            save_text(
                os.path.join(
                    self.run_dir,
                    "best.txt",
                ),
                (
                    f"best_epoch: {epoch}\n"
                    f"{self.policy.monitor}: "
                    f"{self.best_value}\n"
                ),
            )

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ckpt_name: str = "last.pt",
        map_location: Optional[str] = "cpu",
    ) -> Dict[str, Any]:

        ckpt_path = os.path.join(
            self.ckpt_dir,
            ckpt_name,
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(
            ckpt_path,
            map_location=map_location,
        )

        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        # 진짜 historical best 복원
        self.best_value = state.get(
            "best_value",
            None,
        )

        self.best_epoch = state.get(
            "best_epoch",
            None,
        )

        return state

    def resume_if_possible(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        prefer: str = "last",
        map_location: Optional[str] = "cpu",
    ) -> int:

        if prefer not in {"last", "best"}:
            raise ValueError("prefer must be 'last' or 'best'")

        ckpt_name = f"{prefer}.pt"

        ckpt_path = os.path.join(
            self.ckpt_dir,
            ckpt_name,
        )

        if not os.path.exists(ckpt_path):
            return 0

        state = self.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_name=ckpt_name,
            map_location=map_location,
        )

        last_epoch = int(state.get("epoch", -1))

        return last_epoch + 1
