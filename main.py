import os
import yaml
import argparse
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)  # config file location
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def override_cfg(cfg: dict, args) -> dict:
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    return cfg


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = override_cfg(cfg, args)
    # trainer.py로 넘어가기
