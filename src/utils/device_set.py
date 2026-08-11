import torch
import os


def cuda_device_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def mps_device_set():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _device_set():
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")

    print(f"Using device: {device}")

    return device


def device_set(device_name=None):
    if device_name:
        try:
            device = torch.device(device_name)
            print(f"Using manually specified device: {device}")
            return device
        except Exception as e:
            print(f"Invalid device name '{device_name}': {e}")
            print("Automatically detecting device...")
            return _device_set()
    else:
        return _device_set()
