import yaml
import torch
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_checkpoint(state, is_best: bool, checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    torch.save(state, checkpoint_dir / "last_checkpoint.pth")
    if is_best:
        torch.save(state, checkpoint_dir / "best_checkpoint.pth")