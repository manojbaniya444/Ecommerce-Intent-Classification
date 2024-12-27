import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import TrainDataset, ValidationDataset
from src.models.model import TextClassifier
from src.training.trainer import Trainer
from src.utils.utils import load_config
import pandas as pd
import torch

def main():
    parser = argparse.ArgumentParser(description="Train the model on the given data")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    train_dataset = TrainDataset(config["data"]["train_data"])
    
    val_dataset = ValidationDataset(data_path=config["data"]["val_data"], vectorizer=train_dataset.vectorizer, label_encoder=train_dataset.label_encoder)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config["data"]["batch_size"],
                              shuffle=True,
                              num_workers=config["data"]["num_workers"])
    
    val_loader = DataLoader(val_dataset,
                             batch_size=config["data"]["batch_size"],
                             shuffle=False,
                             num_workers=config["data"]["num_workers"])
    
    model = TextClassifier(
        input_size=train_dataset.get_vocab_size(),
        hidden_size=config["model"]["hidden_size"],
        num_classes=train_dataset.get_num_classes()
    )
    
    trainer = Trainer(model=model, config=config)
    
    trainer.train(train_loader, val_loader)
    
if __name__ == "__main__":
    main()
