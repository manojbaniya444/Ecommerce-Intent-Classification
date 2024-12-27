from src.data.dataset import TrainDataset, ValidationDataset
from src.models.model import TextClassifier
from src.training.trainer import Trainer
from src.utils.utils import load_config
import pytest
from torch.utils.data import DataLoader

def test_full_training_cycle():
    config = load_config("config/config.yaml")
    config["training"]["num_epochs"] = 1
    train_dataset = TrainDataset(data_path=config["data"]["train_data"])
    assert len(train_dataset) > 0, "Dataset is empty"
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
    trainer.train(train_loader=train_loader, val_loader=val_loader)
    assert hasattr(trainer, "best_val_acc"), "Model is not training check the model training in trainer file"