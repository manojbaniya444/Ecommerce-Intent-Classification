import pytest
import torch

from src.data.dataset import TrainDataset, ValidationDataset

def test_dataset_loading():
    dataset = TrainDataset(data_path="data/train.csv")
    assert len(dataset) > 0, "Dataset is empty"
    assert hasattr(dataset, "bow_features"), "Dataset does not have bow_features attribute"
    assert hasattr(dataset, "labels"), "Dataset does not have labels attribute"
    
def test_validation_dataset_loading():
    train_dataset = TrainDataset(data_path="data/train.csv")
    val_dataset = ValidationDataset(data_path="data/val.csv", vectorizer=train_dataset.vectorizer, label_encoder=train_dataset.label_encoder)
    assert len(val_dataset) > 0, "Validation dataset is empty"
    