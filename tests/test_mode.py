"""This test the model architecture"""

import pytest
import torch

from src.models.model import TextClassifier

@pytest.fixture # this helps to create a fixture that can be used in multiple tests
def model():
    return TextClassifier(
        input_size=100,
        hidden_size=10,
        num_classes=3
    )
    
def test_model_output_shape(model):
    batch_size = 32
    x = torch.randn((batch_size, 100), dtype=torch.float32)
    output = model(x)
    assert output.shape == (batch_size, 3), f"Expected output shape {(batch_size, 3)}, but got {output.shape}"