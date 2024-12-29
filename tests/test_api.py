from api.model import Model
import pytest

model = Model()

def test_response():
    classes = model.train_dataset.get_labels()
    predicted_label = model.predict_single("I Love this")
    # print(predicted_label)
    assert predicted_label in classes, "Failed to predict label"
    