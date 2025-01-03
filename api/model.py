import torch
from src.models.model import TextClassifier
from src.data.dataset import TrainDataset
import logging
from src.utils.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_dataset = None
        self.config = None
        self.load_features()
        self.load_model()
        
    def load_features(self):
        self.config = load_config(config_path=r"config/config.yaml")
        self.train_dataset = TrainDataset(self.config["data"]["train_data"])
        
    def load_model(self):
        try:
            self.model = TextClassifier(
            input_size=self.train_dataset.get_vocab_size(),
            hidden_size=self.config["model"]["hidden_size"],
            num_classes=self.train_dataset.get_num_classes()
            )
        
            self.model.load_state_dict(torch.load("./checkpoints/best-model.pth", weights_only=True), )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model Loaded Success")
        except Exception as e:
            logger.error(f"Error Loading Model: {str(e)}")
            raise RuntimeError("Model Loading Failed")
        
    def preprocess(self, text: str):
        data = text.strip().lower
        data = self.train_dataset.vectorizer.transform([text]).toarray()
        data = torch.tensor(data, dtype=torch.float32)
        return data
        
    def predict_single(self, text: str):
        data = self.preprocess(text)
        data = data.to(self.device)
        logits = self.model(data)
        # print(logits)
        preds = torch.softmax(logits, dim=-1)
        label = torch.argmax(preds, dim=-1)
        logger.info(f"Prediction: {label.item()}")
        intent = self.train_dataset.label2id(label.item())
        return intent
    
if __name__ == "__main__":
    model = Model()
    label = model.predict_single("malai redmi mobile ko barema jankari dinu.")
    print(label)