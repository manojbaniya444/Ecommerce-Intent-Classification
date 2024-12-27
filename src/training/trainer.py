import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path

class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        """
        Trainer class which will be responsible for training the model
        
        Args:
        model: nn.Module: model to train
        config: Config file yaml format for the configuration
        """
        self.model = model
        self.best_val_acc = 0
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config["training"]["learning_rate"]
        )
        
        self.setup_logging()
        
    def setup_logging(self):
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
    def train_epoch(self, train_loader: DataLoader):
        """
        Train one epoch of the model given the train data
        
        Args:
         train_loader: DataLoader: DataLoader object for training data
         
        Return:
         train_loss: float: average loss on the training data
         accuracy: float: accuracy on the training data
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            logits = self.model(data)
            loss = self.criterion(logits, target)
            
            self.optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        accuracy = 100 * correct / total
            
        # average loss for epoch
        return total_loss / len(train_loader), accuracy
    
    def validate(self, val_loader: DataLoader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader: DataLoader object for validation data
        Return:
            val_loss: float: average loss on the validation dataset
            val_acc: float: average accuracy on the validation dataset
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, target)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        return total_loss / len(val_loader), 100 * correct / total

    def train(self, train_loader, val_loader):
        """
        Train the model
        
        Args:
            train_loader: DataLoader: training dataloader dataset for training the model
            val_loader: DataLoader: validation dataloader dataset for validation
        """
        save_dir = Path(self.config["training"]["save_dir"])
        save_dir.mkdir(exist_ok=True, parents=True)
        
        for epoch in range(self.config["training"]["num_epochs"]):
            logging.info(f"Epoch {epoch}/{self.config["training"]["num_epochs"]}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            
            val_loss, val_acc = self.validate(val_loader)
            
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            
            # save the log metrics
            logging.info(f"Metrics: {metrics}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_dir / "best_model.pth")
                logging.info(f"Saved best model with validation accuracy: {val_acc: .4f}")