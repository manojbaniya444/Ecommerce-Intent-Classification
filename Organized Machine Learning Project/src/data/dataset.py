import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class TrainDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        
        self.vectorizer = CountVectorizer()
        self.bow_features = self.vectorizer.fit_transform(self.df['text']).toarray()
        
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['intent'])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        bow_features = torch.tensor(self.bow_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bow_features, label
    
    def get_labels(self):
        return self.label_encoder.classes_
    
    def get_vocab_size(self):
        return len(self.vectorizer.vocabulary_)
    
    def get_num_classes(self):
        return len(self.label_encoder.classes_)
    
    def label2id(self, label):
        return self.label_encoder.classes_[label]
    
class ValidationDataset(Dataset):
    """Prepare validation dataset because we need to have the same transform as train data"""
    def __init__(self, data_path, vectorizer, label_encoder):
        self.df = pd.read_csv(data_path)
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        
        self.bow_features = self.vectorizer.transform(self.df['text']).toarray()
        self.labels = self.label_encoder.transform(self.df['intent'])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        bow_features = torch.tensor(self.bow_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bow_features, label