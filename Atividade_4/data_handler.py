from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

class DataHandler(Dataset):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(self, texts):
        return DataHandler.tokenizer(texts, padding=True, truncation=True, max_length=512)

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = self.tokenize_function(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # Ensure labels are integers
        return item
