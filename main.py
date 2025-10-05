from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example usage
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TicketDataset(
    texts=your_text_list, 
    labels=your_labels_list, 
    tokenizer=tokenizer, 
    max_len=128
)
data_loader = DataLoader(dataset, batch_size=16)
