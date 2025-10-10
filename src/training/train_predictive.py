import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from src.models.predictive_lstm import PredictiveLSTM

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 128 # Must match max_length during tokenization
PREDICTIVE_DATA_PATH = Path("../data/processed/predictive_data")
ARTIFACTS_PATH = Path("../artifacts/predictive_lstm")
SEQUENCE_LENGTH = 5  # Must match the value used in data_splitter.py

# --- Dataset and DataLoader Setup ---
class PredictiveDataset(Dataset):
    """
    Dataset for the predictive LSTM model.
    Handles tokenization of text within sequences.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Concatenate sequence texts into a single string for tokenization
        sequence_text = " ".join([row[i] for i in range(SEQUENCE_LENGTH)])
        
        # Tokenize the concatenated text
        encoded = self.tokenizer(
            sequence_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # The target label is the 'label' column
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': label
        }

def main():
    """
    Main training function for the predictive LSTM model.
    """
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. Load Data ---
    try:
        train_df = pd.read_csv(PREDICTIVE_DATA_PATH / "train_sequences.csv")
        test_df = pd.read_csv(PREDICTIVE_DATA_PATH / "test_sequences.csv")
    except FileNotFoundError:
        print("Error: Predictive data files not found.")
        print("Please run `src/data_processing/data_splitter.py` first.")
        return

    print(f"Loaded {len(train_df)} training sequences and {len(test_df)} testing sequences.")

    # --- 2. Initialize Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- 3. Setup Dataset and DataLoader ---
    train_dataset = PredictiveDataset(train_df, tokenizer, MAX_SEQ_LENGTH)
    test_dataset = PredictiveDataset(test_df, tokenizer, MAX_SEQ_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. Initialize Model ---
    # The tokenizer's vocab size is needed for the embedding layer
    VOCAB_SIZE = len(tokenizer)
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    OUTPUT_DIM = 1 # For binary classification
    N_LAYERS = 2
    DROPOUT = 0.5
    
    model = PredictiveLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)

    # --- 5. Setup Optimizer and Loss Function ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Binary Cross-Entropy with Logits for binary classification
    loss_function = nn.BCEWithLogitsLoss()

    # --- 6. Training and Evaluation Loop ---
    print("\nStarting predictive LSTM training...")
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze(1)
            
            # Compute loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Training Loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(inputs).squeeze(1)
                
                preds = torch.sigmoid(outputs).round() # Convert logits to probabilities and then to binary predictions
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1} Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
    # --- 7. Save Final Model ---
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    model_save_path = ARTIFACTS_PATH / "predictive_lstm.pt"
    tokenizer_save_path = ARTIFACTS_PATH / "tokenizer"
    
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    
    print(f"\nTraining complete. Model saved to {model_save_path}.")

if __name__ == '__main__':
    # Add src to the system path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    main()
