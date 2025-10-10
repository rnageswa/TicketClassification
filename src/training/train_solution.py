import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
import random
from src.models.solution_dssm import DSSM

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 128
SOLUTION_DATA_PATH = Path("../data/processed/solution_data")
ARTIFACTS_PATH = Path("../artifacts/solution_dssm")

class SolutionDataset(Dataset):
    """
    Dataset for the DSSM model.
    Loads query-document pairs and generates in-batch negatives.
    """
    def __init__(self, queries, documents, tokenizer, max_length):
        self.queries = queries
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        query = self.queries[idx]
        positive_doc = self.documents[idx]
        
        # Tokenize query and positive document
        query_encoded = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        pos_doc_encoded = self.tokenizer(
            positive_doc,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_doc_encoded['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_doc_encoded['attention_mask'].squeeze(0)
        }

def main():
    """
    Main training function for the solution DSSM model.
    """
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. Load Data ---
    # In a real scenario, you would have separate query/doc files
    # Here we load the pairs from the splitter script
    try:
        queries = pd.read_csv(SOLUTION_DATA_PATH / "dssm_train_queries.txt", header=None).squeeze("columns").tolist()
        documents = pd.read_csv(SOLUTION_DATA_PATH / "dssm_train_documents.txt", header=None).squeeze("columns").tolist()
    except FileNotFoundError:
        print("Required DSSM data not found. Creating dummy data for demonstration.")
        queries = ["The login page is broken", "Cannot access my dashboard", "My password doesn't work"]
        documents = ["Clear your browser cache and cookies to fix login issues.", "Check your account permissions. The issue may be due to restricted access.", "Use the 'Forgot Password' link to reset your password."]
    
    print(f"Loaded {len(queries)} query-document pairs for training.")

    # --- 2. Initialize Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DSSM(model_name=MODEL_NAME).to(device)

    # --- 3. Setup Dataset and DataLoader ---
    dataset = SolutionDataset(queries, documents, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. Setup Optimizer and Loss Function ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Contrastive loss (MarginRankingLoss)
    loss_function = nn.MarginRankingLoss(margin=1.0)

    # --- 5. Training Loop ---
    print("\nStarting DSSM training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move batch to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
            pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)
            
            # Generate negative samples (in-batch negatives)
            # Shuffle the documents within the batch to create negative pairs
            neg_doc_input_ids = pos_doc_input_ids[torch.randperm(pos_doc_input_ids.size(0))]
            neg_doc_attention_mask = pos_doc_attention_mask[torch.randperm(pos_doc_attention_mask.size(0))]
            
            # Get embeddings
            query_emb, pos_doc_emb = model(
                query_input_ids, query_attention_mask,
                pos_doc_input_ids, pos_doc_attention_mask
            )
            _, neg_doc_emb = model(
                query_input_ids, query_attention_mask,
                neg_doc_input_ids, neg_doc_attention_mask
            )
            
            # Calculate cosine similarity
            pos_score = torch.cosine_similarity(query_emb, pos_doc_emb)
            neg_score = torch.cosine_similarity(query_emb, neg_doc_emb)
            
            # Calculate loss: we want positive scores to be higher than negative scores
            target = torch.ones(pos_score.size()).to(device)
            loss = loss_function(pos_score, neg_score, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # --- 6. Save Model and Tokenizer ---
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    model_save_path = ARTIFACTS_PATH / "dssm_model.pt"
    tokenizer_save_path = ARTIFACTS_PATH / "tokenizer"
    
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    
    print(f"\nTraining complete. Model saved to {model_save_path}.")

if __name__ == '__main__':
    # Add src to the system path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Run the main training function
    main()

