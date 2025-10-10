import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from transformers import AutoTokenizer
from src.models.triage_model import BertTriageModel
from src.models.solution_dssm import DSSM
from src.models.predictive_lstm import PredictiveLSTM

# --- Configuration ---
MODEL_DIR = Path("artifacts")
# Triage model configuration
TRIAGE_MODEL_PATH = MODEL_DIR / "triage_model/final"
# Solution DSSM model configuration
DSSM_MODEL_PATH = MODEL_DIR / "solution_dssm/dssm_model.pt"
DSSM_TOKENIZER_PATH = MODEL_DIR / "solution_dssm/tokenizer"
# Predictive LSTM model configuration
LSTM_MODEL_PATH = MODEL_DIR / "predictive_lstm/predictive_lstm.pt"
LSTM_TOKENIZER_PATH = MODEL_DIR / "predictive_lstm/tokenizer"
LSTM_SEQUENCE_LENGTH = 5  # Must match the value used in data_splitter.py
# Example placeholder for a list of all known solution documents
# In a real-world scenario, this would be a large, indexed database
solution_documents = pd.Series([
    "Clear your browser cache and cookies to fix login issues.",
    "Check your account permissions. The issue may be due to restricted access.",
    "Use the 'Forgot Password' link to reset your password.",
    # ... more solution documents
])

# Initialize FastAPI app
app = FastAPI(
    title="Ticket Intelligence API",
    description="API for triage, solution recommendation, and predictive support.",
    version="1.0.0"
)

# Global variables to hold models and tokenizers
# Loaded once on startup
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_models():
    """Load all trained models and tokenizers on application startup."""
    print("Loading models and tokenizers...")
    
    try:
        # Load Triage Model
        models['triage_tokenizer'] = AutoTokenizer.from_pretrained(TRIAGE_MODEL_PATH)
        triage_config = models['triage_tokenizer'].get_added_tokens_decoder()
        triage_num_labels = len(triage_config) # Example way to get num_labels
        models['triage_model'] = BertTriageModel('bert-base-uncased', triage_num_labels).to(device)
        models['triage_model'].load_state_dict(torch.load(TRIAGE_MODEL_PATH / "pytorch_model.bin", map_location=device))
        models['triage_model'].eval()
        
        # Load DSSM Model
        models['dssm_tokenizer'] = AutoTokenizer.from_pretrained(DSSM_TOKENIZER_PATH)
        models['dssm_model'] = DSSM().to(device)
        models['dssm_model'].load_state_dict(torch.load(DSSM_MODEL_PATH, map_location=device))
        models['dssm_model'].eval()
        
        # Load Predictive LSTM Model
        models['lstm_tokenizer'] = AutoTokenizer.from_pretrained(LSTM_TOKENIZER_PATH)
        lstm_vocab_size = len(models['lstm_tokenizer'])
        # Instantiate with same parameters used for training
        models['lstm_model'] = PredictiveLSTM(
            vocab_size=lstm_vocab_size,
            embedding_dim=256, # Example value, match your training
            hidden_dim=512,    # Example value, match your training
            output_dim=1,      # Example value, match your training
            n_layers=2,        # Example value, match your training
            dropout=0.5        # Example value, match your training
        ).to(device)
        models['lstm_model'].load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
        models['lstm_model'].eval()
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while loading models: {e}")
        
    print("All models and tokenizers loaded successfully.")

# --- API Input Models ---
class TextIn(BaseModel):
    text: str

class TriageResult(BaseModel):
    predicted_team: str

class SolutionIn(BaseModel):
    query: str

class SolutionResult(BaseModel):
    recommended_solutions: list

class PredictiveIn(BaseModel):
    ticket_history: list # List of strings (last N ticket descriptions)

class PredictiveResult(BaseModel):
    predicted_escalation_risk: float

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Ticket Intelligence API is running."}

@app.post("/predict_triage", response_model=TriageResult)
async def predict_triage(data: TextIn):
    """Predicts the best support team for a new ticket."""
    tokenizer = models['triage_tokenizer']
    model = models['triage_model']
    
    encoded_input = tokenizer(data.text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded_input.to(device))
    
    # Placeholder: convert prediction back to original label
    # This logic depends on the LabelEncoder used in training
    predicted_label_index = torch.argmax(output[0] if isinstance(output, tuple) else output).item()
    le_classes = ['TeamA', 'TeamB', 'TeamC', 'TeamD', 'TeamE'] # Example labels
    predicted_team = le_classes[predicted_label_index]
    
    return {"predicted_team": predicted_team}

@app.post("/recommend_solution", response_model=SolutionResult)
async def recommend_solution(data: SolutionIn):
    """Recommends solutions based on a ticket query using DSSM."""
    tokenizer = models['dssm_tokenizer']
    model = models['dssm_model']
    
    # Encode query
    query_encoded = tokenizer(data.query, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        query_emb = model.get_query_embedding(query_encoded['input_ids'].to(device), query_encoded['attention_mask'].to(device))
    
    # Pre-calculate or load embeddings for all solutions
    # For a real application, you would use a vector database
    # For now, we simulate this by embedding all documents
    solution_embeddings = []
    for doc in solution_documents:
        doc_encoded = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            doc_emb = model.get_document_embedding(doc_encoded['input_ids'].to(device), doc_encoded['attention_mask'].to(device))
        solution_embeddings.append(doc_emb)

    # Find the best matches
    similarities = torch.cosine_similarity(query_emb, torch.cat(solution_embeddings, dim=0))
    top_indices = similarities.argsort(descending=True)[:3] # Top 3 recommendations
    
    recommendations = solution_documents.iloc[top_indices.cpu()].tolist()
    
    return {"recommended_solutions": recommendations}


@app.post("/predict_escalation", response_model=PredictiveResult)
async def predict_escalation(data: PredictiveIn):
    """Predicts the likelihood of ticket escalation based on ticket history."""
    tokenizer = models['lstm_tokenizer']
    model = models['lstm_model']

    if len(data.ticket_history) < LSTM_SEQUENCE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Ticket history must have at least {LSTM_SEQUENCE_LENGTH} entries.")
        
    # Concatenate the sequence of texts
    sequence_text = " ".join(data.ticket_history[-LSTM_SEQUENCE_LENGTH:])
    
    # Tokenize the sequence
    encoded = tokenizer(sequence_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    
    with torch.no_grad():
        output = model(encoded['input_ids'].to(device))
        
    # Apply sigmoid to get probability and detach from GPU
    risk = torch.sigmoid(output).item()
    
    return {"predicted_escalation_risk": risk}

