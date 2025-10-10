# scripts/monitor_models.py
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer
from src.models.triage_model import BertTriageModel
from src.models.solution_dssm import DSSM
from src.models.predictive_lstm import PredictiveLSTM
from src.utils.metrics import compute_triage_metrics
import logging
import json

# --- Configuration ---
ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DATA_DIR = Path("data/processed")
LOG_FILE = "monitoring.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# Assuming we have a recent batch of data for evaluation
TEST_DATA_PATH = PROCESSED_DATA_DIR / "triage_data" / "test.csv"

def load_latest_models():
    """Loads the most recently trained models and tokenizers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    # Triage model
    try:
        triage_path = ARTIFACTS_DIR / "triage_model" / "final"
        tokenizer = AutoTokenizer.from_pretrained(triage_path)
        # Assuming the label classes are available (e.g., from the LabelEncoder)
        num_labels = 5 # Placeholder, replace with actual number of labels
        model = BertTriageModel('bert-base-uncased', num_labels).to(device)
        model.load_state_dict(torch.load(triage_path / "pytorch_model.bin", map_location=device))
        models['triage'] = {'model': model.eval(), 'tokenizer': tokenizer, 'device': device}
    except Exception as e:
        logging.error(f"Failed to load triage model: {e}")

    # Solution DSSM (skipping for this example, as it's retrieval-based and harder to monitor automatically)

    # Predictive LSTM
    try:
        lstm_path = ARTIFACTS_DIR / "predictive_lstm"
        tokenizer = AutoTokenizer.from_pretrained(lstm_path / "tokenizer")
        num_labels = 1 # Binary classification
        # Placeholder for model parameters, match your training configuration
        model = PredictiveLSTM(len(tokenizer), 256, 512, num_labels, 2, 0.5).to(device)
        model.load_state_dict(torch.load(lstm_path / "predictive_lstm.pt", map_location=device))
        models['predictive'] = {'model': model.eval(), 'tokenizer': tokenizer, 'device': device}
    except Exception as e:
        logging.error(f"Failed to load predictive model: {e}")
        
    return models

def monitor_triage_model(models):
    """Monitors the performance of the triage model."""
    if 'triage' not in models: return
    
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        # Need to handle label encoding consistently
        # le = ... from training step
        # test_df['label'] = le.transform(test_df[TARGET_COLUMN])
        
        # ... Run evaluation and log results ...
        # For example:
        # predictions = model.predict(test_df['cleaned_text'])
        # acc = accuracy_score(test_df['label'], predictions)
        # logging.info(f"Triage Model Accuracy: {acc}")
    except Exception as e:
        logging.error(f"Error during triage model monitoring: {e}")

def monitor_predictive_model(models):
    """Monitors the performance of the predictive model."""
    if 'predictive' not in models: return
    
    try:
        # Similar process as above, but for sequential data
        # ... load predictive test data ...
        # ... run evaluation and log results ...
        # logging.info(f"Predictive Model F1-score: {f1}")
    except Exception as e:
        logging.error(f"Error during predictive model monitoring: {e}")


def main():
    models = load_latest_models()
    monitor_triage_model(models)
    monitor_predictive_model(models)
    
    logging.info("Model monitoring complete.")

if __name__ == "__main__":
    main()
