import pandas as pd
import torch
import torchmetrics
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from src.models.triage_model import BertTriageModel
from src.utils.metrics import compute_triage_metrics

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128  # Adjust based on your ticket length
TRIAGE_DATA_PATH = Path("../data/processed/triage_data")
ARTIFACTS_PATH = Path("../artifacts/triage_model")
TARGET_COLUMN = 'assigned_team' # Must match column used in data_splitter.py

def main():
    """
    Main training function for the ticket triage model.
    """
    # --- 1. Load Data ---
    train_df = pd.read_csv(TRIAGE_DATA_PATH / "train.csv")
    test_df = pd.read_csv(TRIAGE_DATA_PATH / "test.csv")
    
    print(f"Loaded {len(train_df)} training examples and {len(test_df)} testing examples.")

    # --- 2. Label Encoding ---
    # Convert string labels (e.g., 'L1', 'L2') into integers
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df[TARGET_COLUMN])
    test_df['label'] = le.transform(test_df[TARGET_COLUMN])
    num_labels = len(le.classes_)
    print(f"Found {num_labels} unique labels: {le.classes_}")

    # --- 3. Tokenize Data ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'label']])

    def tokenize_function(examples):
        return tokenizer(
            examples['cleaned_text'], 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
        
    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text', 'label'])
    tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True, remove_columns=['cleaned_text', 'label'])
    
    # Add back the integer labels
    tokenized_train_datasets = tokenized_train_datasets.add_column('labels', train_df['label'].tolist())
    tokenized_test_datasets = tokenized_test_datasets.add_column('labels', test_df['label'].tolist())

    # --- 4. Initialize Model ---
    model = BertTriageModel(model_name=MODEL_NAME, num_labels=num_labels)
    
    # --- 5. Setup TrainingArguments and Trainer ---
    training_args = TrainingArguments(
        output_dir=ARTIFACTS_PATH / "checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save checkpoint at the end of each epoch
        logging_dir='./logs',
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1-score to select the best model
        push_to_hub=False,
    )
    
    # Custom function for metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate standard classification metrics
        f1 = torchmetrics.functional.f1_score(
            torch.from_numpy(predictions),
            torch.from_numpy(labels),
            task='multiclass',
            num_classes=num_labels,
            average='weighted'
        )
        accuracy = torchmetrics.functional.accuracy(
            torch.from_numpy(predictions),
            torch.from_numpy(labels),
            task='multiclass',
            num_classes=num_labels
        )
        
        return {"f1": f1.item(), "accuracy": accuracy.item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_test_datasets,
        compute_metrics=compute_metrics,
    )

    # --- 6. Train and Evaluate ---
    print("\nStarting model fine-tuning...")
    trainer.train()
    
    # --- 7. Save Final Model ---
    final_model_path = ARTIFACTS_PATH / "final"
    final_model_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\nTraining complete. Best model saved to {final_model_path}.")
    
if __name__ == '__main__':
    # Add src to the system path to allow imports
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Ensure necessary NLTK data is downloaded for preprocess.py dependency
    try:
        nltk.data.find('corpora/stopwords')
    except (nltk.downloader.DownloadError, NameError):
        import nltk
        nltk.download('stopwords')
    
    main()

