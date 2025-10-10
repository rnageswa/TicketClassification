# scripts/retrain_models.py
import pandas as pd
from pathlib import Path
import subprocess
import logging

# --- Configuration ---
LOG_FILE = "retraining.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_latest_data():
    """Simulates fetching new, resolved data, potentially from Salesforce."""
    logging.info("Fetching latest resolved data...")
    # In a real-world scenario, this would connect to Salesforce
    # and fetch new Case data, possibly including agent feedback.
    
    # For now, we'll just assume a new file is available.
    new_data = {
        'ticket_id': [1000001, 1000002],
        'ticket_description': ["Another login issue resolved by a simple cache clear.", "User has forgotten password again."],
        'assigned_team': ['L1', 'L1'],
        'created_at': ['2025-10-01 09:00:00', '2025-10-02 14:30:00'],
        'resolution_text': ["Instructed user to clear cache and cookies.", "Sent password reset link."]
    }
    new_df = pd.DataFrame(new_data)
    
    # Append to the raw data file (simulation)
    raw_path = Path("data/raw/support_tickets.csv")
    new_df.to_csv(raw_path, mode='a', header=False, index=False)
    
    logging.info(f"Appended {len(new_df)} new tickets to raw data.")
    return raw_path

def run_retraining_pipeline():
    """
    Executes the full pipeline for preprocessing, splitting, and retraining.
    """
    logging.info("Starting model retraining pipeline.")
    
    try:
        # Run preprocessing script
        logging.info("Running preprocess.py...")
        subprocess.run(["python", "src/data_processing/preprocess.py"], check=True)
        
        # Run data splitting script
        logging.info("Running data_splitter.py...")
        subprocess.run(["python", "src/data_processing/data_splitter.py"], check=True)
        
        # Run triage model training
        logging.info("Running train_triage.py...")
        subprocess.run(["python", "src/training/train_triage.py"], check=True)
        
        # Run solution model training
        logging.info("Running train_solution.py...")
        subprocess.run(["python", "src/training/train_solution.py"], check=True)
        
        # Run predictive model training
        logging.info("Running train_predictive.py...")
        subprocess.run(["python", "src/training/train_predictive.py"], check=True)

        logging.info("Retraining pipeline completed successfully.")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Retraining failed at a subprocess step: {e}")
        raise
    except FileNotFoundError as e:
        logging.error(f"A required script was not found: {e}")
        raise

def main():
    try:
        fetch_latest_data()
        run_retraining_pipeline()
    except Exception as e:
        logging.critical(f"An unhandled error occurred during the retraining process: {e}")

if __name__ == '__main__':
    main()
