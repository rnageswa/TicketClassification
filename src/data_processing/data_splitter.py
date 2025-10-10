import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define file paths based on the project structure
PROCESSED_DATA_PATH = Path("../data/processed/tickets_clean.csv")
TRIAGE_DATA_PATH = Path("../data/processed/triage_data")
SOLUTION_DATA_PATH = Path("../data/processed/solution_data")

def split_for_triage(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets for the triage classification task.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_column (str): The column to use as the target variable (e.g., 'assigned_team').
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.
    """
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found for triage splitting.")
        return

    print("Splitting data for ticket triage...")
    train_df, test_df = train_test_split(
        df[['cleaned_text', target_column]],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column] if 'stratify' else None # Optional: stratify on the target column if needed
    )
    
    TRIAGE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRIAGE_DATA_PATH / "train.csv", index=False)
    test_df.to_csv(TRIAGE_DATA_PATH / "test.csv", index=False)
    
    print(f"Triage data saved. Train: {len(train_df)} rows, Test: {len(test_df)} rows.")

def create_dssm_pairs(df: pd.DataFrame, query_column: str, document_column: str):
    """
    Prepares the data for the DSSM solution recommendation model.
    Assumes a format where each ticket has a corresponding resolution.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        query_column (str): The column containing the ticket description.
        document_column (str): The column containing the resolved solution.
    """
    if query_column not in df.columns or document_column not in df.columns:
        print(f"Error: Required columns '{query_column}' or '{document_column}' not found for DSSM pairing.")
        return
        
    print("Creating query-document pairs for DSSM...")
    dssm_df = df[[query_column, document_column]].dropna()
    
    SOLUTION_DATA_PATH.mkdir(parents=True, exist_ok=True)
    dssm_df[query_column].to_csv(SOLUTION_DATA_PATH / "dssm_train_queries.txt", header=False, index=False)
    dssm_df[document_column].to_csv(SOLUTION_DATA_PATH / "dssm_train_documents.txt", header=False, index=False)
    
    print(f"DSSM data saved. Generated {len(dssm_df)} query-document pairs.")

def sequence_for_predictive(df: pd.DataFrame, sequence_column: str, time_column: str,
                            group_by_column: str = 'customer_id', sequence_length: int = 5):
    """
    Sequences historical ticket data for the LSTM predictive model.
    This creates a rolling window of tickets to predict the outcome of the next ticket.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        sequence_column (str): The column containing the data to sequence (e.g., 'cleaned_text').
        time_column (str): The column to sort the sequences by (e.g., 'created_at').
        group_by_column (str): The column to group tickets by (e.g., 'customer_id' or 'system_id').
        sequence_length (int): The number of past tickets to consider in each sequence.
    """
    if sequence_column not in df.columns or time_column not in df.columns or group_by_column not in df.columns:
        print(f"Error: Required columns not found for predictive sequencing.")
        return
        
    print("Sequencing data for predictive LSTM model...")
    df = df.sort_values(by=[group_by_column, time_column]).reset_index(drop=True)
    
    sequences = []
    labels = []
    
    for _, group in df.groupby(group_by_column):
        if len(group) > sequence_length:
            for i in range(len(group) - sequence_length):
                sequences.append(group[sequence_column].iloc[i:i + sequence_length].tolist())
                labels.append(group[sequence_column].iloc[i + sequence_length])
                
    # Create a DataFrame for the sequences
    sequence_df = pd.DataFrame(sequences)
    sequence_df['label'] = labels
    
    # Split the sequence data into train and test sets
    train_seq, test_seq = train_test_split(sequence_df, test_size=0.2, random_state=42)
    
    # Save the files
    Path("../data/processed/predictive_data").mkdir(parents=True, exist_ok=True)
    train_seq.to_csv(Path("../data/processed/predictive_data/train_sequences.csv"), index=False)
    test_seq.to_csv(Path("../data/processed/predictive_data/test_sequences.csv"), index=False)
    
    print(f"Predictive data saved. Train: {len(train_seq)} rows, Test: {len(test_seq)} rows.")
    

if __name__ == '__main__':
    # Make sure the preprocessed file exists
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: Processed data file {PROCESSED_DATA_PATH} not found.")
        print("Please run `src/data_processing/preprocess.py` first.")
    else:
        df_clean = pd.read_csv(PROCESSED_DATA_PATH)
        
        # === Configuration for each task ===
        # Note: These columns are examples. Adjust them based on your dataset.
        
        # Triage Model
        TRIAGE_TARGET_COLUMN = 'assigned_team'  # Example target column
        split_for_triage(df_clean, TRIAGE_TARGET_COLUMN)
        
        # Solution Management Model (DSSM)
        DSSM_QUERY_COLUMN = 'cleaned_text'
        DSSM_DOCUMENT_COLUMN = 'resolution_text' # Example column for the resolved solution
        # NOTE: You must have a 'resolution_text' or similar column for this step
        # For demonstration purposes, we will skip this if the column doesn't exist
        if DSSM_DOCUMENT_COLUMN in df_clean.columns:
            create_dssm_pairs(df_clean, DSSM_QUERY_COLUMN, DSSM_DOCUMENT_COLUMN)
        else:
            print(f"Skipping DSSM data preparation: Missing '{DSSM_DOCUMENT_COLUMN}' column.")

        # Predictive Support Model (LSTM)
        PREDICTIVE_SEQUENCE_COLUMN = 'cleaned_text'
        PREDICTIVE_TIME_COLUMN = 'created_at'
        PREDICTIVE_GROUP_BY_COLUMN = 'customer_id' # Example column to group by
        # NOTE: You must have 'customer_id' or a similar grouping column
        # For demonstration, we will skip this if columns don't exist
        if PREDICTIVE_SEQUENCE_COLUMN in df_clean.columns and PREDICTIVE_TIME_COLUMN in df_clean.columns:
            sequence_for_predictive(
                df_clean, 
                PREDICTIVE_SEQUENCE_COLUMN, 
                PREDICTIVE_TIME_COLUMN, 
                PREDICTIVE_GROUP_BY_COLUMN
            )
        else:
            print("Skipping predictive data preparation: Missing required columns.")

