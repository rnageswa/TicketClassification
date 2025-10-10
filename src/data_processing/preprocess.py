import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer

# Ensure NLTK data is downloaded (only need to run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Use a pre-trained tokenizer from Hugging Face for deep learning models
# 'bert-base-uncased' is a good general-purpose choice for English text
HF_TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning:
    - Converts to lowercase.
    - Removes URLs.
    - Removes HTML tags.
    - Removes punctuation.
    - Removes stop words.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return " ".join(tokens)

def tokenize_text(text: str) -> list:
    """
    Tokenizes text using the Hugging Face tokenizer.
    """
    if not isinstance(text, str):
        return []
        
    tokens = HF_TOKENIZER.tokenize(text)
    return tokens

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from existing data.
    - Ticket length (word count).
    - Time of day submitted.
    """
    if 'ticket_description' in df.columns:
        df['ticket_length'] = df['ticket_description'].astype(str).apply(lambda x: len(x.split()))
    
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['submission_hour'] = df['created_at'].dt.hour
        df['submission_day_of_week'] = df['created_at'].dt.day_of_week
        
    return df

def preprocess_data(file_path: str, save_path: str) -> pd.DataFrame:
    """
    Main function to load, preprocess, and save the dataset.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return pd.DataFrame()

    print("Starting data preprocessing...")
    # Fill any potential missing values in the text column
    df['ticket_description'] = df['ticket_description'].fillna('')
    
    # Apply text cleaning
    df['cleaned_text'] = df['ticket_description'].apply(clean_text)
    
    # Apply tokenization
    # Note: Tokenizing 1M rows can take time. For training, models will do this internally.
    # We save the cleaned text for simplicity, tokenization will happen during model training.
    # df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    
    # Apply feature engineering
    df = feature_engineer(df)
    
    print("Preprocessing complete. Saving cleaned data...")
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}.")
    
    return df

if __name__ == '__main__':
    from pathlib import Path

    # Define file paths based on the project structure
    raw_file = Path("../data/raw/support_tickets.csv")
    processed_file = Path("../data/processed/tickets_clean.csv")
    
    # Make sure the raw file exists for the script to run
    # For demonstration, you might create a dummy file
    if not raw_file.exists():
        print(f"Creating a dummy CSV file at {raw_file} for demonstration.")
        dummy_data = {
            'ticket_id': [1, 2, 3],
            'ticket_description': [
                "The login page is not working. I'm unable to log in.",
                "I get an error 404 when accessing my profile. http://example.com/profile",
                "How do I reset my password? This is urgent! <p>Thanks</p>"
            ],
            'created_at': ['2025-10-10 09:00:00', '2025-10-10 14:30:00', '2025-10-10 22:15:00'],
            'assigned_team': ['L1', 'L2', 'L1'],
        }
        pd.DataFrame(dummy_data).to_csv(raw_file, index=False)
        
    # Run the preprocessing pipeline
    df_cleaned = preprocess_data(raw_file, processed_file)
    
    if not df_cleaned.empty:
        print("\nDisplaying a sample of the processed data:")
        print(df_cleaned[['ticket_id', 'cleaned_text', 'ticket_length', 'submission_hour']].head())
