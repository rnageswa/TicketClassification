import torch
import torch.nn as nn

class PredictiveLSTM(nn.Module):
    """
    An LSTM-based model for predicting ticket outcomes based on historical sequences.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        """
        Args:
            vocab_size (int): The number of unique tokens in the dataset.
            embedding_dim (int): The size of the word embeddings.
            hidden_dim (int): The size of the LSTM hidden state.
            output_dim (int): The size of the output, based on the prediction task (e.g., 1 for binary classification).
            n_layers (int): The number of LSTM layers.
            dropout (float): The dropout probability for regularization.
        """
        super(PredictiveLSTM, self).__init__()

        # An embedding layer to convert token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM layer(s) to capture sequential dependencies
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True  # Batch dimension is the first dimension
        )

        # A dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # The final output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        """
        Defines the forward pass of the model.

        Args:
            text (Tensor): The input tensor containing sequences of token IDs.
        """
        # Embed the input text
        embedded = self.embedding(text)
        
        # Pass the embedded sequence through the LSTM
        # The output contains the hidden state for each time step
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # We use the final hidden state for classification
        # Shape of hidden: [n_layers, batch_size, hidden_dim]
        # We select the final layer's hidden state
        hidden_final = hidden[-1, :, :]
        
        # Apply dropout to the final hidden state
        hidden_final = self.dropout(hidden_final)
        
        # Pass through the final fully connected layer
        output = self.fc(hidden_final)
        
        return output

if __name__ == '__main__':
    # Simple self-check for the model architecture
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1 # Binary classification (e.g., escalation vs. no escalation)
    N_LAYERS = 2
    DROPOUT = 0.5
    
    print("Instantiating PredictiveLSTM model...")
    model = PredictiveLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    print("Model instantiated successfully.")
    print("\nModel architecture:")
    print(model)

