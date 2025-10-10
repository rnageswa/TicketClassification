import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertTriageModel(nn.Module):
    """
    A custom PyTorch model for ticket triage built on top of a pre-trained BERT.
    """
    def __init__(self, model_name: str, num_labels: int):
        """
        Args:
            model_name (str): The name of the pre-trained BERT model (e.g., 'bert-base-uncased').
            num_labels (int): The number of output classes for classification (e.g., number of teams).
        """
        super(BertTriageModel, self).__init__()
        self.num_labels = num_labels

        # Load the configuration and pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Add a classification head on top of the BERT model
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Defines the forward pass of the model.
        """
        # Get the output from the pre-trained BERT model
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # The [CLS] token's final hidden state is used for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and pass through the classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return (loss, logits) if labels is not None else logits

if __name__ == '__main__':
    # This block demonstrates how to instantiate the model
    # and is a simple self-check for the module.
    
    # Dummy setup
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 5 # Example: 5 different support teams
    
    print(f"Instantiating BertTriageModel with {MODEL_NAME} and {NUM_LABELS} labels...")
    model = BertTriageModel(MODEL_NAME, NUM_LABELS)
    print("Model instantiated successfully.")
    
    # Demonstrate the model structure
    print("\nModel architecture:")
    print(model)
