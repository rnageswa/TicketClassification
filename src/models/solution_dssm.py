import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DSSM(nn.Module):
    """
    A Deep Structured Semantic Model (DSSM) using a dual-encoder architecture.
    """
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initializes the DSSM with two identical transformer encoders.

        Args:
            model_name (str): The name of the pre-trained model to use for the encoders.
        """
        super(DSSM, self).__init__()
        
        # We use separate encoders for query and document. This allows for
        # asymmetric training, but you can also use a single shared encoder.
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.document_encoder = AutoModel.from_pretrained(model_name)
        
        # A simple non-linear layer after the transformer output
        # You can add more layers here for a deeper model
        self.query_proj = nn.Linear(self.query_encoder.config.hidden_size, 128)
        self.document_proj = nn.Linear(self.document_encoder.config.hidden_size, 128)

    def forward(self, query_input_ids, query_attention_mask,
                doc_input_ids, doc_attention_mask):
        """
        Calculates the embeddings for queries and documents.
        """
        # Get the [CLS] token's final hidden state for the query
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embedding = query_outputs.last_hidden_state[:, 0, :]
        query_embedding = self.query_proj(query_embedding)
        
        # Get the [CLS] token's final hidden state for the document
        doc_outputs = self.document_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask
        )
        doc_embedding = doc_outputs.last_hidden_state[:, 0, :]
        doc_embedding = self.document_proj(doc_embedding)
        
        return query_embedding, doc_embedding

    def get_query_embedding(self, input_ids, attention_mask):
        """
        Returns the embedding for a given query.
        """
        with torch.no_grad():
            outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = self.query_proj(embedding)
            return embedding
            
    def get_document_embedding(self, input_ids, attention_mask):
        """
        Returns the embedding for a given document.
        """
        with torch.no_grad():
            outputs = self.document_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = self.document_proj(embedding)
            return embedding

if __name__ == '__main__':
    # Simple self-check for the model architecture
    model = DSSM()
    print("DSSM model instantiated successfully.")
    print(model)

