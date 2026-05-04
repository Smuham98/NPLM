import torch
import torch.nn as nn

class NPLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size, dropout):
        """
        Neural Probabilistic Language Model (NPLM) implementation.
        """
        super(NPLM, self).__init__()
        
        # Parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, labels=None):
        """
        Forward pass for the NPLM model.
        """
        # Get embeddings for input tokens
        embeddings = self.embedding(input_ids)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        
        # Dropout layer
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # We use the output of the last token in the sequence (not all tokens)
            loss = loss_fn(output[:, -1, :], labels)
            return loss, output
        else:
            return output