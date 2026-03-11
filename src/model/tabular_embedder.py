import torch
import torch.nn as nn
from typing import List

class TabularEmbedder(nn.Module):
    """
    Feature-wise Tokenization Embedder for Tabular Data.
    Projects each of the `continuous_dim` features independently into a `d_model`
    dimensional latent space.
    """
    def __init__(self, continuous_dim: int, d_model: int):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.d_model = d_model
        
        # Create an independent projection layer for EACH feature
        # Weight shape: [continuous_dim, d_model]
        # Bias shape: [continuous_dim, d_model]
        # This allows each feature to have its own dynamic embedding pathway
        self.feature_weights = nn.Parameter(torch.empty(continuous_dim, d_model))
        nn.init.xavier_uniform_(self.feature_weights)
        self.feature_biases = nn.Parameter(torch.zeros(continuous_dim, d_model))
        
        # Layer normalization across the feature embeddings
        self.ln = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, continuous_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous_x: shape [Batch, continuous_dim]
        Returns:
            embedded: shape [Batch, continuous_dim, d_model]
        """
        # continuous_x is [B, D]. We want to multiply each feature value x_i 
        # by the corresponding weight vector W_i [d_model]
        
        # Reshape to [B, D, 1]
        x_expanded = continuous_x.unsqueeze(-1)
        
        # Broadcast multiply: [B, D, 1] * [1, D, d_model] -> [B, D, d_model]
        x_emb = x_expanded * self.feature_weights.unsqueeze(0)
        
        # Add biases: [B, D, d_model] + [1, D, d_model]
        x_emb = x_emb + self.feature_biases.unsqueeze(0)
        
        x_emb = self.ln(x_emb)
        x_emb = self.act(x_emb)
        
        return x_emb
