"""
SAINT: Semantic Attention for Interpretable iNsider Threat Detection
=====================================================================
PyTorch implementation of the SAINT architecture.

Key Components:
- Semantic Multi-Head Attention (SMA): Each head processes one modality
- Temporal Threat Indicator Score (TTIS): Intrinsic explanation generation
- Attention regularization losses for diversity and sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1), :]


class SemanticMultiHeadAttention(nn.Module):
    """
    Semantic Multi-Head Attention (SMA)
    
    Each head processes a specific modality slice of the input features.
    This enables intrinsic interpretability where attention weights 
    directly correspond to threat indicator importance.
    """
    
    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        modality_dims: List[int],
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Total model dimension
            n_heads: Number of attention heads (= number of modalities)
            modality_dims: List of feature dimensions per modality
            dropout: Dropout probability
        """
        super().__init__()
        
        assert n_heads == len(modality_dims), "n_heads must equal number of modalities"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modality_dims = modality_dims
        
        # Per-head projections (one per modality)
        self.W_Q = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(self.d_head, self.d_head) for _ in range(n_heads)
        ])
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_head)
        
        # Store attention weights for explainability
        self.attention_weights = None
    
    def forward(
        self, 
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of Semantic Multi-Head Attention.
        
        Args:
            H: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attended output (batch, seq_len, d_model)
            attention_matrices: List of attention matrices per head
        """
        batch_size, seq_len, _ = H.shape
        
        # Split H into modality-specific representations
        # H^(k) = H[:, :, (k-1)*d_h : k*d_h]
        H_splits = torch.split(H, self.d_head, dim=-1)
        
        heads = []
        attention_matrices = []
        
        for h in range(self.n_heads):
            H_k = H_splits[h]  # (batch, seq_len, d_head)
            
            # Compute Q, K, V for this head
            Q = self.W_Q[h](H_k)  # (batch, seq_len, d_head)
            K = self.W_K[h](H_k)
            V = self.W_V[h](H_k)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            A = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
            A = self.dropout(A)
            
            # Store attention for interpretability
            attention_matrices.append(A.detach())
            
            # Apply attention to values
            head_output = torch.matmul(A, V)  # (batch, seq_len, d_head)
            heads.append(head_output)
        
        # Concatenate heads and project
        concat = torch.cat(heads, dim=-1)  # (batch, seq_len, d_model)
        output = self.W_O(concat)
        
        self.attention_weights = attention_matrices
        
        return output, attention_matrices


class TransformerBlock(nn.Module):
    """Single transformer block with SMA and FFN"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        modality_dims: List[int],
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = SemanticMultiHeadAttention(
            d_model, n_heads, modality_dims, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class SAINT(nn.Module):
    """
    SAINT: Semantic Attention for Interpretable iNsider Threat Detection
    
    A transformer-based model with semantically-assigned attention heads
    for intrinsic interpretability in insider threat detection.
    """
    
    # Define modality names for interpretability
    MODALITY_NAMES = ['Login', 'File', 'Email', 'Device', 'Web']
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 320,
        n_heads: int = 5,
        n_layers: int = 4,
        d_ff: int = 1280,
        seq_len: int = 30,
        dropout: float = 0.1,
        modality_dims: Optional[List[int]] = None
    ):
        """
        Args:
            input_dim: Number of input features
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads (one per modality)
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            seq_len: Sequence length
            dropout: Dropout probability
            modality_dims: Feature dimensions per modality (for semantic assignment)
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers
        
        # If modality_dims not specified, split evenly
        if modality_dims is None:
            modality_dims = [input_dim // n_heads] * n_heads
        self.modality_dims = modality_dims
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        self.input_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, modality_dims, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Attention pooling
        self.pool_attention = nn.Linear(d_model, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Store attention for explainability
        self.all_attention_weights = []
        self.pool_weights = None
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - probs: Classification probabilities
                - attention: (optional) Attention weights per layer/head
                - ttis: (optional) Temporal Threat Indicator Scores
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_encoding(h)
        h = self.input_dropout(h)
        
        # Store attention from all layers
        self.all_attention_weights = []
        
        # Pass through transformer layers
        for layer in self.layers:
            h, attn_weights = layer(h)
            self.all_attention_weights.append(attn_weights)
        
        # Attention pooling
        pool_scores = self.pool_attention(h).squeeze(-1)  # (batch, seq_len)
        beta = F.softmax(pool_scores, dim=-1)  # (batch, seq_len)
        self.pool_weights = beta.detach()
        
        # Weighted sum
        z = torch.bmm(beta.unsqueeze(1), h).squeeze(1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(z).squeeze(-1)  # (batch,)
        probs = torch.sigmoid(logits)
        
        output = {
            'logits': logits,
            'probs': probs
        }
        
        if return_attention:
            output['attention'] = self.all_attention_weights
            output['pool_weights'] = self.pool_weights
            output['ttis'] = self.compute_ttis()
        
        return output
    
    def compute_ttis(self) -> torch.Tensor:
        """
        Compute Temporal Threat Indicator Score (TTIS) for each modality.
        
        TTIS_k = sum_t beta_t * max_t' A^(k)_{t,t'}
        
        Returns:
            ttis: (batch, n_heads) tensor of TTIS scores per modality
        """
        if self.pool_weights is None or len(self.all_attention_weights) == 0:
            return None
        
        # Use last layer's attention
        last_layer_attn = self.all_attention_weights[-1]  # List of (batch, seq, seq)
        beta = self.pool_weights  # (batch, seq)
        
        ttis_scores = []
        for h, A in enumerate(last_layer_attn):
            # Max attention per position
            max_attn = A.max(dim=-1)[0]  # (batch, seq)
            # Weighted by pool weights
            ttis_k = (beta * max_attn).sum(dim=-1)  # (batch,)
            ttis_scores.append(ttis_k)
        
        return torch.stack(ttis_scores, dim=-1)  # (batch, n_heads)
    
    def get_explanation(self, x: torch.Tensor, threshold: float = 0.1) -> List[Dict]:
        """
        Generate human-readable explanations for predictions.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            threshold: TTIS threshold for including in explanation
            
        Returns:
            List of explanation dictionaries per sample
        """
        with torch.no_grad():
            output = self.forward(x, return_attention=True)
        
        ttis = output['ttis']  # (batch, n_heads)
        probs = output['probs']  # (batch,)
        
        explanations = []
        for i in range(x.size(0)):
            exp = {
                'threat_probability': probs[i].item(),
                'indicators': []
            }
            
            for h in range(self.n_heads):
                score = ttis[i, h].item()
                if score > threshold:
                    exp['indicators'].append({
                        'modality': self.MODALITY_NAMES[h],
                        'ttis': score,
                        'rank': None  # Will be filled after sorting
                    })
            
            # Sort by TTIS and assign ranks
            exp['indicators'].sort(key=lambda x: x['ttis'], reverse=True)
            for rank, ind in enumerate(exp['indicators']):
                ind['rank'] = rank + 1
            
            explanations.append(exp)
        
        return explanations


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This focuses training on hard examples and down-weights easy negatives.
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits (batch,)
            targets: Binary targets (batch,)
        """
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SAINTLoss(nn.Module):
    """
    Combined loss function for SAINT model.
    
    L_total = L_cls + lambda_div * L_div + lambda_sparse * L_sparse
    
    Supports both BCE and Focal Loss.
    """
    
    def __init__(
        self,
        lambda_div: float = 0.1,
        lambda_sparse: float = 0.01,
        pos_weight: float = 1.0,
        use_focal: bool = False,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.lambda_div = lambda_div
        self.lambda_sparse = lambda_sparse
        self.use_focal = use_focal
        
        if use_focal:
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: List[List[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            logits: Model logits (batch,)
            targets: Ground truth labels (batch,)
            attention_weights: Nested list [layer][head] of attention matrices
            
        Returns:
            Dictionary with total loss and components
        """
        # Classification loss
        if self.use_focal:
            loss_cls = self.cls_loss(logits, targets.float())
        else:
            loss_cls = self.bce(logits, targets.float())
        
        # Diversity loss (penalize similar attention patterns)
        loss_div = self._diversity_loss(attention_weights)
        
        # Sparsity loss (encourage focused attention)
        loss_sparse = self._sparsity_loss(attention_weights)
        
        # Total loss
        loss_total = loss_cls + self.lambda_div * loss_div + self.lambda_sparse * loss_sparse
        
        return {
            'total': loss_total,
            'cls': loss_cls,
            'div': loss_div,
            'sparse': loss_sparse
        }
    
    def _diversity_loss(self, attention_weights: List[List[torch.Tensor]]) -> torch.Tensor:
        """Compute attention diversity loss"""
        # Use last layer
        last_layer = attention_weights[-1]
        n_heads = len(last_layer)
        
        if n_heads < 2:
            return torch.tensor(0.0, device=last_layer[0].device)
        
        # Flatten attention matrices
        flat = [A.view(A.size(0), -1) for A in last_layer]  # List of (batch, seq*seq)
        
        # Compute pairwise cosine similarity
        div_loss = 0.0
        count = 0
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                sim = F.cosine_similarity(flat[i], flat[j], dim=-1)  # (batch,)
                div_loss = div_loss + F.relu(sim - 0.1).mean()  # Penalize similarity > 0.1
                count += 1
        
        return div_loss / count if count > 0 else torch.tensor(0.0)
    
    def _sparsity_loss(self, attention_weights: List[List[torch.Tensor]]) -> torch.Tensor:
        """Compute attention sparsity loss (entropy minimization)"""
        last_layer = attention_weights[-1]
        
        entropy_sum = 0.0
        for A in last_layer:
            # A: (batch, seq, seq) - already normalized per row
            # Entropy per position: -sum(A * log(A))
            entropy = -torch.sum(A * torch.log(A + 1e-9), dim=-1)  # (batch, seq)
            entropy_sum = entropy_sum + entropy.mean()
        
        return entropy_sum / len(last_layer)


def create_model(input_dim: int, config: Optional[Dict] = None) -> SAINT:
    """Factory function to create SAINT model"""
    
    default_config = {
        'd_model': 320,
        'n_heads': 5,
        'n_layers': 4,
        'd_ff': 1280,
        'seq_len': 30,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return SAINT(input_dim=input_dim, **default_config)


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    seq_len = 30
    input_dim = 15  # Number of features from preprocessing
    
    model = create_model(input_dim)
    print(f"SAINT Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x, return_attention=True)
    
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  TTIS shape: {output['ttis'].shape}")
    
    # Test explanation
    explanations = model.get_explanation(x)
    print(f"\nSample explanation:")
    print(f"  Threat probability: {explanations[0]['threat_probability']:.3f}")
    for ind in explanations[0]['indicators'][:3]:
        print(f"  - {ind['modality']}: TTIS={ind['ttis']:.3f}")
