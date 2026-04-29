import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.tabular_embedder import TabularEmbedder
from src.model.config import SecurityBERTConfig

class CustomTransformerLayer(nn.Module):
    """
    Standard Transformer Encoder Layer implemented with basic modules.
    Bypasses the 'AttributeError' bug in built-in PyTorch Transformer during quantization.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # Multi-head attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SecurityBERT(nn.Module):
    """
    Parametrized SecurityBERT (Transformer Encoder) for tabular cybersecurity data.
    Based on the configuration, it acts as the 32-bit Full Precision Teacher 
    or the TinySecurityBERT (Student) architecture.
    """
    def __init__(self, config: SecurityBERTConfig):
        super().__init__()
        self.config = config
        
        # 1. Project Continuous tabular data to token embeddings of size d_model
        self.embedder = TabularEmbedder(
            continuous_dim=config.input_features, 
            d_model=config.d_model
        )
        
        # 2. Main Transformer Encoder Logic
        # batch_first=True makes shape tracking simpler: [Batch, SeqLen, Features]
        # 2. Main Transformer Encoder Logic using custom layers for quantization stability
        # We use a dummy class to mimic nn.TransformerEncoder's structure for state_dict compatibility
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([
            CustomTransformerLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # 3. Global Pooling & Classification Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )

    def forward(self, continuous_x: torch.Tensor) -> torch.Tensor:
        """
        Flow of tensors:
        In:  [Batch, input_features]
        Out: [Batch, num_classes]
        """
        # Step 1: Feature-wise Tokenization [Batch, input_features, d_model]
        # E.g., for Edge-IIoTset: [Batch, 49, d_model]
        # This means Sequence Length = 49.
        seq_input = self.embedder(continuous_x)
        
        # Step 2: Encoder Pass
        # Self-Attention will calculate cross-dependencies BETWEEN the 49 network features.
        for layer in self.transformer.layers:
            seq_input = layer(seq_input)
        encoded = seq_input # [Batch, 49, d_model]
        
        # Step 3: Global Pooling across the 49 features
        # [Batch, 49, d_model] -> [Batch, d_model, 49] -> AdaptiveAvgPool1d(1) -> [Batch, d_model, 1]
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1) # -> [Batch, d_model]
        
        # Step 4: Classifier Logits
        logits = self.classifier(pooled) # [Batch, num_classes]
        return logits

class VanillaSecurityBERT(SecurityBERT):
    """ The massive 32-bit Teacher Model baseline. """
    def __init__(self, num_classes: int = 15, input_features: int = 82):
        config = SecurityBERTConfig.get_teacher_config()
        config.num_classes = num_classes
        config.input_features = input_features
        super().__init__(config)

class TinySecurityBERT(SecurityBERT):
    """ The Lightweight Student model (X-GATE proposed component). """
    def __init__(self, num_classes: int = 15, input_features: int = 82):
        config = SecurityBERTConfig.get_student_config()
        config.num_classes = num_classes
        config.input_features = input_features
        super().__init__(config)
        

if __name__ == "__main__":
    # verification
    batch_size = 8
    in_feat = 82
    dummy_input = torch.randn(batch_size, in_feat)
    
    teacher = VanillaSecurityBERT(num_classes=15, input_features=in_feat)
    student = TinySecurityBERT(num_classes=15, input_features=in_feat)
    
    out_t = teacher(dummy_input)
    out_s = student(dummy_input)
    
    print(f"Teacher outputs: {out_t.shape}") # expect [8, 15]
    print(f"Student outputs: {out_s.shape}") # expect [8, 15]
