import torch
import torch.nn as nn
from src.model.tabular_embedder import TabularEmbedder
from src.model.config import SecurityBERTConfig

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
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
        encoded = self.transformer(seq_input) # [Batch, 49, d_model]
        
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
