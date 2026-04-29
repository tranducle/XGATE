from dataclasses import dataclass

@dataclass
class SecurityBERTConfig:
    """
    Configuration for the SecurityBERT and X-GATE architecture variants.
    """
    # ==========================
    # Global Settings
    # ==========================
    num_classes: int = 15      # 1 (Normal) + 14 (Attacks) in Edge-IIoTset
    input_features: int = 49   # Total features post-preprocessing
    
    # ==========================
    # Transformer Architecture 
    # ==========================
    d_model: int = 128         # Hidden dimension
    n_heads: int = 4           # Attention heads
    num_layers: int = 3        # Transformer encoder layers
    dim_feedforward: int = 512 # Feedforward layer inside Transformer
    dropout: float = 0.1       # Dropout probability
    
    # ==========================
    # Variant Definitions 
    # ==========================
    @classmethod
    def get_teacher_config(cls):
        """ Vanilla SecurityBERT (32-bit Full-Precision) """
        return cls(
            d_model=256,
            n_heads=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1
        )
        
    @classmethod
    def get_student_config(cls):
        """ TinySecurityBERT (X-GATE Configuration) """
        return cls(
            d_model=128,
            n_heads=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1
        )
