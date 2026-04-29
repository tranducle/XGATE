import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Standard DL Temporal: BiLSTM / 1D-CNN 
# ==========================================

class CNN1D_BiLSTM(nn.Module):
    """
    Standard sequential deep learning baseline combining 1D-CNN for 
    local feature extraction and BiLSTM for temporal/sequential dependencies.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        # Input shape expected: [Batch, Channels, Length]
        # For tabular data mapped to a sequence: [Batch, 1, input_dim]
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate length after two poolings of size 2
        reduced_len = input_dim // 4
        
        self.bilstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * reduced_len, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, input_dim]
        Returns:
            Logits of shape [Batch, num_classes]
        """
        # Reshape for Conv1D: [Batch, 1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # BiLSTM expects [Batch, Seq_Len, Features]
        # Currently x is [Batch, Channels, Length]
        x = x.transpose(1, 2)
        
        x, _ = self.bilstm(x)
        
        # Flatten for FC
        x = x.flatten(start_dim=1)
        
        logits = self.fc(x)
        return logits


# ==========================================
# 2. SOTA Lightweight (2025): TBCLNN (1D Variant)
# ==========================================

class TiedBlockConv1d(nn.Module):
    """
    Tied Block Convolution for 1D tabular vector sequences.
    Splits channels into B blocks, applies the SAME convolution filter 
    to each block to share parameters and reduce complexity.
    """
    def __init__(self, channels: int, blocks: int = 4, kernel_size: int = 3):
        super().__init__()
        assert channels % blocks == 0, "Channels must be divisible by blocks"
        self.blocks = blocks
        self.channels_per_block = channels // blocks
        
        # Shared convolution filter
        self.shared_conv = nn.Conv1d(
            in_channels=self.channels_per_block, 
            out_channels=self.channels_per_block, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=self.channels_per_block # Depthwise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x shape: [Batch, channels, length] """
        # Split into blocks along channel dimension
        blocks = torch.split(x, self.channels_per_block, dim=1)
        
        # Apply shared logic
        out_blocks = [self.shared_conv(b) for b in blocks]
        
        # Recombine
        return torch.cat(out_blocks, dim=1)


class TBCLNN(nn.Module):
    """
    Tree-Based Convolutional Lightweight Neural Network (1D Tabular adaptation).
    Utilizes Tied Block Convolutions (TBC) to minimize parameters.
    """
    def __init__(self, input_dim: int, num_classes: int, blocks: int = 4):
        super().__init__()
        # Initial projection to divisible channel size
        self.init_conv = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        
        self.tbc1 = TiedBlockConv1d(channels=64, blocks=blocks)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.tbc2 = TiedBlockConv1d(channels=64, blocks=blocks)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.init_conv(x))
        
        # Residual Block 1
        res = x
        x = F.relu(self.bn1(self.tbc1(x)))
        x = x + res
        
        # Residual Block 2
        res = x
        x = F.relu(self.bn2(self.tbc2(x)))
        x = x + res
        
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits


# ==========================================
# 3. SOTA Transformer IDS (2025): MBConv-ViT (1D Variant)
# ==========================================

class MBConv1D(nn.Module):
    """ Inverted Residual Block (MobileNet) applied to 1D """
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
        self.use_res_connect = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out

class MBConv_ViT_1D(nn.Module):
    """
    Fuses local features from MBConv with global features from a Vision Transformer
    adapted for 1D sequences. 
    """
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        
        # Initial linear mapping to patch embeddings equivalent
        self.patch_proj = nn.Conv1d(1, d_model, kernel_size=1)
        
        # Local Branch: MobileNetV2 Convolutions
        self.local_branch = nn.Sequential(
            MBConv1D(d_model, d_model),
            MBConv1D(d_model, d_model)
        )
        
        # Global Branch: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            batch_first=True
        )
        self.global_branch = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fusion and Classification
        self.fusion_conv = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1) # [Batch, 1, input_dim]
            
        x = self.patch_proj(x) # [Batch, d_model, input_dim]
        
        # 1. Extract Local Features
        local_features = self.local_branch(x) # [Batch, d_model, Length]
        
        # 2. Extract Global Features
        # Transformer expects [Batch, Seq_Len, d_model]
        global_features = x.transpose(1, 2)
        global_features = self.global_branch(global_features)
        global_features = global_features.transpose(1, 2) # Back to [Batch, d_model, Length]
        
        # 3. Feature Fusion (Concatenation)
        fused = torch.cat([local_features, global_features], dim=1) # [Batch, d_model*2, Length]
        fused = self.fusion_conv(fused)
        
        fused = self.pool(fused).squeeze(-1)
        logits = self.classifier(fused)
        return logits
        
# ==========================================
# 4. Traditional ML Ensembles: LightGBM / RF
# ==========================================

class MLBaselineWrapper:
    """
    A wrapper to keep Scikit-Learn/LightGBM models API-compatible 
    (in terms of execution flow) with our PyTorch trainers as much as possible,
    though they will be fitted independently of the epoch loop.
    """
    def __init__(self, model_type="lightgbm", random_state: int = 42):
        self.model_type = model_type
        if model_type == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=random_state,
                n_jobs=-1,
                verbose=-1 # Suppress noisy leaf warnings
            )
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError("Unsupported ML model type.")

    def fit(self, X, y):
        # Convert PyTorch tensors to NumPy arrays if needed
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        self.model.fit(X, y)

    def predict(self, X):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if torch.is_tensor(X):
                X_np = X.cpu().numpy()
            else:
                X_np = X
            preds = self.model.predict(X_np)
        return torch.tensor(preds)

    def predict_proba(self, X):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if torch.is_tensor(X):
                X_np = X.cpu().numpy()
            else:
                X_np = X
            probs = self.model.predict_proba(X_np)
        return torch.tensor(probs)


if __name__ == "__main__":
    # --- Shape Verification ---
    batch_size = 4
    input_dim = 128 # Assume our tabular_embedder resolves to 128
    num_classes = 15
    
    dummy_input = torch.randn(batch_size, input_dim)
    
    # 1. BiLSTM/1D-CNN
    model1 = CNN1D_BiLSTM(input_dim, num_classes)
    out1 = model1(dummy_input)
    print(f"CNN1D_BiLSTM Output: {out1.shape}") # Expected: [4, 15]
    
    # 2. TBCLNN
    model2 = TBCLNN(input_dim, num_classes)
    out2 = model2(dummy_input)
    print(f"TBCLNN Output: {out2.shape}") # Expected: [4, 15]
    
    # 3. MBConv-ViT
    model3 = MBConv_ViT_1D(input_dim, num_classes)
    out3 = model3(dummy_input)
    print(f"MBConv_ViT_1D Output: {out3.shape}") # Expected: [4, 15]
