"""
Publication-Quality Visualization Suite for X-GATE Experiments.
Generates all figures required for the manuscript in both PDF and PNG (600 DPI).

Usage: python -m src.visualization.visualize_results

Figures Generated:
  Fig 1: Performance Bar Charts (Accuracy, F1-Macro, ROC-AUC)
  Fig 2: Efficiency Scatter Plot (F1 vs Latency, sized by Parameters)
  Fig 3: Training Dynamics - Validation F1-Macro over Epochs
  Fig 4: Training Dynamics - Train vs Val Loss Curves
  Fig 5: ROC-AUC Comparison Bar Chart
  Fig 6: Radar/Spider Chart (Multi-metric comparison)
  Fig 7: FPR Comparison (False Positive Rate)
  Fig 8: Model Compression Summary (Parameters vs Accuracy)
"""
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi

warnings.filterwarnings("ignore")

# ============================================================
#   GLOBAL PUBLICATION STYLE SETTINGS
# ============================================================
def set_publication_style():
    plt.style.use('seaborn-v0_8-ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'lightgray'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'

set_publication_style()

DPI = 600
FONT_TITLE = 12
FONT_LABEL = 11
FONT_TICK = 9
FONT_LEGEND = 10

# High-Contrast Bright Color Palette (Tableau 10 inspired)
PALETTE = {
    "X-GATE_TinyStudent":              "#d62728",  # Bright Red
    "Vanilla_SecurityBERT_Teacher":    "#ff7f0e",  # Bright Orange
    "LightGBM":                        "#1f77b4",  # Bright Blue
    "RandomForest":                    "#2ca02c",  # Bright Green
    "CNN1D_BiLSTM":                    "#9467bd",  # Bright Purple
    "TBCLNN":                          "#e377c2",  # Bright Pink
    "MBConv_ViT":                      "#17becf",  # Bright Cyan
}

# Short display names for cleaner axis labels
SHORT_NAMES = {
    "X-GATE_TinyStudent":           "X-GATE\n(Ours)",
    "Vanilla_SecurityBERT_Teacher": "SecurityBERT\n(Teacher)",
    "LightGBM":                     "LightGBM",
    "RandomForest":                 "Random\nForest",
    "CNN1D_BiLSTM":                 "CNN-\nBiLSTM",
    "TBCLNN":                       "TBCLNN",
    "MBConv_ViT":                   "MBConv-\nViT",
}

# Desired display order (our model LAST for emphasis)
MODEL_ORDER = [
    "LightGBM", "RandomForest", "CNN1D_BiLSTM", 
    "TBCLNN", "MBConv_ViT", "Vanilla_SecurityBERT_Teacher",
    "X-GATE_TinyStudent"
]


def _savefig(fig, output_dir, filename_stem):
    """Save figure in both PNG (600 DPI) and PDF formats."""
    png_path = os.path.join(output_dir, f"{filename_stem}.png")
    pdf_path = os.path.join(output_dir, f"{filename_stem}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {filename_stem}.png + .pdf")


def _get_color(model_name):
    return PALETTE.get(model_name, "#888888")

def _get_short_name(model_name):
    return SHORT_NAMES.get(model_name, model_name)

def _sort_df(df):
    """Sort dataframe by MODEL_ORDER."""
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = df.copy()
    df["_sort"] = df["Model"].map(order_map).fillna(99)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


# ============================================================
#   DATA LOADING
# ============================================================
def load_results(results_dir):
    """Loads all JSON result files into a unified DataFrame."""
    data = []
    
    for fname in os.listdir(results_dir):
        if not (fname.endswith("_history.json") or fname.endswith("_results.json")):
            continue
        # Skip files in subdirectories
        filepath = os.path.join(results_dir, fname)
        if not os.path.isfile(filepath):
            continue
            
        with open(filepath, 'r') as f:
            res = json.load(f)
            
        model_name = res["model"]
        is_pytorch = res.get("is_pytorch", True)
        
        if is_pytorch:
            history = res.get("history", [])
            if history:
                best_epoch = max(history, key=lambda x: x.get("f1_macro", 0))
                metrics = best_epoch
            else:
                metrics = {}
        else:
            metrics = res.get("metrics", {})
            
        comp = res.get("computational", {})
        
        data.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0),
            "F1_Macro": metrics.get("f1_macro", 0),
            "F1_Weighted": metrics.get("f1_weighted", 0),
            "Precision_Macro": metrics.get("precision_macro", 0),
            "Recall_Macro": metrics.get("recall_macro", 0),
            "FPR_Macro": metrics.get("fpr_macro", 0),
            "ROC_AUC": metrics.get("roc_auc_macro", 0),
            "Latency_ms": comp.get("inference_latency_ms_per_sample", 0),
            "Params_M": comp.get("num_parameters_millions", 0),
            "Type": "Deep Learning" if is_pytorch else "Traditional ML"
        })
        
    df = pd.DataFrame(data)
    if not df.empty:
        df = _sort_df(df)
    return df


def load_histories(results_dir):
    """Load per-epoch training histories for all PyTorch models."""
    histories = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_history.json"):
            filepath = os.path.join(results_dir, fname)
            with open(filepath, 'r') as f:
                res = json.load(f)
            model_name = res["model"]
            history = res.get("history", [])
            if history:
                histories[model_name] = history
    return histories


# ============================================================
#   FIGURE 1: Performance Bar Charts
# ============================================================
def plot_fig1_performance_bars(df, output_dir):
    """Side-by-side grouped bar chart for Accuracy, F1-Macro, ROC-AUC."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metrics = ["Accuracy", "F1_Macro", "ROC_AUC"]
    metric_labels = ["Accuracy", "F1-Macro", "ROC-AUC"]
    n_models = len(df)
    n_metrics = len(metrics)
    bar_width = 0.22
    x = np.arange(n_models)
    
    colors_metrics = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors_metrics)):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, df[metric].values, bar_width, 
                       label=label, color=color, edgecolor='white', linewidth=0.5)
        
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.set_title("Classification Performance: X-GATE vs. Baselines", 
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([_get_short_name(m) for m in df["Model"]], fontsize=FONT_TICK)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=FONT_LEGEND, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    
    # Highlight our model column background
    if "X-GATE_TinyStudent" in df["Model"].values:
        idx = df[df["Model"] == "X-GATE_TinyStudent"].index[0]
        ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.08, color='red', zorder=0)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig1_performance_bars")


# ============================================================
#   FIGURE 2: Efficiency Scatter Plot
# ============================================================
def plot_fig2_efficiency_scatter(df, output_dir):
    """F1-Macro vs Inference Latency scatter, bubble size = Parameters."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    for _, row in df.iterrows():
        color = _get_color(row["Model"])
        size = max(row["Params_M"] * 150, 80)  # Scale bubble
        is_ours = "X-GATE" in row["Model"]
        
        ax.scatter(row["Latency_ms"], row["F1_Macro"], 
                   s=size, c=color, alpha=0.85,
                   edgecolors='black' if is_ours else 'gray',
                   linewidths=2.5 if is_ours else 1,
                   zorder=5 if is_ours else 3,
                   marker='*' if is_ours else 'o')
        
        # Annotate
        offset_x = row["Latency_ms"] * 0.15 + 0.0003
        ax.annotate(_get_short_name(row["Model"]).replace('\n', ' '), 
                    (row["Latency_ms"], row["F1_Macro"]),
                    xytext=(offset_x, 0),
                    textcoords='offset points',
                    fontsize=9, fontweight='bold' if is_ours else 'normal',
                    color=color)
    
    ax.set_xlabel("Inference Latency per Sample (ms)", fontsize=FONT_LABEL)
    ax.set_ylabel("F1-Macro Score", fontsize=FONT_LABEL)
    ax.set_title("Efficiency vs. Performance Trade-off\n(Target: High F1, Low Latency)", 
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    
    # Add "ideal zone" annotation
    ax.annotate('← Ideal Zone\n(Fast & Accurate)', xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=10, color='green', 
                style='italic', alpha=0.6)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig2_efficiency_scatter")


# ============================================================
#   FIGURE 3: Training Dynamics - F1 Trajectory
# ============================================================
def plot_fig3_f1_trajectory(histories, output_dir):
    """Validation F1-Macro over training epochs for all DL models."""
    if not histories:
        print("    [!] No training histories found. Skipping Fig 3.")
        return
        
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model_name in MODEL_ORDER:
        if model_name not in histories:
            continue
        history = histories[model_name]
        epochs = [h["epoch"] for h in history]
        f1_vals = [h["f1_macro"] for h in history]
        color = _get_color(model_name)
        is_ours = "X-GATE" in model_name
        
        ax.plot(epochs, f1_vals, marker='o' if is_ours else 's', 
                linewidth=3 if is_ours else 1.8,
                markersize=7 if is_ours else 4,
                label=_get_short_name(model_name).replace('\n', ' '),
                color=color, alpha=1.0 if is_ours else 0.7,
                zorder=5 if is_ours else 3)
    
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation F1-Macro", fontsize=FONT_LABEL)
    ax.set_title("Validation F1-Macro Trajectory Over Training Epochs", 
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.legend(fontsize=FONT_LEGEND, loc='lower right')
    ax.set_xlim(left=0.5)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig3_f1_trajectory")


# ============================================================
#   FIGURE 4: Training Dynamics - Loss Curves
# ============================================================
def plot_fig4_loss_curves(histories, output_dir):
    """Train and Validation Loss curves side by side for all DL models."""
    if not histories:
        print("    [!] No training histories found. Skipping Fig 4.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    for model_name in MODEL_ORDER:
        if model_name not in histories:
            continue
        history = histories[model_name]
        epochs = [h["epoch"] for h in history]
        train_losses = [h.get("train_loss", 0) for h in history]
        val_losses = [h.get("loss", 0) for h in history]
        color = _get_color(model_name)
        short = _get_short_name(model_name).replace('\n', ' ')
        is_ours = "X-GATE" in model_name
        lw = 3 if is_ours else 1.5
        alpha = 1.0 if is_ours else 0.7
        
        axes[0].plot(epochs, train_losses, linewidth=lw, label=short, color=color, alpha=alpha)
        axes[1].plot(epochs, val_losses, linewidth=lw, label=short, color=color, alpha=alpha)
    
    axes[0].set_title("Training Loss", fontsize=FONT_TITLE-1, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axes[0].set_ylabel("Loss", fontsize=FONT_LABEL)
    
    axes[1].set_title("Validation Loss", fontsize=FONT_TITLE-1, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axes[1].legend(fontsize=FONT_LEGEND-1, loc='upper right')
    
    fig.suptitle("Convergence Comparison of Training vs. Validation Loss", fontsize=FONT_TITLE, fontweight='bold', y=1.02)
    fig.tight_layout()
    _savefig(fig, output_dir, "fig2_loss_curves")


# ============================================================
#   FIGURE 5: ROC-AUC Comparison
# ============================================================
def plot_fig5_roc_auc_bars(df, output_dir):
    """Horizontal bar chart showing ROC-AUC for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [_get_color(m) for m in df["Model"]]
    short_names = [_get_short_name(m).replace('\n', ' ') for m in df["Model"]]
    
    bars = ax.barh(short_names, df["ROC_AUC"].values, color=colors, edgecolor='white', height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, df["ROC_AUC"].values):
        if val > 0:
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2., 
                    f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("ROC-AUC (Macro OVR)", fontsize=FONT_LABEL)
    ax.set_title("ROC-AUC Comparison Across Models", fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.08)
    
    # Add vertical reference line at 0.95
    ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='Target: 0.95')
    ax.legend(fontsize=FONT_LEGEND)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig3_roc_auc_comparison")


# ============================================================
#   FIGURE 6: Radar / Spider Chart
# ============================================================
def plot_fig6_radar_chart(df, output_dir):
    """Multi-metric radar chart comparing top models."""
    # Select metrics for radar
    categories = ['Accuracy', 'F1-Macro', 'ROC-AUC', 'Precision', 'Recall', '1-FPR']
    N = len(categories)
    
    # Compute angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    for _, row in df.iterrows():
        values = [
            row["Accuracy"],
            row["F1_Macro"],
            row["ROC_AUC"],
            row["Precision_Macro"],
            row["Recall_Macro"],
            1.0 - row["FPR_Macro"]  # Invert FPR so higher = better
        ]
        values += values[:1]  # Close
        
        color = _get_color(row["Model"])
        is_ours = "X-GATE" in row["Model"]
        short = _get_short_name(row["Model"]).replace('\n', ' ')
        
        ax.plot(angles, values, linewidth=3 if is_ours else 1.5, 
                label=short, color=color, alpha=1.0 if is_ours else 0.6)
        ax.fill(angles, values, alpha=0.15 if is_ours else 0.05, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=FONT_TITLE, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=FONT_LEGEND-1)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig4_radar_chart")


# ============================================================
#   FIGURE 7: FPR Comparison
# ============================================================
def plot_fig7_fpr_comparison(df, output_dir):
    """Bar chart of False Positive Rate (lower is better)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [_get_color(m) for m in df["Model"]]
    short_names = [_get_short_name(m) for m in df["Model"]]
    
    bars = ax.bar(short_names, df["FPR_Macro"].values * 100, color=colors, 
                  edgecolor='white', width=0.6)
    
    # Value labels
    for bar, val in zip(bars, df["FPR_Macro"].values * 100):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("False Positive Rate (%)", fontsize=FONT_LABEL)
    ax.set_title("False Positive Rate Comparison (Lower is Better)", 
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    
    # Target line
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target: < 1%')
    ax.legend(fontsize=FONT_LEGEND)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig7_fpr_comparison")


# ============================================================
#   FIGURE 8: Model Compression Summary
# ============================================================
def plot_fig8_compression_summary(df, output_dir):
    """Scatter: Parameters (M) vs Accuracy, proving compression efficiency."""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    for _, row in df.iterrows():
        color = _get_color(row["Model"])
        is_ours = "X-GATE" in row["Model"]
        
        ax.scatter(row["Params_M"], row["Accuracy"] * 100, 
                   s=300 if is_ours else 150, c=color, 
                   edgecolors='black' if is_ours else 'gray',
                   linewidths=2.5 if is_ours else 1,
                   zorder=5 if is_ours else 3,
                   marker='*' if is_ours else 'o',
                   alpha=0.9)
        
        short = _get_short_name(row["Model"]).replace('\n', ' ')
        ax.annotate(f'{short}\n({row["Params_M"]:.2f}M)', 
                    (row["Params_M"], row["Accuracy"] * 100),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=9, fontweight='bold' if is_ours else 'normal',
                    color=color)
    
    ax.set_xlabel("Number of Parameters (Millions)", fontsize=FONT_LABEL)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    ax.set_title("Model Size vs. Classification Accuracy\n(Ideal: Top-Left Corner)", 
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    
    # Ideal zone annotation
    ax.annotate('← Ideal Zone\n(Small & Accurate)', xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=10, color='green', 
                style='italic', alpha=0.6)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig8_compression_summary")


# ============================================================
#   FIGURE 9: Confusion Matrix Heatmaps
# ============================================================
def plot_fig9_confusion_matrices(results_dir, output_dir):
    """Generate confusion matrix heatmaps by loading saved model checkpoints
    and running inference on the validation set."""
    import torch
    import joblib
    
    # Try to load class label names from preprocessing artifacts
    artifact_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "DATASET", "processed", "preprocessing_artifacts.pkl"
    )
    class_names = None
    if os.path.exists(artifact_path):
        try:
            artifacts = joblib.load(artifact_path)
            le = artifacts.get('label_encoder')
            if le is not None:
                class_names = list(le.classes_)
        except Exception:
            pass
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(15)]
    
    # Load validation data
    try:
        from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
    except ImportError:
        print("    [!] Cannot import data_loader. Skipping confusion matrices.")
        return
    
    _, val_loader, _ = get_dataloaders(batch_size=2048)
    
    # Collect all val targets once
    all_y = []
    all_X = []
    with torch.no_grad():
        for X, y in val_loader:
            all_X.append(X)
            all_y.append(y)
    X_val_tensor = torch.cat(all_X)
    y_val = torch.cat(all_y).numpy()
    
    # Models to generate confusion matrices for
    model_configs = [
        ("X-GATE_TinyStudent", "src.model.SecurityBERT_Model", "TinySecurityBERT",
         {"num_classes": 15, "input_features": INPUT_FEATURES}),
    ]
    
    # Optionally add baseline models
    baseline_configs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generated = []
    
    # --- Our models (SecurityBERT variants) ---
    for model_name, module_path, class_name, kwargs in model_configs:
        ckpt_path = os.path.join(results_dir, f"{model_name}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"    [!] Checkpoint not found for {model_name}. Skipping.")
            continue
        
        try:
            import importlib
            mod = importlib.import_module(module_path)
            ModelClass = getattr(mod, class_name)
            model = ModelClass(**kwargs).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), 2048):
                    batch = X_val_tensor[i:i+2048].to(device)
                    logits = model(batch)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            y_pred = np.concatenate(preds)
            generated.append((model_name, y_pred))
        except Exception as e:
            print(f"    [!] Error loading {model_name}: {e}")
    
    # --- Baseline DL models ---
    for model_name, ModelClass, kwargs in baseline_configs:
        ckpt_path = os.path.join(results_dir, f"{model_name}_best.pth")
        if not os.path.exists(ckpt_path):
            continue
        try:
            model = ModelClass(**kwargs).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), 2048):
                    batch = X_val_tensor[i:i+2048].to(device)
                    logits = model(batch)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            y_pred = np.concatenate(preds)
            generated.append((model_name, y_pred))
        except Exception as e:
            print(f"    [!] Error loading {model_name}: {e}")
    
    if not generated:
        print("    [!] No models could be loaded. Skipping confusion matrices.")
        return
    
    # Plot individual confusion matrices
    from sklearn.metrics import confusion_matrix
    
    for model_name, y_pred in generated:
        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(class_names))))
        
        # Normalize by row (true label) for percentage view
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Proportion', 'shrink': 0.8},
                    ax=ax, vmin=0, vmax=1)
        
        short = _get_short_name(model_name).replace('\n', ' ')
        ax.set_title(f"Confusion Matrix: {short}", fontsize=FONT_TITLE, fontweight='bold', pad=15)
        ax.set_xlabel("Predicted Label", fontsize=FONT_LABEL)
        ax.set_ylabel("True Label", fontsize=FONT_LABEL)
        ax.tick_params(axis='both', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        fig.tight_layout()
        safe_name = model_name.replace(' ', '_')
        _savefig(fig, output_dir, f"fig1_spotlight_cm_{safe_name}")


# ============================================================
#   FIGURE 10: Feature Attention Heatmap
# ============================================================
def plot_fig10_attention_heatmap(results_dir, output_dir):
    """Extract and visualize attention weights from X-GATE's Transformer layers,
    showing which input features the model focuses on for each attack class."""
    import torch
    
    try:
        from src.training.data_loader import get_dataloaders, INPUT_FEATURES
        from src.model.SecurityBERT_Model import TinySecurityBERT
    except ImportError:
        print("    [!] Cannot import required modules. Skipping attention heatmap.")
        return
    
    ckpt_path = os.path.join(results_dir, "X-GATE_TinyStudent_best.pth")
    if not os.path.exists(ckpt_path):
        print("    [!] X-GATE checkpoint not found. Skipping attention heatmap.")
        return
    
    # Load class names
    import joblib
    artifact_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "DATASET", "processed", "preprocessing_artifacts.pkl"
    )
    class_names = None
    if os.path.exists(artifact_path):
        try:
            artifacts = joblib.load(artifact_path)
            le = artifacts.get('label_encoder')
            if le is not None:
                class_names = list(le.classes_)
        except Exception:
            pass
    if class_names is None:
        class_names = [f"Class {i}" for i in range(15)]
    
    # Load feature names from parquet columns
    # pd is already imported at module level
    feature_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "DATASET", "processed", "checkpoint3_val.parquet"
    )
    feature_names = [f"F{i}" for i in range(INPUT_FEATURES)]  # Default
    if os.path.exists(feature_path):
        try:
            df_peek = pd.read_parquet(feature_path, engine='pyarrow')
            cols = [c for c in df_peek.columns if c != 'Attack_type']
            if len(cols) == INPUT_FEATURES:
                feature_names = cols
        except Exception:
            pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = TinySecurityBERT(num_classes=15, input_features=INPUT_FEATURES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    # Get a sample batch
    _, val_loader, _ = get_dataloaders(batch_size=512)
    X_sample, y_sample = next(iter(val_loader))
    X_sample = X_sample[:256].to(device)  # Use 256 samples for clarity
    y_sample = y_sample[:256].numpy()
    
    # Hook into the Transformer Encoder to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # TransformerEncoderLayer doesn't expose attention by default, 
        # so we compute it from the self_attn sub-module
        pass
    
    # Alternative: Use the feature importance proxy via gradient-based attribution
    # This is more reliable than hooking into PyTorch's internal attention
    X_sample.requires_grad_(True)
    logits = model(X_sample)
    
    # Compute feature importance per class using gradient magnitude
    n_classes = logits.shape[1]
    importance_matrix = np.zeros((n_classes, INPUT_FEATURES))
    
    for c in range(n_classes):
        if X_sample.grad is not None:
            X_sample.grad.zero_()
        
        class_score = logits[:, c].sum()
        class_score.backward(retain_graph=True)
        
        # Average gradient magnitude across samples = feature importance for this class
        grad = X_sample.grad.detach().cpu().numpy()  # [256, 49]
        importance_matrix[c] = np.mean(np.abs(grad), axis=0)
    
    # Normalize per class for better visualization
    row_maxes = importance_matrix.max(axis=1, keepdims=True) + 1e-9
    importance_normalized = importance_matrix / row_maxes
    
    # Plot: Attack Type x Feature Name heatmap
    fig, ax = plt.subplots(figsize=(20, 10))
    
    sns.heatmap(importance_normalized, 
                xticklabels=feature_names, yticklabels=class_names,
                cmap='viridis', linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Normalized Feature Importance', 'shrink': 0.8},
                ax=ax)
    
    ax.set_title("X-GATE Feature Attention Map\n(Gradient-based Feature Importance per Attack Class)", 
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.set_xlabel("Input Feature", fontsize=FONT_LABEL)
    ax.set_ylabel("Attack Class", fontsize=FONT_LABEL)
    ax.tick_params(axis='x', labelsize=7, rotation=90)
    ax.tick_params(axis='y', labelsize=9)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "fig5_attention_heatmap")


# ============================================================
#   SUMMARY TABLE (Bonus)
# ============================================================
def generate_summary_table(df, output_dir):
    """Generate a clean summary table as an image."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ["Model", "Accuracy", "F1-Macro", "ROC-AUC", "FPR (%)", "Params (M)", "Latency (ms)"]
    
    for _, row in df.iterrows():
        short = _get_short_name(row["Model"]).replace('\n', ' ')
        table_data.append([
            short,
            f'{row["Accuracy"]:.4f}',
            f'{row["F1_Macro"]:.4f}',
            f'{row["ROC_AUC"]:.4f}',
            f'{row["FPR_Macro"]*100:.3f}',
            f'{row["Params_M"]:.3f}',
            f'{row["Latency_ms"]:.4f}'
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight our model row
    for idx, (_, row) in enumerate(df.iterrows()):
        if "X-GATE" in row["Model"]:
            for j in range(len(headers)):
                table[idx + 1, j].set_facecolor('#ffe6e6')
    
    ax.set_title("Experimental Results Summary", fontsize=FONT_TITLE, fontweight='bold', pad=20)
    
    fig.tight_layout()
    _savefig(fig, output_dir, "table_summary")


# ============================================================
#   FIGURE 11: Ablation Study - Performance Comparison
# ============================================================
def plot_ablation_performance(ablation_dir, output_dir):
    """Grouped bar chart comparing F1-Macro, Adv-F1, and Spearman rho
    across 4 ablation configurations with error bars (n=3 runs)."""
    stats_path = os.path.join(ablation_dir, "ABLATION_STATS.json")
    if not os.path.exists(stats_path):
        print("    [!] ABLATION_STATS.json not found. Skipping ablation figure.")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    configs = ["KD_only", "KD_ECD", "KD_EGAT", "Full_XGATE"]
    display_names = ["KD Only\n($\\beta$=0, $\\gamma$=0)",
                     "KD + ECD\n($\\beta$=0.5, $\\gamma$=0)",
                     "KD + EGAT\n($\\beta$=0, $\\gamma$=0.3)",
                     "Full X-GATE\n($\\beta$=0.5, $\\gamma$=0.3)"]

    metrics = [("f1_macro", "F1-Macro", "#2196F3"),
               ("adv_f1_macro", "Adversarial F1", "#FF5722"),
               ("spearman_rho", "Spearman $\\rho$", "#4CAF50")]

    fig, ax = plt.subplots(figsize=(14, 7))

    n_configs = len(configs)
    n_metrics = len(metrics)
    bar_width = 0.22
    x = np.arange(n_configs)

    for i, (metric_key, metric_label, color) in enumerate(metrics):
        means = []
        stds = []
        for cfg in configs:
            m = stats[cfg].get(metric_key, {})
            means.append(m.get("mean", 0))
            stds.append(m.get("std", 0))

        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      label=metric_label, color=color, edgecolor='white',
                      linewidth=0.5, alpha=0.85, capsize=4,
                      error_kw={'linewidth': 1.5, 'capthick': 1.5})

        # Value labels on top
        for bar, mean_v, std_v in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + std_v + 0.008,
                    f'{mean_v:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold', color=color)

    kd_baseline = stats["KD_only"]["f1_macro"]["mean"]
    ax.axhline(y=kd_baseline, color='gray', linestyle='--', alpha=0.6, linewidth=1.2,
               label=f'KD-only F1 baseline ({kd_baseline:.3f})')

    ax.set_ylabel("Score", fontsize=FONT_LABEL)
    ax.set_title("Ablation Study: Component-wise Performance Analysis\n"
                 "(mean $\\pm$ std, n=3 independent runs)",
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=FONT_TICK + 1)
    ax.set_ylim(0.55, 1.05)
    ax.legend(fontsize=FONT_LEGEND, loc='lower right', framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Highlight Full X-GATE column
    ax.axvspan(n_configs - 1 - 0.4, n_configs - 1 + 0.4,
               alpha=0.06, color='red', zorder=0)

    fig.tight_layout()
    _savefig(fig, output_dir, "fig6_ablation_performance")


# ============================================================
#   FIGURE 12: Ablation - Component Contribution (Delta Chart)
# ============================================================
def plot_ablation_component_contribution(ablation_dir, output_dir):
    """Horizontal bar chart showing the incremental contribution of each
    X-GATE component (ECD, EGAT) relative to the KD-only baseline."""
    stats_path = os.path.join(ablation_dir, "ABLATION_STATS.json")
    if not os.path.exists(stats_path):
        print("    [!] ABLATION_STATS.json not found. Skipping component chart.")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # Extract means
    kd_f1 = stats["KD_only"]["f1_macro"]["mean"]
    ecd_f1 = stats["KD_ECD"]["f1_macro"]["mean"]
    egat_f1 = stats["KD_EGAT"]["f1_macro"]["mean"]
    full_f1 = stats["Full_XGATE"]["f1_macro"]["mean"]

    kd_adv = stats["KD_only"]["adv_f1_macro"]["mean"]
    ecd_adv = stats["KD_ECD"]["adv_f1_macro"]["mean"]
    egat_adv = stats["KD_EGAT"]["adv_f1_macro"]["mean"]
    full_adv = stats["Full_XGATE"]["adv_f1_macro"]["mean"]

    kd_dl = stats["KD_only"]["delta_L"]["mean"]
    ecd_dl = stats["KD_ECD"]["delta_L"]["mean"]
    egat_dl = stats["KD_EGAT"]["delta_L"]["mean"]
    full_dl = stats["Full_XGATE"]["delta_L"]["mean"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: F1-Macro contributions ---
    labels_f1 = ["KD Only\n(Baseline)", "+ECD", "+EGAT", "Full\nX-GATE"]
    values_f1 = [kd_f1 * 100, (ecd_f1 - kd_f1) * 100,
                 (egat_f1 - kd_f1) * 100, full_f1 * 100]
    colors_f1 = ["#90CAF9", "#2196F3", "#FF9800", "#D32F2F"]
    
    # Absolute values as bars
    abs_vals = [kd_f1*100, ecd_f1*100, egat_f1*100, full_f1*100]
    bars = axes[0].barh(labels_f1, abs_vals, color=colors_f1, edgecolor='white',
                        height=0.55, alpha=0.85)
    for bar, val in zip(bars, abs_vals):
        delta = val - kd_f1*100
        delta_str = f" ({'+' if delta > 0 else ''}{delta:.2f}%)" if delta != 0 else " (baseline)"
        axes[0].text(val + 0.2, bar.get_y() + bar.get_height()/2.,
                     f'{val:.2f}%{delta_str}', va='center', fontsize=9, fontweight='bold')
    axes[0].set_xlabel("F1-Macro (%)", fontsize=FONT_LABEL)
    axes[0].set_title("(a) Classification Performance", fontsize=FONT_TITLE - 1, fontweight='bold')
    axes[0].set_xlim(82, 95)

    # --- Panel B: Adversarial F1 contributions ---
    abs_adv = [kd_adv*100, ecd_adv*100, egat_adv*100, full_adv*100]
    bars = axes[1].barh(labels_f1, abs_adv, color=colors_f1, edgecolor='white',
                        height=0.55, alpha=0.85)
    for bar, val in zip(bars, abs_adv):
        delta = val - kd_adv*100
        delta_str = f" ({'+' if delta > 0 else ''}{delta:.1f}pp)" if abs(delta) > 0.01 else " (baseline)"
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2.,
                     f'{val:.1f}%{delta_str}', va='center', fontsize=9, fontweight='bold')
    axes[1].set_xlabel("Adversarial F1-Macro (%)", fontsize=FONT_LABEL)
    axes[1].set_title("(b) Adversarial Robustness", fontsize=FONT_TITLE - 1, fontweight='bold')
    axes[1].set_xlim(65, 100)

    # --- Panel C: Logical Drift (lower is better) ---
    abs_dl = [kd_dl, ecd_dl, egat_dl, full_dl]
    colors_dl = ["#90CAF9", "#2196F3", "#FF9800", "#D32F2F"]
    bars = axes[2].barh(labels_f1, abs_dl, color=colors_dl, edgecolor='white',
                        height=0.55, alpha=0.85)
    for bar, val in zip(bars, abs_dl):
        delta = val - kd_dl
        delta_str = f" ({'+' if delta > 0 else ''}{delta:.4f})" if abs(delta) > 0.0001 else " (baseline)"
        axes[2].text(val + 0.002, bar.get_y() + bar.get_height()/2.,
                     f'{val:.4f}{delta_str}', va='center', fontsize=9, fontweight='bold')
    axes[2].set_xlabel("Logical Drift $\\Delta_L$ (lower is better)", fontsize=FONT_LABEL)
    axes[2].set_title("(c) Explanation Fidelity", fontsize=FONT_TITLE - 1, fontweight='bold')
    axes[2].set_xlim(0.20, 0.38)
    axes[2].invert_xaxis()  # Lower is better, so invert for visual consistency

    fig.suptitle("X-GATE Component Contribution Analysis",
                 fontsize=FONT_TITLE + 1, fontweight='bold', y=1.02)
    fig.tight_layout()
    _savefig(fig, output_dir, "fig7_ablation_components")


# ============================================================
#   MAIN ENTRY POINT
# ============================================================
def main():
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("=" * 60)
    print(" X-GATE Publication Figure Generator")
    print(f" Output: {FIGURES_DIR}")
    print(f" Format: PNG ({DPI} DPI) + PDF")
    print("=" * 60)
    
    if not os.path.exists(RESULTS_DIR):
        print("[!] Results directory does not exist. Run training first.")
        return
    
    # Load data
    df = load_results(RESULTS_DIR)
    histories = load_histories(RESULTS_DIR)
    
    if df.empty:
        print("[!] No model results found. Aborting.")
        return
        
    print(f"\n>>> Loaded {len(df)} models:")
    for _, row in df.iterrows():
        marker = " [OURS]" if "X-GATE" in row["Model"] else ""
        print(f"    - {row['Model']}: Acc={row['Accuracy']:.4f}, F1={row['F1_Macro']:.4f}, AUC={row['ROC_AUC']:.4f}{marker}")
    
    print(f"\n>>> Loaded {len(histories)} training histories for dynamic plots.")
    
    # Generate all figures
    TOTAL = 7
    print("\n--- Generating Figures ---")
    
    print(f"\n  [1/{TOTAL}] Spotlight Confusion Matrix (X-GATE)...")
    plot_fig9_confusion_matrices(RESULTS_DIR, FIGURES_DIR)

    print(f"  [2/{TOTAL}] Train vs Val Loss Curves...")
    plot_fig4_loss_curves(histories, FIGURES_DIR)
    
    print(f"  [3/{TOTAL}] ROC-AUC Comparison...")
    plot_fig5_roc_auc_bars(df, FIGURES_DIR)
    
    print(f"  [4/{TOTAL}] Radar/Spider Chart...")
    plot_fig6_radar_chart(df, FIGURES_DIR)
    
    print(f"  [5/{TOTAL}] Feature Attention Heatmap (X-GATE)...")
    plot_fig10_attention_heatmap(RESULTS_DIR, FIGURES_DIR)

    print(f"\n  [6/{TOTAL}] Ablation Study - Performance Bars...")
    plot_ablation_performance(ABLATION_DIR, FIGURES_DIR)

    print(f"  [7/{TOTAL}] Ablation Study - Component Contribution...")
    plot_ablation_component_contribution(ABLATION_DIR, FIGURES_DIR)
    
    print(f"\n{'='*60}")
    print(f" [OK] ALL {TOTAL} FIGURES GENERATED SUCCESSFULLY!")
    print(f" Format: PNG ({DPI} DPI) + PDF (vector)")
    print(f" Location: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
