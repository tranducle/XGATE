# X-GATE: Attribution-Aware Distillation and Hardening for Compressed Edge-IIoT Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** *X-GATE: Attribution-Aware Distillation and Hardening for Compressed Edge-IIoT Intrusion Detection*
> **Authors:** Tran Duc Le, Yida Bao, Mohammad Arifuzzaman
> **Affiliation:** Department of Mathematics, Statistics & Computer Science, University of Wisconsin-Stout
> **Journal:** Electronics (MDPI), 2026

---

## Overview

X-GATE (eXplanation-Guided Adversarial Training Engine) is an attribution-aware training framework for compressed Edge-IIoT intrusion detection. It addresses the critical challenge that model compression can alter the feature-attribution structure learned by a full-precision model.

The framework combines two novel components:

1. **Explanation-Consistency Distillation (ECD):** Aligns Teacher–Student feature-attribution rankings via a differentiable soft-rank Spearman penalty, reducing Logical Drift by 17.24%.
2. **Explanation-Guided Adversarial Training (EGAT):** Hardens the Student on teacher-salient feature coordinates, improving adversarial F1 by 10.57 percentage points.

### Key Results (Edge-IIoTset 2022, 3-seed average)

| Metric | Value |
|--------|-------|
| F1-Macro (Full X-GATE) | 89.30 ± 3.89% |
| Parameters | 0.617M |
| INT8 F1-Macro (deployment) | 79.11 ± 5.47% |
| Adversarial FPR reduction | 0.46% → 0.16% |
| CPU Latency | 1.25 ms/sample |

## Repository Structure

```text
├── .gitignore
├── README.md
├── requirements.txt
├── run_all_experiments.py       # Comprehensive multi-seed training pipeline
├── run_ablation_study.py        # Component-wise ablation study (ECD, EGAT contributions)
├── run_claim_closure.py         # Deployment evidence: INT8, latency, Logical Drift
├── src/
│   ├── model/
│   │   ├── SecurityBERT_Model.py  # Teacher (SecurityBERT) & TinyStudent architectures
│   │   └── baselines.py           # CNN-BiLSTM, TBCLNN, MBConv-ViT baselines
│   ├── training/
│   │   ├── data_loader.py         # Parquet-based data loading with StandardScaler
│   │   ├── trainer.py             # Core training loop with KD + ECD + EGAT
│   │   └── xgate_core.py          # X-GATE loss computation engine
│   ├── visualization/
│   │   └── visualize_results.py   # Publication-quality figure generation
│   └── tools/                     # Utility scripts
├── COLABCODE/                     # Google Colab reproduction package
│   ├── COLAB_SETUP.md             # Colab-specific setup instructions
│   ├── colab_run_canonical.py     # One-click canonical experiment runner
│   ├── colab_run_ablation.py      # One-click ablation runner
│   ├── colab_run_claim_closure.py # One-click claim closure runner
│   └── src/                       # Colab-adapted source modules
└── dataset/                       # Place Edge-IIoTset 2022 data here
```

## System Requirements

- **Python** 3.10+
- **PyTorch** 2.0+ (CUDA recommended for training; CPU for edge simulation)
- **RAM:** ≥ 32 GB recommended (dataset is ~10M samples)

### Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The experiments utilize the **Edge-IIoTset 2022 Cybersecurity Dataset** (15-class attack taxonomy).

1. Download the dataset from [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).
2. Place the CSV files in the `dataset/` directory.
3. The data loader (`src/training/data_loader.py`) handles automatic preprocessing, stratified splitting (50/25/25 train/val/test), and Parquet caching.

## Reproducing Results

### 1. Main Multi-Seed Experiment

Trains the Teacher (SecurityBERT), then distills into TinyStudent with ECD+EGAT across multiple seeds:

```bash
python run_all_experiments.py
```

Outputs are saved to `results/canonical_multirun_fixed_<date>/`.

### 2. Component-Wise Ablation Study

Isolates the contributions of KD-only, KD+ECD, KD+EGAT, and Full X-GATE:

```bash
python run_ablation_study.py
```

### 3. Deployment Claim Closure

Computes INT8 quantization metrics, CPU latency, artifact sizes, and Logical Drift from saved checkpoints:

```bash
python run_claim_closure.py --results-root results/canonical_multirun_fixed_<date>
```

### 4. Google Colab Reproduction

For Colab users, see [`COLABCODE/COLAB_SETUP.md`](COLABCODE/COLAB_SETUP.md) for step-by-step instructions using the pre-configured wrapper scripts.

## Citation

If you find this work useful, please cite:

```bibtex
@article{le2026xgate,
  title={X-GATE: Attribution-Aware Distillation and Hardening for Compressed Edge-IIoT Intrusion Detection},
  author={Le, Tran Duc and Bao, Yida and Arifuzzaman, Mohammad},
  journal={Electronics},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is released under the [MIT License](LICENSE).
