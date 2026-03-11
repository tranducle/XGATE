# X-GATE: Securing Edge-IIoT Networks via Explanation-Guided Adversarial Training and Logical Consistency Distillation

This repository contains the official code for the anonymous peer-review submission of **X-GATE**.

X-GATE is an ultra-lightweight, 8-bit quantized trust-aware intrusion detection framework designed specifically for resource-constrained Edge-IIoT micro-controllers. The framework solves the diagnostic and adversarial limitations of generic deep learning models by enforcing a logical stability protocol via Explanation-Consistency Distillation (ECD) and Explanation-Guided Adversarial Training (EGAT).

## Repository Structure

```text
├── dataset/                     # Directory for Edge-IIoTset 2022 dataset (CSV format)
├── results/                     # Directory where outputs, logs, and figures are saved
├── src/                         # Core framework logic
│   ├── model.py                 # Neural architectures (Teacher, Vanilla Student, TinyStudent)
│   ├── train.py                 # Core X-GATE Training Logic (Algorithm 1 implementation)
│   ├── data/                    # Data loaders and tabular preprocessing pipelines
│   ├── utils/                   # Helpers, config, and explanation extractors (InputxGradients)
│   ├── training/                # Optimization and KD modules
│   └── evaluation/              # Test-time routines (Adversarial attacks, Logical Drift, ROC)
├── run_ablation_study.py        # Reproduces the Component-Wise Ablation Study (Table IV)
├── run_all_experiments.py       # Comprehensive Multi-Run robust training pipeline
└── requirements.txt             # Python dependencies
```

## System Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended for training; inference operates heavily on CPU for Edge simulation)

To install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The experiments utilize the **Edge-IIoTset 2022 Cybersecurity Dataset**.

1. Download the tabular CSV version of the dataset from the official public repository:
   - [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
2. Place the dataset files within the `dataset/` directory.
3. Ensure the core label columns correspond to the 15-class attack families outlined in the manuscript.

## Run Instructions

### 1. Reproducing the Main Results

To run the primary X-GATE training protocol, which involves initializing the Teacher, extracting explanations, and distiling into the 8-bit quantized TinyStudent under the EGAT protocol:

```bash
python run_all_experiments.py
```

*Outputs, test metrics (F1-Macro, ROC-AUC), and the Spotlight Confusion Matrix will be generated within the `results/multirun/` directory.*

### 2. Reproducing the Component-Wise Ablation

To verify the individual contributions of Standard KD, ECD, and EGAT toward the final adversarial robustness and Logical Drift ($\Delta_L$):

```bash
python run_ablation_study.py
```

*Outputs and ablation logs will be stored in `results/ablation/`.*

## Important Note for Reviewers

In accordance with double-blind review policies, all authors' names, affiliations, and identifying metadata have been entirely stripped from this repository. The codebase focuses exclusively on reproducible algorithmic fidelity as described in the accompanying manuscript.
