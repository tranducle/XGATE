# X-GATE Colab Setup

This bundle is prepared so you can upload code and dataset to Google Drive,
mount Drive in Colab, and run the three experiment stages with clear output
folders.

## 1. Google Drive folder layout

Create this exact structure in `MyDrive`:

```text
MyDrive/
└── XGATE_COLAB/
    ├── CODE/
    │   └── COLABCODE/
    │       ├── src/
    │       ├── run_all_experiments.py
    │       ├── run_ablation_study.py
    │       ├── run_claim_closure.py
    │       ├── colab_run_canonical.py
    │       ├── colab_run_ablation.py
    │       ├── colab_run_claim_closure.py
    │       ├── colab_verify_setup.py
    │       └── requirements_colab.txt
    ├── DATASET/
    │   └── processed/
    │       ├── checkpoint3_test.parquet
    │       ├── checkpoint3_val.parquet
    │       └── final/
    │           └── final_balanced_train.parquet
    └── RUNS/
```

`RUNS/` can start empty. The wrappers will create the subfolders automatically.

## 2. What to upload from your local machine

Upload these two local folders:

1. Code:
   `projects/XGATE/RELATED_DATA/XGATE_Public/COLABCODE`
   Upload to:
   `MyDrive/XGATE_COLAB/CODE/COLABCODE`

2. Processed dataset:
   `projects/XGATE/RELATED_DATA/DATASET/processed`
   Upload to:
   `MyDrive/XGATE_COLAB/DATASET/processed`

Do not upload the raw dataset unless you also want it for archival purposes.
For running the current experiments on Colab, the `processed` folder is the one
that matters.

## 3. Start a Colab notebook

In Colab, switch runtime to GPU:

`Runtime` -> `Change runtime type` -> `GPU`

Then run these cells:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd /content/drive/MyDrive/XGATE_COLAB/CODE/COLABCODE
```

```python
!pip install -q -r requirements_colab.txt
```

```python
!python colab_verify_setup.py
```

If `colab_verify_setup.py` says `Dataset status : READY`, the folder layout is
correct.

## 4. Run the canonical experiment

This saves everything to:
`/content/drive/MyDrive/XGATE_COLAB/RUNS/canonical_multirun`

```python
!python colab_run_canonical.py --n-runs 5
```

Useful variants:

```python
!python colab_run_canonical.py --smoke-test
!python colab_run_canonical.py --n-runs 3 --teacher-epochs 10 --student-epochs 8 --dl-epochs 8
```

Main outputs:

- `RUNS/canonical_multirun/run_all.log`
- `RUNS/canonical_multirun/FINAL_STATS.json`
- `RUNS/canonical_multirun/FINAL_STATS_TABLE.txt`
- per-run checkpoint folders under `RUNS/canonical_multirun/run_*`

## 5. Run the ablation study

This reads teacher checkpoints from the canonical folder and saves ablation
results to:
`/content/drive/MyDrive/XGATE_COLAB/RUNS/ablation`

```python
!python colab_run_ablation.py --n-runs 5
```

Useful variant:

```python
!python colab_run_ablation.py --smoke-test
```

Main outputs:

- `RUNS/ablation/ablation.log`
- `RUNS/ablation/ABLATION_STATS.json`
- `RUNS/ablation/ABLATION_TABLE.txt`

## 6. Run claim-closure evaluation

This reads the canonical checkpoints and writes supplemental deployment /
robustness evidence to:
`/content/drive/MyDrive/XGATE_COLAB/RUNS/claim_closure`

```python
!python colab_run_claim_closure.py
```

Useful variant for a quick dry run:

```python
!python colab_run_claim_closure.py --max-eval-samples 512
```

## 7. Where every result goes

The wrappers deliberately force outputs into three stable folders:

- canonical:
  `MyDrive/XGATE_COLAB/RUNS/canonical_multirun`
- ablation:
  `MyDrive/XGATE_COLAB/RUNS/ablation`
- claim closure:
  `MyDrive/XGATE_COLAB/RUNS/claim_closure`

That means when you come back later, you do not need to guess where logs,
checkpoints, or final tables were written.

## 8. If you want a different Drive root

You can keep the same wrapper scripts and point them somewhere else:

```python
!python colab_run_canonical.py --drive-root /content/drive/MyDrive/MY_OTHER_XGATE_ROOT
```

If you do that, keep the same internal folder logic:

- dataset in `<drive-root>/DATASET/processed`
- results in `<drive-root>/RUNS/...`

## 9. Minimal run order

If you want the shortest reliable sequence:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/XGATE_COLAB/CODE/COLABCODE
!pip install -q -r requirements_colab.txt
!python colab_verify_setup.py
!python colab_run_canonical.py --n-runs 5
!python colab_run_ablation.py --n-runs 5
!python colab_run_claim_closure.py
```
