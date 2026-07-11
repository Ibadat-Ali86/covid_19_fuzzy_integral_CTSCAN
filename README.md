# TopoKD Research Repository

**Production-oriented, self-contained PyTorch research package for topology-aware knowledge distillation in binary COVID-19 chest CT classification.**

**Authors:** Ibadat Ali, Shawaiz Ali, Muhammad Abdullah, Muhammad Usama  
**Affiliation:** Department of Computer Science, GIFT University, Gujranwala, Pakistan

> Research software only. The generated predictions are not clinical diagnoses.

## 1. What this repository contains

This repository consolidates the completed and attempted topology-aware research methods into one reproducible codebase:

| Family | Core method | Status in this repository |
|---|---|---|
| Standard Custom CNN + TDA | Conventional CNN embedding + fixed persistent-homology descriptor | Complete baseline |
| TopoLite-KD | Lightweight depthwise-separable CNN + 134-D topology + reliability-aware fusion + EfficientNet-B0 KD | Main complete method |
| TopoLite-MSF-KD | 144 multi-scale/multi-filtration topology tokens + H0/H1 Transformers + cross-attention + FiLM + routed experts | Complete extended/negative experiment |
| TopoLite-FKD-SAM | Response KD + feature KD + Sharpness-Aware Minimization | Complete framework; exact archived weights unavailable |
| TopoFM-Slice-v1 | DINOv2 + learnable topology tokens + bidirectional cross-attention + supervised contrastive learning | Component-faithful slice-level reconstruction |
| A0–A10 | Visual, topology, fusion, KD, attention, homology, and filtration ablations | Complete configuration suite |

The repository explicitly separates:

- **generated experiment outputs** under `results/`;
- **author-supplied historical records** under `research_records/`;
- **unknown archived values** documented with `[UPPERCASE_BRACKETS]` in templates.

Historical values are never injected into generated metrics.

## 2. Paper-aligned TopoLite-KD architecture

```text
Input: grayscale CT slice, 1×224×224

VISUAL BRANCH
  Conv 3×3/s2 + GroupNorm + SiLU, 24 channels
  Depthwise-separable residual stages: 32 → 48 → 96 → 160
  Coordinate Attention after stages 2 and 3
  Global average pooling + projection → 64-D visual embedding

TOPOLOGY BRANCH
  Resize original grayscale image to 64×64
  Cubical persistence: sublevel I and superlevel 1-I
  H0 connected components + H1 loops
  134-D fixed descriptor
  MLP 134 → 128 → 64

FUSION
  Per-feature reliability gate
  Weighted visual/topology blend
  Multiplicative visual–topology interaction
  64-D fused embedding → binary logit

DISTILLATION
  EfficientNet-B0 teacher
  Temperature T=3
  0.60 fused BCE + 0.30 response KD
  + 0.05 visual auxiliary BCE + 0.05 topology auxiliary BCE
```

The current implementation has **196,773 trainable parameters**, close to the historical approximate count without adding non-functional padding.

## 3. Repository layout

```text
TopoKD_Research_Repository/
├── README.md
├── CODEBASE_AUDIT.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── Makefile
├── train.py
├── evaluate.py
├── visualize.py
├── data_loader.py
├── utils.py
├── ablation_studies.py
├── configs/
│   ├── base.yaml
│   ├── baseline_cnn_tda.yaml
│   ├── topolite_kd.yaml
│   ├── topolite_msf_kd.yaml
│   ├── topolite_fkd_sam.yaml
│   ├── topofm_slice_v1.yaml
│   ├── archive_override_template.yaml.example
│   └── ablations/
│       ├── A0_visual_only.yaml
│       ├── A1_tda_only.yaml
│       ├── A2_concat.yaml
│       ├── A3_fixed_fusion.yaml
│       ├── A4_gated_no_kd.yaml
│       ├── A5_visual_kd.yaml
│       ├── A6_full_topolite_kd.yaml
│       ├── A7_no_coordinate_attention.yaml
│       ├── A8_h0_only.yaml
│       ├── A9_h1_only.yaml
│       ├── A10_sublevel_only.yaml
│       └── A10_superlevel_only.yaml
├── src/topokd/
│   ├── data/                 # discovery, hashing, splitting, transforms, datasets
│   ├── topology/             # cubical PH, 134-D descriptors, 144-token extraction, cache
│   ├── models/               # TopoLite, MSF, TopoFM, teacher, blocks, fusion
│   ├── losses/               # BCE, response KD, feature KD, SupCon
│   ├── optim/                # AdamW/SGD/SAM, schedulers
│   ├── engine/               # training, inference, calibration, final evaluation
│   ├── evaluation/           # metrics, bootstrap CIs, paired significance tests
│   ├── visualization/        # curves, diagnostics, Grad-CAM
│   └── utils/                # seeds, I/O, logging, hashes, checkpoints, environment
├── scripts/
│   ├── run_pipeline.py
│   ├── prepare_manifest.py
│   ├── build_tda_cache.py
│   ├── train_teacher.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── infer.py
│   ├── run_ablation_suite.py
│   ├── aggregate_seeds.py
│   ├── compare_models.py
│   ├── profile_model.py
│   ├── export_paper_tables.py
│   ├── audit_run.py
│   ├── package_run.py
│   └── validate_install.py
├── research_records/         # archival results; never used as generated predictions
├── docs/
├── tests/
├── notebooks/Kaggle_Quickstart.ipynb
└── results/                  # generated run artifacts
```

## 4. Installation

### Kaggle

```bash
cd /kaggle/working/TopoKD_Research_Repository
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Colab or local Linux

```bash
git clone [REPOSITORY_URL]
cd TopoKD_Research_Repository
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The default Kaggle dataset root is already configured as:

```text
/kaggle/input/datasets/plameneduardo/sarscov2-ctscan-dataset
```

Override any configuration value without editing source code:

```bash
python scripts/prepare_manifest.py \
  --config configs/topolite_kd.yaml \
  --set data.root=/[DATASET_DIRECTORY]
```

## 5. End-to-end reproducible workflow

Run the complete dependency-aware workflow with one command:

```bash
python scripts/run_pipeline.py --config configs/topolite_kd.yaml --device cuda
```

It reuses an existing frozen manifest, builds the matching topology cache, trains the teacher only when KD is enabled and no checkpoint exists, and then trains/evaluates the student.

### 5.1 Prepare or restore the frozen manifest

For exact reproduction, restore the archived file:

```text
artifacts/split_manifest.csv
```

To generate a new leakage-audited split:

```bash
python scripts/prepare_manifest.py --config configs/topolite_kd.yaml
```

The command:

- discovers supported image files recursively;
- computes SHA-256 hashes;
- stops on cross-label duplicate conflicts;
- removes exact same-label duplicates when configured;
- creates stratified train/validation/test assignments;
- performs patient-group splitting when `data.patient_id_regex` is supplied;
- refuses to overwrite an existing frozen manifest unless explicitly forced.

### 5.2 Precompute topology and fit train-only standardization

```bash
python scripts/build_tda_cache.py --config configs/topolite_kd.yaml
```

The topology cache path is configuration-hashed. Descriptor mean and standard deviation are fitted **only on the training split** and reused unchanged for validation and test samples.

For the 144-token experiment:

```bash
python scripts/build_tda_cache.py --config configs/topolite_msf_kd.yaml
```

### 5.3 Train the EfficientNet-B0 teacher

```bash
python scripts/train_teacher.py --config configs/topolite_kd.yaml
```

Expected checkpoint location:

```text
artifacts/teacher_best.pt
```

For offline ImageNet initialization, set `model.teacher.backbone_checkpoint=/[EFFICIENTNET_B0_WEIGHTS_FILE]`; alternatively attach the archived trained teacher checkpoint.

KD-enabled experiments stop with a clear error if the teacher checkpoint is missing. Random-teacher distillation is not allowed.

### 5.4 Train a student model

```bash
python scripts/train.py --config configs/topolite_kd.yaml
```

Other implemented techniques:

```bash
python scripts/train.py --config configs/baseline_cnn_tda.yaml
python scripts/train.py --config configs/topolite_msf_kd.yaml
python scripts/train.py --config configs/topolite_fkd_sam.yaml
python scripts/train.py --config configs/topofm_slice_v1.yaml
```

### 5.5 Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --config configs/topolite_kd.yaml \
  --checkpoint results/topolite_kd/seed_42/checkpoints/best.pt
```

Temperature scaling and decision-threshold selection are fitted on validation predictions only. The frozen test split is evaluated once with those fixed values.

### 5.6 Generate qualitative explanations

```bash
python scripts/visualize.py \
  --config configs/topolite_kd.yaml \
  --checkpoint results/topolite_kd/seed_42/checkpoints/best.pt
```

The visual branch supports class-targeted Grad-CAM, with confident correct cases, errors, and uncertain examples selected from the frozen test predictions.

### 5.7 Run the complete ablation matrix

```bash
python ablation_studies.py \
  --config-dir configs/ablations \
  --seeds 42 1337 2026 3407 9001 \
  --continue-on-error
```

Aggregate seed-level results and export paper tables:

```bash
python scripts/aggregate_seeds.py
python scripts/export_paper_tables.py
```

## 6. TopoFM-Slice-v1 offline setup

`configs/topofm_slice_v1.yaml` uses the official DINOv2 torch.hub entrypoint. For an offline Kaggle session, attach a local DINOv2 repository and checkpoint:

```bash
python scripts/train.py \
  --config configs/topofm_slice_v1.yaml \
  --set model.dinov2.repository_path=/kaggle/input/[DINOV2_REPOSITORY_DIRECTORY] \
  --set model.dinov2.checkpoint=/kaggle/input/[DINOV2_CHECKPOINT_FILE]
```

The completed archived run was slice-level. The following remain disabled and must not be claimed as tested: patient-level 2.5D aggregation, VREx, domain-adversarial learning, soft masks, external validation, and ensembling.

## 7. Outputs saved automatically

Each run is isolated under:

```text
results/[EXPERIMENT_NAME]/seed_[SEED]/
├── resolved_config.yaml
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── epoch_[EPOCH].pt             # when periodic saving is enabled
├── logs/
│   ├── run.log
│   ├── history.csv
│   ├── environment.json
│   ├── pip_freeze.txt
│   └── tensorboard/
├── metrics/
│   ├── calibration.json
│   ├── validation_metrics.json
│   ├── test_metrics.json
│   └── test_bootstrap_ci.json
├── predictions/
│   ├── validation_predictions.csv
│   └── test_predictions.csv
├── figures/
│   ├── loss_curves.png
│   ├── validation_metrics.png
│   ├── learning_rate.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── calibration_curve.png
│   ├── probability_histogram.png
│   ├── threshold_analysis.png
│   ├── fusion_gate_distribution.png       # gated models
│   ├── router_expert_utilization.png      # routed models
│   └── curve_data/
│       ├── roc_curve.csv
│       ├── precision_recall_curve.csv
│       ├── calibration_curve.csv
│       └── threshold_analysis.csv
└── gradcam/
    └── [CASE_TYPE]_[LABEL]_[PROBABILITY]_[HASH].png
```

## 8. Quantitative evaluation

The package computes and records:

- accuracy and balanced accuracy;
- sensitivity/recall and specificity;
- precision/PPV and negative predictive value;
- F1, MCC, and Cohen’s kappa;
- false-positive and false-negative rates;
- AUROC and AUPRC;
- Brier score, negative log-likelihood, and expected calibration error;
- TN, FP, FN, and TP counts;
- stratified bootstrap confidence intervals;
- exact McNemar testing for paired classifications;
- paired bootstrap AUROC differences;
- parameter count, latency, throughput, and peak CUDA memory.

Profile a model:

```bash
python scripts/profile_model.py --config configs/topolite_kd.yaml --device cuda
```

Compare aligned frozen-test predictions:

```bash
python scripts/compare_models.py \
  --model-a results/topolite_kd/seed_42/predictions/test_predictions.csv \
  --model-b results/baseline_cnn_tda/seed_42/predictions/test_predictions.csv \
  --name-a TopoLite-KD \
  --name-b CNN-TDA
```

## 9. Scientific safeguards

- Test data are not used for early stopping, calibration, threshold selection, or hyperparameter choice.
- Exact duplicate SHA-256 conflicts across labels stop the pipeline.
- Patient-level leakage is checked when patient identifiers are available.
- TDA standardization is fitted using training data only.
- Every prediction retains its source path and SHA-256 hash.
- Teacher checkpoints are mandatory for KD.
- Cache signatures prevent incompatible topology configurations from sharing features.
- Existing non-empty run directories are protected unless resume behavior is explicit.
- Historical metrics are stored separately from generated outputs.
- A runnable reconstruction is not described as an exact historical reproduction without the archived manifest, checkpoints, resolved config, and environment.

## 10. Reproduction checklist

Before reporting a result in a paper, archive:

- [ ] immutable `split_manifest.csv`;
- [ ] dataset name/version and acquisition date;
- [ ] exact duplicate-removal report;
- [ ] patient-ID extraction rule or an explicit statement that the split is image-level;
- [ ] `resolved_config.yaml`;
- [ ] teacher and student checkpoints;
- [ ] topology cache signature and standardizer;
- [ ] package versions and hardware capture;
- [ ] validation and test prediction CSV files;
- [ ] calibration temperature and selected threshold;
- [ ] all metric JSON files and confidence intervals;
- [ ] all curves and underlying curve CSV files;
- [ ] Grad-CAM cases including errors and uncertain examples;
- [ ] multi-seed mean, standard deviation, and paired significance analysis.

## 11. Validation, tests, and documentation

```bash
python scripts/validate_install.py \
  --config configs/topolite_kd.yaml \
  --set data.image_size=32

pytest
mkdocs build --strict
```

Audit required paper artifacts, then package a completed run with checksums:

```bash
python scripts/audit_run.py --run-dir results/topolite_kd/seed_42 --require-gradcam
python scripts/package_run.py \
  --run-dir results/topolite_kd/seed_42
```

Review `CODEBASE_AUDIT.md` before making historical or novelty claims.

## 12. Historical result records

Author-supplied records are stored in:

```text
research_records/historical_results.yaml
```

They document the successful TopoLite-KD run, the negative TopoLite-MSF-KD experiment, and the slice-level TopoFM result. They are provenance records—not generated evidence.

## 13. References

1. Kundu, R., Singh, P. K., Mirjalili, S., & Sarkar, R. (2021). *COVID-19 detection from lung CT-scans using a fuzzy integral-based CNN ensemble*. Computers in Biology and Medicine, 138, 104895.
2. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*.
3. Hou, Q., Zhou, D., & Feng, J. (2021). *Coordinate Attention for Efficient Mobile Network Design*. CVPR.
4. Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR.
5. Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*.
6. Khosla, P. et al. (2020). *Supervised Contrastive Learning*. NeurIPS.
7. Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). *Topological Persistence and Simplification*.
8. GUDHI Project documentation for cubical complexes and persistent homology.

## License

Apache License 2.0. See `LICENSE`.
