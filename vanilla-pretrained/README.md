# Vanilla Pretrained

This is the supervised reference pipeline for BAC classification. It fine-tunes ImageNet-pretrained CNN and transformer architectures on labeled mammograms and provides the baseline used to compare pseudolabeling, distillation and self-supervised pretraining.

## Before running

The mammography images and annotations are private and are not included in this repository. Prepare CSV files with at least these columns:

```text
image_path,label,fold
relative/or/absolute/image.png,0,0
```

`image_path`, `label` and `fold` are read by `src/data/dataset.py`.
`patient_id` is additionally required by `src/utils/aggregate.py` when
computing patient-level metrics. Keep all views from a patient in the
same split or fold. Update the paths in `src/config/cfg_train.yaml` and
`src/config/cfg_test.yaml`.

Use patient-level train/validation/test splits. Do not place patient data, checkpoints or credentials in the repository. See the root [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the project-wide data and output policy.

The thesis used stratified 3-fold cross-validation and a final 70/10/20
patient-level holdout. The checked-in training configuration defaults to
`num_folds: 1`; set the fold values in the CSVs and configuration
explicitly when reproducing cross-validation.

## Preprocessing contract

The thesis preprocessing has two stages: DICOM/window-level conversion and
Felzenszwalb-based breast masking/cropping followed by resizing to
`1120 x 576`. The active runtime transform currently applies RGB
conversion and tensor conversion; the masking/cropping transform is not
enabled by default. Training should therefore receive the already
preprocessed PNGs, unless the user explicitly enables and validates the
offline preprocessing path in `src/data/pre_processing.py`.

## Environment

Create a Python environment with a PyTorch build appropriate for the available hardware and install the dependencies required by the selected models. Run commands from this directory so the relative configuration paths resolve:

```powershell
cd vanilla-pretrained
$env:WANDB_MODE = "offline"
```

Set `WANDB_MODE=online` only when external logging is intentional, and provide `WANDB_API_KEY` through the environment rather than source files.

## Train and test

```powershell
python train_caller.py
python test_caller.py
```

The callers load the YAML configuration, construct the model and dataloaders, and delegate to `src/utils/train.py` and `src/utils/test.py`.

## Code map

- `src/data/`: CSV dataset loading, mammogram preprocessing and transforms.
- `src/models/models.py`: model construction and optional layer freezing.
- `src/utils/loss.py`: binary and multiclass loss functions, including weighted losses.
- `src/utils/eval.py`: ROC/PR metrics and threshold selection.
- `src/utils/interpret.py`: Grad-CAM-based interpretation utilities.
- `src/utils/aggregate.py`: image metrics and patient-level aggregation.
  An exam-level aggregation helper is not currently committed.

## Outputs

Runs are written to `results/<model_name>/<timestamp>/`. Training saves
weights and training logs; testing saves `predictions.csv` and logs AUC
metrics to W&B. Grad-CAM generation is currently not invoked by default,
and the checked-in test configuration contains an example student
checkpoint path plus a ResNet-style interpretation layer. Replace both
before testing a different model. Retain the configuration, dependency
versions, seed and checkpoint provenance with every reported result.

The pipeline has no dedicated lockfile. A compatible environment normally
needs PyTorch, torchvision, timm, OmegaConf, pandas, scikit-learn,
scikit-image, OpenCV, W&B and `pytorch-grad-cam`; pin versions appropriate
to the selected CUDA/PyTorch build before running an experiment.
