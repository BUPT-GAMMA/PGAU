# PGAU: Static IR-drop prediction with Attention U-Net and label distribution smoothing.

Supported data:

- [CircuitNet](https://github.com/circuitnet/CircuitNet/tree/icisc_2023): an open-source dataset for machine learning in chip design.
- ICCAD2023: a power-grid dataset for static IR-drop prediction.

## Installation

```bash
pip install -r requirements.txt
```

## Data preprocessing

CircuitNet raw reports (`.rpt` and `.rpt.gz`):

```bash
python data_preprocess.py all \
  --input <raw_data> \
  --output processed \
  --image-size 256 \
  --label-mode sum
```

CircuitNet aligned CSV or Feather tables:

```bash
python data_preprocess.py make-maps \
  --input <aligned_tables> \
  --output processed/maps \
  --image-size 256 \
  --label-mode sum
```

ICCAD2023:

```bash
python data_preprocess.py iccad \
  --input <grid_data> \
  --output processed/iccad2023 \
  --image-size 256 \
  --normalize per_map
```

Generated CSV rows contain feature, label and optional instance-mapping paths. Data are loaded as `(batch, channels, 256, 256)` tensors.

## Feature analysis

```bash
python analysis_feature.py \
  --input <aligned_tables> \
  --output result/feature_analysis \
  --max-rows 200000 \
  --repeats 3
```

Outputs include feature rankings, validation metrics, selected features and a summary plot.

## Training and evaluation

CircuitNet:

```bash
python train.py --config configs/pgau_circuitnet.json --gpu 4 --amp
python test.py --config configs/pgau_test.json --checkpoint <checkpoint> --gpu 4 --amp
```

ICCAD2023:

```bash
python train.py --config configs/pgau_iccad2023.json --gpu 4 --amp
python test.py --config configs/pgau_iccad2023_test.json --checkpoint <checkpoint> --gpu 4 --amp
```

Select a model with `--model-type`. Training supports LDS, AMP, checkpoint resume and CPU execution.

## Six-fold cross-validation

Prepare splits only:

```bash
python cross_validate.py \
  --csv processed/iccad2023/all.csv \
  --config configs/pgau_iccad2023.json \
  --output runs/iccad2023_cv \
  --folds 6 \
  --prepare-only
```

Remove `--prepare-only` to run all folds.
