# MegaNorm

Official implementation of **MegaNorm: Local Patch Embeddings for Efficient and
Robust Point Normal Orientation at Super-Large Scale**.

MegaNorm orients point-cloud normals with a local-to-global pipeline: PatchNet
orients normals inside local patches, EdgeNet predicts pairwise patch
consistency, and a global flip optimization produces the final oriented normals.

## Installation

```bash
conda create -n meganorm python=3.12 -y
conda activate meganorm
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install -r requirements.txt
```

Build the C++ patch utilities:

```bash
python cpp_alg/setup.py build_ext --inplace
```

## Pretrained Weights

Download pretrained weights from Hugging Face:

```text
https://huggingface.co/wlbbbbb/meganorm/tree/main/checkpoints
```

Put them under `checkpoints/`. The default inference config expects:

```text
checkpoints/patchnet_scenenn.pth
checkpoints/edgenet_scenenn.pth
```

## Inference

For the full MegaNorm pipeline, edit checkpoint paths and options in
`configs/inference/base_config.yaml`, then run:

```bash
python inference/infer_unified.py \
  --config configs/inference/base_config.yaml \
  --input path/to/input.ply \
  --output outputs/inference/example \
  --gpu 0
```

For a ScanNet-style directory:

```bash
python inference/infer_unified.py \
  --config configs/inference/scannet_v2.yaml \
  --input path/to/scans \
  --input-mode scan_root \
  --pattern '*_raw_pointcloud.ply' \
  --output outputs/inference/scannet \
  --gpu 0
```

## Data Layout

Training expects PLY files with point coordinates and ground-truth normals:

```text
data/SceneNN_part/
  train/*.ply
  val/*.ply
  test/*.ply
```

## `main.py` Usage

`main.py` is the LightningCLI entry point for PatchNet training, evaluation, and
prediction. It loads configs from `pl_configs/` and supports `base_configs`
merging, so `pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml`
inherits the trainer, model, optimizer, and scheduler defaults from
`pl_configs/base/`.

Train PatchNet:

```bash
python main.py fit \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml
```

Test a checkpoint:

```bash
python main.py test \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml \
  --ckpt_path checkpoints/patchnet_scenenn.pth
```

Run PatchNet prediction on the config's test split:

```bash
python main.py predict \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml \
  --ckpt_path checkpoints/patchnet_scenenn.pth
```

Common overrides can be passed directly on the command line:

```bash
python main.py fit \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml \
  --trainer.devices 1 \
  --data.init_args.root data/SceneNN_part \
  --data.init_args.batch_size 4
```

Logs are written to `pl_logs/`; checkpoints are written according to the
`trainer.callbacks` section in the active config.

## Train EdgeNet

First precompute patch features with a trained PatchNet checkpoint configured in
`configs/global_flip/feat_extra/scenenn_train.yaml`:

```bash
python dataset/precompute_patch_features_multiscale.py \
  --config configs/global_flip/feat_extra/scenenn_train.yaml \
  --gpu 0 \
  --split train,val,test
```

Then train EdgeNet:

```bash
python train_edge_consistency.py \
  --config configs/global_flip/edge_consistency/scenenn_train.yaml \
  --gpu 0
```

## Citation

```bibtex
@inproceedings{li2026meganorm,
  title={MegaNorm: Local Patch Embeddings for Efficient and Robust Point Normal Orientation at Super-Large Scale},
  author={Li, Zhuodong and Liu, Zengke and Hou, Fei and Chen, Xuhui and Wang, Wencheng and He, Ying},
  booktitle={SIGGRAPH Conference Papers},
  year={2026},
  note={11 pages}
}
```
