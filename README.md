# MegaNorm

This repository is the official implementation of **MegaNorm: Local Patch
Embeddings for Efficient and Robust Point Normal Orientation at Super-Large
Scale**.

MegaNorm orients point-cloud normals with a local-to-global pipeline. It first orients normals inside independent local patches with a self-conditioning Point Transformer V3 model, then predicts pairwise patch consistency with EdgeNet and solves patch flips globally.

## Algorithm

1. Split the point cloud into patches with FPS and connected-component splitting.
2. Estimate initial unoriented normals with PCA.
3. Run PatchNet for three self-conditioning iterations on `[xyz, normal, confidence]` features.
4. Pool PatchNet patch features and build a kNN graph over patch centers.
5. Run EdgeNet on neighboring patch pairs to predict same/opposite orientation scores.
6. Optimize one binary flip variable per patch and write the final oriented normals.

## Installation

```bash
conda create -n meganorm python=3.12 -y
conda activate meganorm
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install spconv-cu124 open3d plyfile timm tensorboard pandas scikit-learn scipy pyyaml tqdm addict
```

Build the C++ patch utilities:

```bash
python cpp_alg/setup.py build_ext --inplace
```

## Data Layout
Use PLY files with point coordinates and ground-truth normals for supervised training:

```text
data/SceneNN_part/
  train/*.ply
  val/*.ply
  test/*.ply
```
## Train PatchNet

```bash
python train_direct_orientation_i.py \
  --config configs/direct_orientation/SceneNN_part_iterative_multiscale_bs16_mixup.yaml \
  --gpu 0
```

Evaluate a checkpoint:

```bash
python train_direct_orientation_i.py \
  --config configs/direct_orientation/SceneNN_part_iterative_multiscale_bs16_mixup.yaml \
  --resume checkpoints/patchnet_scenenn.pth \
  --gpu 0 \
  --test
```

## Train EdgeNet

First precompute patch features with a trained PatchNet checkpoint configured in `configs/global_flip/feat_extra/SceneNN_fps_overlap2_big_i.yaml`:

```bash
python dataset/precompute_patch_features_multiscale.py \
  --config configs/global_flip/feat_extra/SceneNN_fps_overlap2_big_i.yaml \
  --gpu 0 \
  --split train,val,test
```

Then train EdgeNet:

```bash
python train_edge_consistency.py \
  --config configs/global_flip/edge_consistency/SceneNN.yaml \
  --gpu 0
```

## Inference

Set checkpoint paths in `configs/inference/base_config.yaml`, then run:

```bash
python inference/infer_unified.py \
  --config configs/inference/base_config.yaml \
  --input path/to/input.ply \
  --output outputs/inference/example \
  --gpu 0
```

Batch ScanNet-style root:

```bash
python inference/infer_unified.py \
  --config configs/inference/base_config.yaml \
  --input path/to/scans \
  --input-mode scan_root \
  --pattern '*_raw_pointcloud.ply' \
  --output outputs/inference/scannet \
  --gpu 0
```

## Optimizer Notes

The global patch flip step is a binary XOR optimization over EdgeNet same/opposite
edge scores. The experiments in the paper use Gurobi for this step. For easier
reproduction without a commercial solver, we also tested two open-source Python
solver interfaces on exported patch-level instances: OR-Tools CP-SAT and
SCIP/PySCIPOpt. On these test cases, both open-source solvers reached the same
optimized edge energies as Gurobi, but they were slower on larger instances.

`Mean Delta GT` is the mean optimized edge energy minus the energy obtained by
the ground-truth flip labels under the same predicted EdgeNet edge objective.
This value can be positive because the EdgeNet-predicted weights are not always
perfectly aligned with the ground-truth flip labels. Higher optimized energy is
better for the predicted objective. `Max Diff vs Gurobi` is the maximum absolute
energy difference from Gurobi on matched instances.

| Dataset | Cases | Solver | Status | Mean Energy | Mean Delta GT | Mean Solve (s) | Max Diff vs Gurobi |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| SceneNN scale=2 | 15 | Gurobi socket | OK | 154.212816 | 13.319571 | 1.288 | 0 |
| SceneNN scale=2 | 15 | OR-Tools CP-SAT | OPTIMAL | 154.212816 | 13.319571 | 1.144 | 0 |
| SceneNN scale=2 | 15 | SCIP/PySCIPOpt | optimal | 154.212816 | 13.319571 | 2.854 | 0 |
| ScanNetV2 scale=0 | 30 | Gurobi socket | OK | 783.736960 | 3.411295 | 1.492 | 0 |
| ScanNetV2 scale=0 | 30 | OR-Tools CP-SAT | OPTIMAL | 783.736960 | 3.411295 | 0.369 | 0 |
| ScanNetV2 scale=0 | 30 | SCIP/PySCIPOpt | optimal | 783.736960 | 3.411295 | 1.429 | 0 |
| T2 scale=2 | 2 | Gurobi socket | OK | 6090.806073 | 58.182655 | 2.609 | 0 |
| T2 scale=2 | 2 | OR-Tools CP-SAT | FEASIBLE/OPTIMAL | 6090.806073 | 58.182655 | 84.029 | 0 |
| T2 scale=2 | 2 | SCIP/PySCIPOpt | optimal | 6090.806073 | 58.182655 | 340.560 | 0 |

## Citation

If you use MegaNorm in your research, please cite:

```bibtex
@inproceedings{li2026meganorm,
  title={MegaNorm: Local Patch Embeddings for Efficient and Robust Point Normal Orientation at Super-Large Scale},
  author={Li, Zhuodong and Liu, Zengke and Hou, Fei and Chen, Xuhui and Wang, Wencheng and He, Ying},
  booktitle={SIGGRAPH Conference Papers},
  year={2026},
  note={11 pages}
}
```
