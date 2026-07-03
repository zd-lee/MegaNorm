<div align="center">

# MegaNorm

**Local Patch Embeddings for Efficient and Robust Point Normal Orientation at Super-Large Scale**

Official implementation of MegaNorm, a local-to-global pipeline for orienting
point-cloud normals at large scale.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
  <a href="https://huggingface.co/wlbbbbb/meganorm/tree/main/checkpoints">
    <img alt="Checkpoints" src="https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
  </a>
</p>

</div>

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>PatchNet</h3>
      <p>Orient normals inside local patches with iterative self-conditioning.</p>
    </td>
    <td width="33%" valign="top">
      <h3>EdgeNet</h3>
      <p>Predict whether neighboring patches should keep the same or opposite orientation.</p>
    </td>
    <td width="33%" valign="top">
      <h3>Global Solver</h3>
      <p>Optimize one binary flip variable per patch and write final oriented normals.</p>
    </td>
  </tr>
</table>

## Quick Start

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>1. Environment</h3>

<pre><code>conda create -n meganorm python=3.12 -y
conda activate meganorm

pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install -r requirements.txt</code></pre>

  </td>
  <td width="50%" valign="top">
    <h3>2. Native Patch Ops</h3>

<pre><code>python cpp_alg/setup.py build_ext --inplace</code></pre>

  </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>3. Checkpoints</h3>
      <p>Download weights from Hugging Face:</p>
      <p><a href="https://huggingface.co/wlbbbbb/meganorm/tree/main/checkpoints">huggingface.co/wlbbbbb/meganorm/checkpoints</a></p>

<pre><code>checkpoints/
  patchnet_scenenn.pth
  edgenet_scenenn.pth</code></pre>

    </td>
    <td width="50%" valign="top">
      <h3>4. Run Inference</h3>

<pre><code>python inference/infer_unified.py \
  --config configs/inference/base_config.yaml \
  --input path/to/input.ply \
  --output outputs/inference/example \
  --gpu 0</code></pre>

    </td>
  </tr>
</table>

## Pipeline

<table>
  <tr>
    <td align="center"><b>Input PLY</b><br><code>xyz + normals</code></td>
    <td align="center">-></td>
    <td align="center"><b>PatchNet</b><br>local orientation</td>
    <td align="center">-></td>
    <td align="center"><b>EdgeNet</b><br>patch consistency</td>
    <td align="center">-></td>
    <td align="center"><b>Global Flip</b><br>final normals</td>
  </tr>
</table>

## Inference Recipes

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>Single Point Cloud</h3>
      <p>Edit <code>configs/inference/base_config.yaml</code> to change checkpoints, patch extraction, optimization, and runtime settings.</p>

<pre><code>python inference/infer_unified.py \
  --config configs/inference/base_config.yaml \
  --input path/to/input.ply \
  --output outputs/inference/example \
  --gpu 0</code></pre>

    </td>
    <td width="50%" valign="top">
      <h3>ScanNet-Style Folder</h3>

<pre><code>python inference/infer_unified.py \
  --config configs/inference/scannet_v2.yaml \
  --input path/to/scans \
  --input-mode scan_root \
  --pattern '*_raw_pointcloud.ply' \
  --output outputs/inference/scannet \
  --gpu 0</code></pre>

    </td>
  </tr>
</table>

## PatchNet with `main.py`

`main.py` is the LightningCLI entry point for PatchNet. It supports
`fit`, `validate`, `test`, and `predict`, and it merges `base_configs` declared
inside configs under `pl_configs/`.

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>Train</h3>

<pre><code>python main.py fit \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml</code></pre>

    </td>
    <td width="33%" valign="top">
      <h3>Test</h3>

<pre><code>python main.py test \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml \
  --ckpt_path checkpoints/patchnet_scenenn.pth</code></pre>

    </td>
    <td width="33%" valign="top">
      <h3>Predict</h3>

<pre><code>python main.py predict \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml \
  --ckpt_path checkpoints/patchnet_scenenn.pth</code></pre>

    </td>
  </tr>
</table>

<details>
<summary><b>Common command-line overrides</b></summary>

```bash
python main.py fit \
  --config pl_configs/SceneNN_part_iterative_multiscale_bs8_mixup_aug.yaml \
  --trainer.devices 1 \
  --data.init_args.root data/SceneNN_part \
  --data.init_args.batch_size 4
```

Logs are written to `pl_logs/`. Checkpoint behavior is controlled by the
`trainer.callbacks` section in the active config.

</details>

## Training Data

<table>
  <tr>
    <td width="45%" valign="top">
      <h3>Expected Layout</h3>

<pre><code>data/SceneNN_part/
  train/*.ply
  val/*.ply
  test/*.ply</code></pre>

    </td>
    <td width="55%" valign="top">
      <h3>File Requirements</h3>
      <p>Training PLY files should contain point coordinates and ground-truth normals.</p>
    </td>
  </tr>
</table>

## EdgeNet Training

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>1. Precompute Features</h3>

<pre><code>python dataset/precompute_patch_features_multiscale.py \
  --config configs/global_flip/feat_extra/scenenn_train.yaml \
  --gpu 0 \
  --split train,val,test</code></pre>

    </td>
    <td width="50%" valign="top">
      <h3>2. Train EdgeNet</h3>

<pre><code>python train_edge_consistency.py \
  --config configs/global_flip/edge_consistency/scenenn_train.yaml \
  --gpu 0</code></pre>

    </td>
  </tr>
</table>

## Solver Notes

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>Optimization Objective</h3>
      <p>The global patch flip step is a binary XOR optimization over EdgeNet same/opposite edge scores. Higher optimized energy is better for the predicted objective.</p>
    </td>
    <td width="50%" valign="top">
      <h3>Solvers</h3>
      <p>The paper experiments use Gurobi. For easier reproduction, we also tested OR-Tools CP-SAT and SCIP/PySCIPOpt on exported patch-level instances.</p>
    </td>
  </tr>
</table>

`Mean Delta GT` is the mean optimized edge energy minus the energy obtained by
the ground-truth flip labels under the same predicted EdgeNet edge objective.
This value can be positive because the EdgeNet-predicted weights are not always
perfectly aligned with the ground-truth flip labels. `Max Diff vs Gurobi` is the
maximum absolute energy difference from Gurobi on matched instances.

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

```bibtex
@inproceedings{li2026meganorm,
  title={MegaNorm: Local Patch Embeddings for Efficient and Robust Point Normal Orientation at Super-Large Scale},
  author={Li, Zhuodong and Liu, Zengke and Hou, Fei and Chen, Xuhui and Wang, Wencheng and He, Ying},
  booktitle={SIGGRAPH Conference Papers},
  year={2026},
  note={11 pages}
}
```
