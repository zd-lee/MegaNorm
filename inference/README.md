# DACPO: Directed Point Cloud Orientation

DACPO is an algorithm for unifying normal orientations in oriented point clouds using a trained PTv3 MaskDecoder model.

## Overview

Given an oriented point cloud (points with normals), DACPO automatically determines which normals should be flipped to achieve consistent orientation across the entire point cloud.

### Algorithm Steps

1. **Query Sampling**: Sample N_q query points uniformly from the point cloud
2. **KNN Graph Construction**: Build a k-nearest neighbor graph (k=10)
3. **BFS Patch Extraction**: Extract patches of points around each query using BFS
4. **Model Inference**: Use trained PTv3 MaskDecoder to predict flip labels for each patch
5. **Consistency Graph**: Build a graph measuring flip label consistency between overlapping patches
6. **Optimization**: Solve 0-1 optimization to find globally consistent patch orientations
7. **Point Label Aggregation**: Aggregate patch-level decisions to point-level flip labels
8. **Normal Flipping**: Apply flips using confidence-weighted voting

## Installation

The inference module requires the following dependencies:

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.20.0
scipy>=1.7.0
open3d>=0.16.0
torch-cluster>=1.6.0
pyyaml

# Already installed in this environment
```

## Configuration

Edit `dacpo_config.yaml` to configure the algorithm:

### Key Parameters

**Model Configuration:**
- `checkpoint_path`: Path to trained PTv3 MaskDecoder checkpoint (required)
- `config_path`: Path to model configuration YAML
- `device`: 'cuda' or 'cpu'
- `mask_threshold`: Sigmoid threshold for binary flip prediction (default: 0.5)

**DACPO Algorithm:**
- `num_query_points`: Number of query points to sample (default: 100)
- `sampling_method`: 'fps' (farthest point sampling) or 'random'
- `knn_k`: Number of nearest neighbors for KNN graph (default: 10)
- `num_per_patch`: Number of points per patch (default: 256)
- `min_overlap_points`: Minimum overlap to consider patches as neighbors (default: 10)
- `consistency_threshold`: Minimum consistency ratio for graph edges (default: 0.6)
- `optimization_method`: 'greedy', 'spectral', or 'connected_components' (default: 'greedy')
- `voting_method`: 'confidence_weighted' or 'simple_majority' (default: 'confidence_weighted')

### Python API

```python
import numpy as np
from inference.dacpo_pipeline import DACPOPipeline
from inference.utils import load_point_cloud, save_point_cloud
import yaml

# Load configuration
with open('inference/dacpo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set checkpoint path
config['model']['checkpoint_path'] = 'path/to/checkpoint.pth'

# Load point cloud
points, normals = load_point_cloud('input.ply')

# Initialize pipeline
pipeline = DACPOPipeline(config)

# Run DACPO
oriented_normals, stats = pipeline.run(
    points,
    normals,
    save_intermediate=True,
    intermediate_dir='results/intermediate'
)

# Save result
save_point_cloud('output.ply', points, oriented_normals)

# Print statistics
print(f"Flipped {stats['num_flipped']}/{stats['num_points']} points")
print(f"Total time: {stats['total_time']:.2f}s")
```

## File Structure

```
inference/
├── __init__.py                # Module initialization
├── dacpo_config.yaml          # Configuration file
├── dacpo_pipeline.py          # Main DACPO pipeline
├── patch_extractor.py         # KNN graph + BFS patch extraction
├── model_wrapper.py           # PTv3 MaskDecoder wrapper
├── patch_graph.py             # Consistency graph builder
├── optimization.py            # 0-1 optimization solver
├── utils.py                   # I/O and visualization utilities
├── inference.py               # CLI interface
└── README.md                  # This file
```

## Model Requirements

DACPO requires a trained PTv3 MaskDecoder model. The model should be trained on oriented point cloud data where:

- **Input**: Point clouds with coordinates and normals (N × 6)
- **Query**: Query point with position and features (6D)
- **Output**: Binary mask indicating which points should have flipped normals

Training script: `train_ptv3_maskdecoder.py`

## Input/Output Formats

**Supported file formats:**
- `.ply` - PLY format (ASCII or binary)
- `.pcd` - Point Cloud Data format
- `.xyz` - XYZ ASCII format
- `.npy` - NumPy array (N × 6: coordinates + normals)

**Point cloud requirements:**
- Must contain 3D coordinates
- Must contain normal vectors (or use `estimate_normals=True`)
- Normals will be normalized automatically

## Performance

**Typical performance on a point cloud with 10,000-30,000 points:**
- Query sampling: < 0.1s
- Patch extraction: 0.5-1s
- Model inference: 2-5s (GPU), 10-30s (CPU)
- Consistency graph: 0.5-1s
- Optimization: 0.1-0.5s
- Total: 3-8s (GPU), 12-33s (CPU)

**Memory requirements:**
- GPU: ~2-4 GB VRAM
- RAM: ~2-4 GB

## Optimization Methods

### MIQP (Default, Recommended)
Uses a remote Gurobi server to solve the Mixed-Integer Quadratic Programming problem optimally. Provides the best solution quality but requires a running MIQP server.

**Configuration:**
```yaml
dacpo:
  optimization_method: miqp
  miqp_server_host: "192.168.8.19"  # Server IP
  miqp_server_port: 11111           # Server port
  miqp_timeout: 300.0               # Timeout in seconds
```

**Fallback:** If the MIQP server is unavailable, automatically falls back to the greedy solver.

**Starting the MIQP Server:**
The server code is provided in `inference/MIQP.py` (currently commented out). Uncomment and run on a machine with Gurobi installed:
```bash
python inference/MIQP.py
```

### Greedy
Fast heuristic that propagates labels from high-consistency patches. Works well in practice and doesn't require external dependencies.

### Spectral
Uses spectral clustering on the consistency graph. More principled but can be slower.

### Connected Components
Treats each connected component independently. Good for disconnected point clouds.

## Troubleshooting

### Model fails to load
- Check that `checkpoint_path` in config points to a valid .pth file
- Verify that the model was trained with the same architecture (check config)

### Out of memory (GPU)
- Reduce `num_query_points` (e.g., 50 instead of 100)
- Reduce `num_per_patch` (e.g., 128 instead of 256)
- Reduce `inference_batch_size` (e.g., 8 instead of 32)
- Use `device: cpu` in config

### Poor results
- Increase `num_query_points` for better coverage
- Adjust `consistency_threshold` (lower = more edges, higher = fewer edges)
- Try different `optimization_method` (greedy vs spectral)
- Check if the model was trained on similar data

### Point cloud has no normals
- Use `estimate_normals: true` in config's `io` section
- Or pre-process point cloud to add normals using Open3D

### MIQP server connection fails
- Check that the MIQP server is running: `telnet 192.168.8.19 11111`
- Verify the server IP and port in config match the running server
- Check firewall settings allow connections to port 11111
- If server is unavailable, the algorithm will automatically fall back to greedy solver
- To use greedy solver directly, set `optimization_method: greedy` in config

## Examples

### Unified Inference CLI

`infer_unified.py` now supports three common input shapes:

- Single `.ply` file
- Single ScanNet scene directory such as `scans/scene0000_00`
- A ScanNet scan root such as `scans/`, which will search child scene directories for `*_raw_pointcloud.ply`

Single ScanNet scene:
```bash
python inference/infer_unified.py \
  --config configs/inference/optimize/optimized_scene_on_scannetv2.yaml \
  --input data/lzk_scan_raw/scans/scene0000_00
```

Batch ScanNet root:
```bash
python inference/infer_unified.py \
  --config configs/inference/optimize/optimized_scene_on_scannetv2.yaml \
  --input data/lzk_scan_raw/scans
```

Dry-run discovery without loading models:
```bash
python inference/infer_unified.py \
  --config configs/inference/optimize/optimized_scene_on_scannetv2.yaml \
  --input data/lzk_scan_raw/scans \
  --dry-run
```

Useful discovery flags:

- `--input-mode auto|file|scene_dir|scan_root`
- `--pattern '*_raw_pointcloud.ply'`
- `--recursive` to force directory inputs to be treated as scan roots

Batch runs write a summary CSV under `outputs/inference/summary/`, including `input_path`, `scene_name`, and per-file status.

Test the optimization module (requires MIQP server running at 192.168.8.19:11111):
```bash
cd inference
python optimization.py
```

This will test all optimization methods including MIQP and verify:
- All solvers work correctly
- MIQP server connection and communication
- Matrix format correctness (A + B = 1 for non-diagonal elements)

Test other modules individually:
```bash
python dataset/patch_extractor.py  # Test patch extraction
python inference/patch_graph.py      # Test consistency graph
python inference/utils.py            # Test I/O functions
```

## Citation

If you use DACPO in your research, please cite:

```bibtex
@article{dacpo2024,
  title={DACPO: Directed Point Cloud Orientation using Deep Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

[Add your license information here]

## Contact

For questions or issues, please contact [your contact information].
