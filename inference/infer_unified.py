"""Unified inference: M1+Optimization or M1+M2 pipeline."""
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from models.direct_orientation_model import create_direct_orientation_model
from models.global_flip_transformer import create_global_flip_transformer
from dataset.patch_extractor import extract_patches_unified, get_overlap_fast
from dataset.dataset import estimate_normals_torch
from dataset.transforms import NormalEstimationNormalize
from utils.pointcloud_io import write_ply_ascii
from utils.downsample import random_downsample
from inference.optimization import FlipOptimizer, compute_objective
# from inference.orientation_solver import VoteFlipSolver
from inference.vis import visualize_patch_graphs
from utils.cache_utils import (
    get_inference_cache_dir,
    load_inference_cache,
    save_inference_cache,
    save_inference_cache_metadata,
)


def _sort_patches_by_size(patches, *arrays):
    """Sort patches descending by size, applying the same order to any parallel arrays."""
    order = sorted(range(len(patches)), key=lambda i: len(patches[i]), reverse=True)
    sorted_patches = [patches[i] for i in order]
    sorted_arrays = tuple([arr[i] for i in order] for arr in arrays)
    return (sorted_patches,) + sorted_arrays


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def detect_pipeline_type(config):
    method = config.get('global_flip_method', 'miqp')
    if method == 'm2':
        return 'two_stage'
    elif method in ['vote', 'miqp']:
        return 'optimize'
    else:
        raise ValueError(f"Unknown global_flip_method: {method}")


def generate_output_path(config_path, input_path, output_root='outputs/inference'):
    """Generate output directory path for inference results.

    Args:
        config_path: Path to configuration file
        input_path: Path to input data
        output_root: Root directory for all inference outputs (default: 'outputs/inference')

    Returns:
        Output path: outputs/inference/output/{config_name}/
    """
    config_name = Path(config_path).stem
    return Path(output_root) / "output" / config_name


def infer_scene_name_from_ply(ply_path):
    parent_name = ply_path.parent.name
    return parent_name if ply_path.stem.startswith(parent_name) else None


def find_scene_ply(scene_dir, pattern):
    matches = sorted(p for p in scene_dir.glob(pattern) if p.is_file())
    if not matches:
        return None, f"no files matching '{pattern}'"

    preferred_name = f"{scene_dir.name}_raw_pointcloud.ply"
    preferred = [p for p in matches if p.name == preferred_name]
    if len(preferred) == 1:
        return preferred[0], None
    if len(preferred) > 1:
        return None, f"multiple files named '{preferred_name}'"

    prefix_matches = [p for p in matches if p.stem.startswith(scene_dir.name)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], None
    if len(prefix_matches) > 1:
        names = ', '.join(p.name for p in prefix_matches[:5])
        return None, f"multiple scene-matched files: {names}"

    if len(matches) == 1:
        return matches[0], None

    names = ', '.join(p.name for p in matches[:5])
    return None, f"multiple files matching '{pattern}': {names}"


def build_inference_target(ply_path, source_mode, scene_name=None):
    scene_name = scene_name or infer_scene_name_from_ply(ply_path)
    output_name = scene_name or ply_path.stem
    return {
        'ply_path': ply_path,
        'source_mode': source_mode,
        'scene_name': scene_name,
        'output_name': output_name,
    }


def resolve_input_targets(input_path, input_mode='auto', pattern='*_raw_pointcloud.ply', recursive=False):
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_mode == 'file':
        resolved_mode = 'file'
    elif input_mode == 'scene_dir':
        resolved_mode = 'scene_dir'
    elif input_mode == 'scan_root':
        resolved_mode = 'scan_root'
    elif input_mode == 'auto':
        if input_path.is_file():
            resolved_mode = 'file'
        else:
            scene_ply, _ = find_scene_ply(input_path, pattern)
            has_subdirs = any(p.is_dir() for p in input_path.iterdir())
            if scene_ply is not None:
                resolved_mode = 'scene_dir'
            elif recursive or has_subdirs:
                resolved_mode = 'scan_root'
            else:
                resolved_mode = 'flat_dir'
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")

    targets = []
    skipped = []

    if resolved_mode == 'file':
        if not input_path.is_file():
            raise ValueError(f"Expected a file for input_mode=file, got: {input_path}")
        if input_path.suffix.lower() != '.ply':
            raise ValueError(f"Only .ply input files are supported, got: {input_path}")
        targets.append(build_inference_target(input_path, source_mode='file'))

    elif resolved_mode == 'scene_dir':
        if not input_path.is_dir():
            raise ValueError(f"Expected a directory for input_mode=scene_dir, got: {input_path}")
        scene_ply, reason = find_scene_ply(input_path, pattern)
        if scene_ply is None:
            raise FileNotFoundError(f"Could not resolve scene point cloud in {input_path}: {reason}")
        targets.append(build_inference_target(scene_ply, source_mode='scene_dir', scene_name=input_path.name))

    elif resolved_mode == 'scan_root':
        if not input_path.is_dir():
            raise ValueError(f"Expected a directory for input_mode=scan_root, got: {input_path}")

        child_dirs = sorted(p for p in input_path.iterdir() if p.is_dir())
        if child_dirs:
            for child_dir in child_dirs:
                scene_ply, reason = find_scene_ply(child_dir, pattern)
                if scene_ply is None:
                    skipped.append({
                        'model_name': child_dir.name,
                        'scene_name': child_dir.name,
                        'input_path': str(child_dir),
                        'status': 'skipped',
                        'error': reason,
                    })
                    continue
                targets.append(build_inference_target(scene_ply, source_mode='scan_root', scene_name=child_dir.name))
        else:
            matches = sorted(p for p in input_path.rglob(pattern) if p.is_file())
            for ply_path in matches:
                targets.append(build_inference_target(ply_path, source_mode='scan_root'))

    else:
        if not input_path.is_dir():
            raise ValueError(f"Expected a directory for flat input, got: {input_path}")
        matches = sorted(p for p in input_path.glob(pattern) if p.is_file())
        effective_pattern = pattern
        if not matches and pattern != '*.ply':
            matches = sorted(p for p in input_path.glob('*.ply') if p.is_file())
            effective_pattern = '*.ply'

        for ply_path in matches:
            targets.append(build_inference_target(ply_path, source_mode='flat_dir'))

        if not targets:
            skipped.append({
                'model_name': input_path.name,
                'scene_name': None,
                'input_path': str(input_path),
                'status': 'skipped',
                'error': f"no files matching '{effective_pattern}'",
            })

    return targets, skipped, resolved_mode


def load_m1_model(config, device):
    print("Loading M1...")
    m1_config = load_config(config['models']['m1']['config'])
    m1 = create_direct_orientation_model(m1_config)
    ckpt = torch.load(config['models']['m1']['checkpoint'], map_location='cpu')
    m1.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    return m1.to(device).eval()


def load_edgenet_model(config, device):
    """Load EdgeNet model for edge consistency prediction.

    Returns:
        EdgeNet model or None if disabled
    """
    edge_config = config.get('edge_consistency', {})
    if not edge_config.get('enabled', False):
        return None

    print("Loading EdgeNet...")
    from models.edge_consistency_mlp import create_edge_consistency_mlp
    import os

    edge_ckpt_path = edge_config['checkpoint']
    edge_cfg_path = edge_config['config']

    # Validate paths
    if not os.path.exists(edge_ckpt_path):
        raise FileNotFoundError(f"EdgeNet checkpoint not found: {edge_ckpt_path}")
    if not os.path.exists(edge_cfg_path):
        raise FileNotFoundError(f"EdgeNet config not found: {edge_cfg_path}")

    edge_cfg = load_config(edge_cfg_path)
    edgenet = create_edge_consistency_mlp(edge_cfg)
    ckpt = torch.load(edge_ckpt_path, map_location='cpu')
    edgenet.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))

    return edgenet.to(device).eval()


def load_models(config, device):
    m1 = load_m1_model(config, device)
    print("Loading M2...")
    m2_config = load_config(config['models']['m2']['config'])
    m2 = create_global_flip_transformer(m2_config)
    ckpt = torch.load(config['models']['m2']['checkpoint'], map_location='cpu')
    m2.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    return m1, m2.to(device).eval()


def run_m1_iterative_inference(patches, coords, m1, config, device, extract_m2_features=False):
    transform = NormalEstimationNormalize()
    grid_size = config['inference']['grid_size']
    pca_max_nn = config['inference']['pca_max_nn']
    batch_size = config['inference'].get('batch_size', 1)

    iterative_config = config.get('iterative', {})
    num_iterations = iterative_config.get('num_iterations', 1)
    use_confidence = iterative_config.get('use_confidence', False)

    if num_iterations > 1:
        print(f"Iterative: {num_iterations} iterations, confidence={use_confidence}")
    print(f"Batch size: {batch_size}")

    all_patch_normals = []
    all_patch_features = [] if extract_m2_features else None
    all_patch_centers = [] if extract_m2_features else None
    num_batches = (len(patches) + batch_size - 1) // batch_size
    pca_time = 0.0

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="M1"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(patches))
            batch_patches = patches[batch_start:batch_end]

            patch_data_list = []
            for patch_indices in batch_patches:
                patch_coords = torch.from_numpy(coords[patch_indices.cpu().numpy()]).float()
                _t = time.time()
                pca_normals = torch.from_numpy(estimate_normals_torch(patch_coords.numpy(), max_nn=pca_max_nn)[:, 3:6]).float()
                pca_time += time.time() - _t
                conf = torch.zeros(len(patch_coords), 1) if use_confidence else None
                patch_data_list.append([patch_coords, pca_normals, conf])

            for _ in range(num_iterations):
                transformed_data = []
                for coords_i, normals_i, conf_i in patch_data_list:
                    feat = torch.cat([coords_i, normals_i, conf_i], dim=1) if use_confidence else torch.cat([coords_i, normals_i], dim=1)
                    pd = {'coord': coords_i.to(device), 'feat': feat.to(device),
                          'offset': torch.tensor([len(coords_i)]).long().to(device), 'grid_size': grid_size}
                    pd, _ = transform(pd)
                    transformed_data.append((pd['coord'], pd['feat'], len(coords_i)))

                batch_coords = torch.cat([d[0] for d in transformed_data], dim=0)
                batch_feats = torch.cat([d[1] for d in transformed_data], dim=0)
                batch_offsets = torch.cumsum(torch.tensor([d[2] for d in transformed_data]), dim=0).long().to(device)
                grid_size = torch.tensor([grid_size]).float().to(device)
                logits = m1({'coord': batch_coords, 'feat': batch_feats, 'offset': batch_offsets, 'grid_size': grid_size})[:, 0]

                flip_prob = torch.sigmoid(logits.cpu())
                start_idx = 0
                for i, (coords_i, normals_i, conf_i) in enumerate(patch_data_list):
                    end_idx = start_idx + len(coords_i)
                    flip_mask = flip_prob[start_idx:end_idx] > 0.5
                    normals_i[flip_mask] = -normals_i[flip_mask]
                    if use_confidence:
                        patch_data_list[i][2] = (torch.abs(flip_prob[start_idx:end_idx] - 0.5) * 2).unsqueeze(1)
                    start_idx = end_idx

            all_patch_normals.extend([normals for _, normals, _ in patch_data_list])

            if extract_m2_features:
                m2_enabled = config.get('m2_iterative', {}).get('enabled', False)
                for coords_i, normals_i, conf_i in patch_data_list:
                    feat = torch.cat([coords_i, normals_i, conf_i], dim=1) if use_confidence else torch.cat([coords_i, normals_i], dim=1)
                    pd = {'coord': coords_i.to(device), 'feat': feat.to(device),
                          'offset': torch.tensor([len(coords_i)]).long().to(device), 'grid_size': grid_size}
                    pd, _ = transform(pd)
                    backbone_output = m1.backbone(pd, False)
                    patch_feature = backbone_output['feat'].max(dim=0)[0].cpu()
                    patch_center = coords_i.mean(dim=0)

                    if m2_enabled:
                        inv_feat = feat.clone()
                        inv_feat[:, 3:6] *= -1
                        pd_inv = {'coord': coords_i.to(device), 'feat': inv_feat.to(device),
                                  'offset': torch.tensor([len(coords_i)]).long().to(device), 'grid_size': grid_size}
                        pd_inv, _ = transform(pd_inv)
                        patch_feature_inv = m1.backbone(pd_inv, False)['feat'].max(dim=0)[0].cpu()
                        all_patch_features.append(torch.stack([patch_feature, patch_feature_inv]))
                    else:
                        all_patch_features.append(patch_feature)
                    all_patch_centers.append(patch_center)

    if extract_m2_features:
        return all_patch_normals, all_patch_features, all_patch_centers, pca_time
    else:
        return all_patch_normals, pca_time


def build_knn_graph_for_edgenet(patch_centers, k=10):
    """Build KNN graph from patch centers.

    Args:
        patch_centers: List of (3,) tensors or (P, 3) tensor
        k: Number of neighbors

    Returns:
        edges: (N_edges, 2) numpy array of [source, target] indices
    """
    from dataset.edge_consistency_dataset import build_knn_graph

    if isinstance(patch_centers, list):
        centers_np = np.stack([c.cpu().numpy() for c in patch_centers])
    else:
        centers_np = patch_centers.cpu().numpy()

    # Adaptive k for small patch counts
    k_actual = min(k, len(centers_np) - 1)
    if k_actual < k:
        print(f"Warning: Only {k_actual} neighbors available (requested {k})")

    return build_knn_graph(centers_np, k=k_actual)


def run_edgenet_inference(edgenet, patch_features, patch_centers, edges, device):
    """Run EdgeNet to predict edge consistency.

    Args:
        edgenet: EdgeConsistencyMLP model
        patch_features: List of (512,) tensors
        patch_centers: List of (3,) tensors
        edges: (N_edges, 2) numpy array
        device: torch device

    Returns:
        edge_logits: (N_edges,) numpy array
    """
    features = torch.stack(patch_features).to(device)
    centers = torch.stack(patch_centers).to(device)

    # Extract edge features
    features_A = features[edges[:, 0]]  # (N_edges, 512)
    features_B = features[edges[:, 1]]  # (N_edges, 512)
    centers_A = centers[edges[:, 0]]    # (N_edges, 3)
    centers_B = centers[edges[:, 1]]    # (N_edges, 3)

    # Feed all edges at once (no batching)
    with torch.no_grad():
        edge_logits = edgenet(features_A, features_B, centers_A, centers_B)

    return edge_logits.squeeze().cpu().numpy()


def edgenet_logits_to_matrices(edges, edge_logits, num_patches):
    P = num_patches
    A = np.zeros((P, P), dtype=np.float32)
    B = np.zeros((P, P), dtype=np.float32)

    # edge_logit越大表示越需要翻转
    edge_probs = 1.0 / (1.0 + np.exp(edge_logits))  # Sigmoid

    for idx, (i, j) in enumerate(edges):
        prob_same = edge_probs[idx]
        prob_opposite = 1.0 - prob_same

        # Symmetric assignment
        A[i, j] = A[j, i] = prob_same
        B[i, j] = B[j, i] = prob_opposite

    return A, B


def run_global_optimization(patches, patch_normals, patch_features, patch_centers,
                           config, device, coords, edgenet=None):
    """Run global flip optimization using EdgeNet or geometric method.

    NEW PARAMETERS:
        patch_features: List of (512,) tensors - patch features from M1 backbone
        patch_centers: List of (3,) tensors - patch center coordinates
        edgenet: EdgeNet model (optional)
    """
    opt_config = config['optimization']
    edge_config = config.get('edge_consistency', {})
    mode = edge_config.get('mode', 'geometric')
    method = opt_config['method']

    timings = {}
    print(f"Computing consistency matrix (mode={mode})...")
    t_start = time.time()

    # Select method based on mode
    if mode == 'edgenet' and edgenet is not None:
        # EdgeNet-only mode
        k = edge_config.get('k', 10)
        edges = build_knn_graph_for_edgenet(patch_centers, k=k)
        edge_logits = run_edgenet_inference(edgenet, patch_features, patch_centers,
                                           edges, device)
        A, B = edgenet_logits_to_matrices(edges, edge_logits, len(patches))

    elif mode == 'geometric':
        # Geometric-only mode (current method)
        patch_normals_device = [n.to(device) for n in patch_normals]
        A, B = get_overlap_fast(patches, patch_normals_device, len(coords))
        A, B = A.cpu().numpy(), B.cpu().numpy()

    elif mode == 'hybrid' and edgenet is not None:
        # Hybrid mode: weighted combination
        k = edge_config.get('k', 10)
        edges = build_knn_graph_for_edgenet(patch_centers, k=k)
        edge_logits = run_edgenet_inference(edgenet, patch_features, patch_centers,
                                           edges, device)
        A_edge, B_edge = edgenet_logits_to_matrices(edges, edge_logits, len(patches))

        # Compute geometric matrices
        patch_normals_device = [n.to(device) for n in patch_normals]
        A_geo, B_geo = get_overlap_fast(patches, patch_normals_device, len(coords))
        A_geo, B_geo = A_geo.cpu().numpy(), B_geo.cpu().numpy()

        # Weighted combination
        weights = edge_config.get('hybrid_weights', {'edgenet': 0.7, 'geometric': 0.3})
        w_edge = weights['edgenet']
        w_geo = weights['geometric']
        A = w_edge * A_edge + w_geo * A_geo
        B = w_edge * B_edge + w_geo * B_geo
    else:
        # Fallback to geometric if EdgeNet unavailable
        patch_normals_device = [n.to(device) for n in patch_normals]
        A, B = get_overlap_fast(patches, patch_normals_device, len(coords))
        A, B = A.cpu().numpy(), B.cpu().numpy()

    # Normalize matrices
    sum_AB = A + B
    edge_mask = sum_AB != 0
    A[edge_mask] /= sum_AB[edge_mask]
    B[edge_mask] /= sum_AB[edge_mask]
    timings['edgenet'] = time.time() - t_start
    print(f"Matrix done: A={A.sum():.0f}, B={B.sum():.0f} ({timings['edgenet']:.2f}s)")

    print(f"Optimizing ({method})...")
    t_start = time.time()
    obj_before = compute_objective(np.zeros(len(patches)), A, B)
    print(f"Objective before: {obj_before:.2f}")

    if method == 'vote':
        optimizer = VoteFlipSolver(n_iterations=opt_config.get('vote_iterations', 10))
        flip_decisions, stats = optimizer.solve(A, B)
    else:  # miqp
        optimizer = FlipOptimizer(
            miqp_server_host=opt_config.get('miqp_server_host', '192.168.8.19'),
            miqp_server_port=opt_config.get('miqp_server_port', 11111),
            miqp_timeout=opt_config.get('miqp_timeout', 3000.0)
        )
        flip_decisions, stats = optimizer.solve(A, B)

    obj_after = compute_objective(flip_decisions, A, B)
    timings['optimization'] = time.time() - t_start
    print(f"Objective after: {obj_after:.2f} ({timings['optimization']:.2f}s)")
    stats.update({'objective_before': obj_before, 'objective_after': obj_after, 'A': A, 'B': B})
    return flip_decisions.astype(int), stats, timings


def run_m2_inference(patch_features, patch_centers, config, device):
    features = torch.stack(patch_features).to(device)
    centers = torch.stack(patch_centers).to(device)
    transform = NormalEstimationNormalize()
    centers, _, _ = transform.direct_call(centers)
    batch_offsets = torch.tensor([len(patch_features)], dtype=torch.long).to(device)
    num_patches = len(patch_features)

    m2_config = load_config(config['models']['m2']['config'])
    m2 = create_global_flip_transformer(m2_config)
    ckpt = torch.load(config['models']['m2']['checkpoint'], map_location='cpu')
    m2.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    m2 = m2.to(device).eval()

    m2_iter_config = config.get('m2_iterative', {})
    m2_enabled = m2_iter_config.get('enabled', False)
    m2_num_iterations = m2_iter_config.get('num_iterations', 1) if m2_enabled else 1
    can_iterate = m2_enabled and features.dim() == 3 and features.size(1) == 2

    with torch.no_grad():
        if can_iterate:
            flip_state = torch.zeros(num_patches, dtype=torch.long, device=device)
            for _ in range(m2_num_iterations):
                selected_features = features[torch.arange(num_patches, device=device), flip_state]
                patch_flip_logits = m2(selected_features, centers, batch_offsets)
                should_flip = (torch.sigmoid(patch_flip_logits.squeeze()) > 0.5).long()
                flip_state = flip_state ^ should_flip
            flip_decisions = flip_state.cpu().numpy()
        else:
            patch_flip_logits = m2(features, centers, batch_offsets)
            flip_decisions = (patch_flip_logits.cpu().squeeze() > 0).numpy()

    return flip_decisions

from plyfile import PlyData
def process_single_file(input_path, output_dir, config, m1, device, pipeline_type, cache_dir=None, edgenet=None, no_chunk=False):
    model_name = input_path.stem
    timing = {}
    t0 = time.time()
    gt_normals = None

    print(f"Loading {input_path}")
    
    plydata = PlyData.read(input_path)
    coords = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1).astype(np.float32)

    if 'nx' in plydata['vertex'].data.dtype.names:
        gt_normals = np.stack([
            plydata['vertex']['nx'],
            plydata['vertex']['ny'],
            plydata['vertex']['nz']
        ], axis=1).astype(np.float32)
    # pcd = o3d.io.read_point_cloud(str(input_path))
    # coords = np.asarray(pcd.points, dtype=np.float32)
    # gt_normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
    print(f"Loaded {len(coords)} points")

    cached_data = None
    if cache_dir:
        cached_data = load_inference_cache(cache_dir, model_name, device)
        if cached_data:
            coords_cached, patches, patch_normals, patch_centers, patch_features, _ = cached_data
            if np.array_equal(coords, coords_cached):
                print(f"✓ Cache loaded")
                timing['patch_extraction'] = timing['m1_inference'] = timing['pca'] = 0.0
                if len(patches) > 1:
                    patches, patch_normals, patch_features, patch_centers = \
                        _sort_patches_by_size(patches, patch_normals, patch_features, patch_centers)
            else:
                cached_data = None

    if not cached_data:
        if no_chunk:
            print("No-chunk mode: using entire point cloud as single patch")
            t_start = time.time()
            all_indices = torch.arange(len(coords), dtype=torch.long).to(device)
            patches = [all_indices]
            timing['patch_extraction'] = time.time() - t_start
            print(f"Single patch with {len(coords)} points")
        else:
            print("Extracting patches...")
            t_start = time.time()
            coords_torch = torch.from_numpy(coords).to(device)
            patch_config = config['patch_extraction']
            patches, _ = extract_patches_unified(
                coords_torch,
                method=patch_config.get('method', 'fps'),
                num_per_patch=patch_config.get('max_points_per_patch', 20000),
                patch_count=patch_config.get('patch_count'),
                overlap_rate=patch_config.get('overlap_rate', 0),
                device=device
            )
            timing['patch_extraction'] = time.time() - t_start
            print(f"Extracted {len(patches)} patches ({timing['patch_extraction']:.2f}s)")
            if len(patches) > 1:
                patches, = _sort_patches_by_size(patches)
                print(f"Patch sizes (top 5): {[len(p) for p in patches[:5]]}")

        print("M1 inference...")
        t_start = time.time()

        # Extract features if EdgeNet enabled OR M2 pipeline
        edge_enabled = config.get('edge_consistency', {}).get('enabled', False)
        need_features = (pipeline_type == 'two_stage') or edge_enabled

        if need_features:
            patch_normals, patch_features, patch_centers, pca_time = run_m1_iterative_inference(
                patches, coords, m1, config, device, extract_m2_features=True)
        else:
            patch_normals, pca_time = run_m1_iterative_inference(patches, coords, m1, config, device)
            patch_features = [torch.zeros(512) for _ in patches]  # Dummy features
            patch_centers = [torch.zeros(3) for _ in patches]      # Dummy centers
        total_m1 = time.time() - t_start
        timing['pca'] = pca_time
        timing['m1_inference'] = total_m1 - pca_time
        print(f"M1 done ({total_m1:.2f}s, pca={pca_time:.2f}s)")

        if cache_dir:
            m2_enabled = pipeline_type == 'two_stage' and config.get('m2_iterative', {}).get('enabled', False)
            save_inference_cache(cache_dir, model_name, coords, patches, patch_normals,
                                 patch_centers, patch_features, m2_enabled)

    if pipeline_type == 'optimize':
        flip_decisions, opt_stats, opt_timings = run_global_optimization(
            patches, patch_normals, patch_features, patch_centers,  # Pass features + centers
            config, device, coords, edgenet  # Pass edgenet
        )
        timing['edgenet'] = opt_timings['edgenet']
        timing['optimization'] = opt_timings['optimization']
    else:
        print("M2 inference...")
        t_start = time.time()
        flip_decisions = run_m2_inference(patch_features, patch_centers, config, device)
        timing['m2_inference'] = time.time() - t_start
        print(f"M2 done ({timing['m2_inference']:.2f}s)")
        opt_stats = {}

    print("Merging results...")
    final_normals = np.zeros((len(coords), 3), dtype=np.float32)
    for i, patch_indices in enumerate(patches):
        indices = patch_indices.cpu().numpy()
        norms = patch_normals[i].numpy()
        if flip_decisions[i] == 1:
            norms = -norms
        final_normals[indices] = norms

    if gt_normals is not None:
        from utils.metrics import calculate_edge_accuracy

        gt_flip = []
        for i, patch_indices in enumerate(patches):
            idx = patch_indices.cpu().numpy()
            dots = (patch_normals[i].numpy() * gt_normals[idx]).sum(axis=1)
            gt_flip.append(int((dots < 0).sum() > len(dots) / 2))
        gt_flip = np.array(gt_flip)

        flip_accuracy = max((flip_decisions == gt_flip).mean(), 1 - (flip_decisions == gt_flip).mean())

        # Compute point accuracy (each patch independently chooses direction to maximize correct points)
        total_correct = 0
        total_count = 1
        patch_acc = []
        for i, patch_indices in enumerate(patches):
            indices = patch_indices.cpu().numpy()
            patch_pred = patch_normals[i].numpy()
            patch_gt = gt_normals[indices]
            # Calculate correct/error counts with current direction
            dots = (patch_pred * patch_gt).sum(axis=1)
            correct_count = (dots > 0).sum()
            error_count = (dots < 0).sum()
            total_count += len(dots)
            total_correct += max(correct_count, error_count)
            patch_acc.append(max(correct_count, error_count) / len(dots))


        point_accuracy = total_correct / total_count

        correct_mask = (final_normals * gt_normals).sum(axis=1) > 0
        final_accuracy = max(correct_mask.mean(), 1 - correct_mask.mean())

        print(f"Flip accuracy: {flip_accuracy*100:.2f}%")
        print(f"Point accuracy (per-patch best): {point_accuracy*100:.2f}%")
        print(f"Final accuracy (global optimized): {final_accuracy*100:.2f}%")

        # Compute GT flip per-point accuracy (what accuracy would be if all patches flipped correctly)
        final_normals_gt = np.zeros((len(coords), 3), dtype=np.float32)
        for i, patch_indices in enumerate(patches):
            indices = patch_indices.cpu().numpy()
            norms = patch_normals[i].numpy()

            # Compute GT flip for this patch
            patch_gt = gt_normals[indices]
            dots = (norms * patch_gt).sum(axis=1)
            gt_flip_patch = int((dots < 0).sum() > len(dots) / 2)

            if gt_flip_patch == 1:
                norms = -norms
            final_normals_gt[indices] = norms

        correct_mask_gt = (final_normals_gt * gt_normals).sum(axis=1) > 0
        overall_acc_gt = max(correct_mask_gt.mean(), 1 - correct_mask_gt.mean())
        print(f"Overall Accuracy (GT flip): {overall_acc_gt*100:.2f}%")

        colors = np.zeros((len(coords), 3), dtype=np.uint8)
        colors[correct_mask] = [0, 255, 0]
        colors[~correct_mask] = [255, 0, 0]

        if pipeline_type == 'optimize':
            objective_gt = compute_objective(gt_flip, opt_stats.get('A', np.zeros((len(patches), len(patches)))),
                                             opt_stats.get('B', np.zeros((len(patches), len(patches)))))
            # Compute edge accuracy
            edge_accuracy, total_edges = calculate_edge_accuracy(
                opt_stats.get('A', np.zeros((len(patches), len(patches)))),
                opt_stats.get('B', np.zeros((len(patches), len(patches)))),
                gt_flip
            )
            print(f"Edge Accuracy: {edge_accuracy:.2f}% ({total_edges} edges)")
        else:
            objective_gt = None
            edge_accuracy = None
            total_edges = 0
    else:
        flip_accuracy = point_accuracy = final_accuracy = overall_acc_gt = objective_gt = None
        edge_accuracy = None
        total_edges = 0
        colors = None

    timing['total'] = time.time() - t0

    print("Saving...")
    write_ply_ascii(str(output_dir / f"{model_name}.ply"), coords, normals=final_normals)
    if colors is not None:
        write_ply_ascii(str(output_dir / f"{model_name}_colored.ply"), coords, normals=final_normals, colors=colors)

        # Save downsampled version if point cloud is large (> 10 million points)
        if len(coords) > 10_000_000:
            print(f"Large point cloud ({len(coords)} points), saving 0.1 downsampled version...")
            ds_coords, ds_normals, ds_colors = random_downsample(
                coords, final_normals, colors, ratio=0.1, seed=42
            )
            write_ply_ascii(
                str(output_dir / f"{model_name}_colored_ds01.ply"),
                ds_coords,
                normals=ds_normals,
                colors=ds_colors
            )
            print(f"  Downsampled: {len(ds_coords)} points saved")

    if config['inference'].get('visualize_graphs', False) and gt_normals is not None:
        try:
            patch_centers_vis = [torch.from_numpy(coords[p.cpu().numpy()]).mean(dim=0) for p in patches]
            patch_normals_t = [n.to(patches[0].device) for n in patch_normals]
            A, B = get_overlap_fast(patches, patch_normals_t, coords.shape[0])
            visualize_patch_graphs(patches, patch_normals, patch_centers_vis, gt_normals, output_dir, A=A, B=B)
        except Exception as e:
            print(f"Viz failed: {e}")

    overall_metrics = None
    if gt_normals is not None:
        from utils.metrics import calculate_normal_metrics, calculate_metrics_inv
        correct_mask = (final_normals * gt_normals).sum(axis=1) > 0
        final_normals_for_metrics = -final_normals if correct_mask.mean() < 0.5 else final_normals
        overall_metrics = calculate_normal_metrics(torch.from_numpy(final_normals_for_metrics), torch.from_numpy(gt_normals))

        # Compute per-patch metrics
        per_patch_metrics = []
        patch_accs = []

        for i, patch_indices in enumerate(patches):
            indices = patch_indices.cpu().numpy()

            # Get patch normals (after global flip decision)
            patch_pred = patch_normals[i].numpy()
            if flip_decisions[i] == 1:
                patch_pred = -patch_pred

            patch_gt = gt_normals[indices]

            # Calculate angular metrics for this patch
            patch_metrics = calculate_normal_metrics(
                torch.from_numpy(patch_pred),
                torch.from_numpy(patch_gt)
            )

            # Calculate accuracy (with per-patch optimal flip)
            dots = (patch_pred * patch_gt).sum(axis=1)
            patch_pred_labels = (dots > 0).astype(np.float32)
            patch_gt_labels = np.ones(len(indices), dtype=np.float32)

            patch_pred_labels_torch = torch.from_numpy(patch_pred_labels)
            patch_gt_labels_torch = torch.from_numpy(patch_gt_labels)

            logits = torch.logit(torch.clamp(patch_pred_labels_torch, 0.01, 0.99))
            acc_inv, iou, precision, recall, mean_gt = calculate_metrics_inv(
                logits, patch_gt_labels_torch
            )
            error_rate = 1.0 - acc_inv

            per_patch_metrics.append({
                "patch_id": i,
                "num_points": len(indices),
                "accuracy_inv": float(acc_inv),
                "error_rate": float(error_rate),
                **{k: float(v) for k, v in patch_metrics.items()}
            })

            # Compute GT flip for this patch
            gt_flip_patch = int((dots < 0).sum() > len(dots) / 2)

            patch_accs.append({
                "patch_id": i,
                "num_points": len(indices),
                "accuracy": float(acc_inv),
                "accuracy_inv": float(acc_inv),
                "error_rate": float(error_rate),
                "num_correct": int((dots > 0).sum()),
                "num_error": int((dots < 0).sum()),
                "flip_pred": int(flip_decisions[i]),
                "gt_flip": gt_flip_patch
            })
    else:
        per_patch_metrics = None
        patch_accs = None

    metrics = {
        "model_name": model_name,
        "num_points": len(coords),
        "num_patches": len(patches),
        "pipeline_type": pipeline_type,
        "accuracy": {
            "point_accuracy": float(point_accuracy) if point_accuracy else None,
            "final_accuracy": float(final_accuracy) if final_accuracy else None,
            "overall_gt_flip": float(overall_acc_gt) if overall_acc_gt else None,
            "flip_decision": float(flip_accuracy) if flip_accuracy else None,
            "edge_accuracy": float(edge_accuracy) if edge_accuracy is not None else None,
            "total_edges": int(total_edges) if total_edges > 0 else 0,
        },
        "overall_metrics": overall_metrics,
        "per_patch_accuracy": patch_accs,
        "per_patch_metrics": per_patch_metrics,
        "timing": timing
    }

    if pipeline_type == 'optimize':
        metrics["optimization_stats"] = {
            "objective_before": opt_stats.get('objective_before'),
            "objective_after": opt_stats.get('objective_after'),
            "objective_gt": float(objective_gt) if objective_gt else None,
        }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)

    # Save per-patch segmentations if requested
    if config['inference'].get('save_patches', False) and patch_accs is not None:
        print("Saving per-patch segmentation...")
        segs_dir = output_dir / "segs"
        segs_dir.mkdir(exist_ok=True)

        save_filter = config['inference'].get('save_patches_filter', 'all')
        saved_patches_info = []

        for i, patch_indices in enumerate(patches):
            should_save = False

            if save_filter == 'all':
                should_save = True
            elif isinstance(save_filter, dict):
                if 'min_error_rate' in save_filter:
                    if patch_accs[i]['error_rate'] >= save_filter['min_error_rate']:
                        should_save = True
                elif 'max_error_rate' in save_filter:
                    if patch_accs[i]['error_rate'] <= save_filter['max_error_rate']:
                        should_save = True

            if should_save:
                indices = patch_indices.cpu().numpy()
                patch_coords = coords[indices]

                # Get final normals for this patch
                patch_norms = patch_normals[i].numpy()
                if flip_decisions[i] == 1:
                    patch_norms = -patch_norms

                # Save patch PLY
                patch_ply = segs_dir / f"seg_{i}.ply"
                write_ply_ascii(str(patch_ply), patch_coords, normals=patch_norms)

                # Save colored version if GT available
                if gt_normals is not None:
                    patch_gt = gt_normals[indices]
                    correct_mask_patch = (patch_norms * patch_gt).sum(axis=1) > 0

                    patch_colors = np.zeros((len(indices), 3), dtype=np.uint8)
                    patch_colors[correct_mask_patch] = [0, 255, 0]  # Green = correct
                    patch_colors[~correct_mask_patch] = [255, 0, 0]  # Red = error

                    patch_ply_colored = segs_dir / f"seg_{i}_colored.ply"
                    write_ply_ascii(str(patch_ply_colored), patch_coords,
                                  normals=patch_norms, colors=patch_colors)

                saved_patches_info.append({
                    "patch_id": i,
                    "filename": f"seg_{i}.ply",
                    "num_points": len(indices),
                    "accuracy": patch_accs[i].get("accuracy", None),
                    "accuracy_inv": patch_accs[i].get("accuracy_inv", None),
                    "error_rate": patch_accs[i].get("error_rate", None),
                    "flip_pred": patch_accs[i].get("flip_pred", None),
                    "gt_flip": patch_accs[i].get("gt_flip", None)
                })

        # Save patch info
        with open(segs_dir / "patch_info.json", 'w') as f:
            json.dump({
                "total_patches": len(patches),
                "saved_patches": len(saved_patches_info),
                "filter": save_filter,
                "patches": saved_patches_info
            }, f, indent=2)

        print(f"Saved {len(saved_patches_info)}/{len(patches)} patches to {segs_dir}")

    print(f"Done! {timing['total']:.2f}s")
    if overall_metrics:
        print(f"Mean error: {overall_metrics['Mean_Error']:.2f}°")
    return metrics


def save_summary_csv(csv_path, results):
    import csv
    fieldnames = [
        'model_name', 'scene_name', 'input_path', 'output_dir', 'status',
        'num_points', 'num_patches', 'pipeline_type',
        'point_accuracy', 'final_accuracy', 'flip_accuracy', 'mean_error',
        'time_total', 'error_message'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            status = result.get('status', 'success')
            if status in {'failed', 'skipped'}:
                row = {
                    'model_name': result.get('model_name'),
                    'scene_name': result.get('scene_name'),
                    'input_path': result.get('input_path'),
                    'output_dir': result.get('output_dir'),
                    'status': status,
                    'error_message': result.get('error', ''),
                }
            else:
                acc = result.get('accuracy', {})
                om = result.get('overall_metrics', {})
                row = {
                    'model_name': result['model_name'],
                    'scene_name': result.get('scene_name'),
                    'input_path': result.get('input_path'),
                    'output_dir': result.get('output_dir'),
                    'status': 'success',
                    'num_points': result.get('num_points'),
                    'num_patches': result.get('num_patches'),
                    'pipeline_type': result.get('pipeline_type'),
                    'point_accuracy': acc.get('point_accuracy', 0) * 100 if acc.get('point_accuracy') else None,
                    'final_accuracy': acc.get('final_accuracy', 0) * 100 if acc.get('final_accuracy') else None,
                    'flip_accuracy': acc.get('flip_decision', 0) * 100 if acc.get('flip_decision') else None,
                    'mean_error': om.get('Mean_Error') if om else None,
                    'time_total': result.get('timing', {}).get('total'),
                    'error_message': ''
                }
            for field in fieldnames:
                if field not in row:
                    row[field] = None
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--input-mode', default='auto', choices=['auto', 'file', 'scene_dir', 'scan_root'])
    parser.add_argument('--pattern', default='*_raw_pointcloud.ply',
                        help="File pattern used for directory discovery (default: '*_raw_pointcloud.ply')")
    parser.add_argument('--recursive', action='store_true',
                        help='Treat directory input as a scan root and search child scene directories')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache (neither read nor write)')
    parser.add_argument('--no-chunk', action='store_true', help='Skip patch extraction, run M1 on entire point cloud as a single patch')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.input:
        config['input'] = args.input

    input_path = Path(config['input'])
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    pipeline_type = detect_pipeline_type(config)
    print(f"Pipeline: {pipeline_type}")

    if args.output:
        output_root = Path(args.output)
    else:
        output_root = generate_output_path(args.config, input_path)
        config['output'] = str(output_root)

    print(f"Output: {output_root}")
    targets, skipped_inputs, resolved_mode = resolve_input_targets(
        input_path,
        input_mode=args.input_mode,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    print(f"Input mode: {resolved_mode}")
    print(f"Discovered {len(targets)} file(s)")
    if skipped_inputs:
        print(f"Skipped {len(skipped_inputs)} input(s) without a valid target file")

    if args.dry_run:
        for target in targets[:10]:
            print(f"  - {target['ply_path']}")
        if len(targets) > 10:
            print(f"  ... ({len(targets) - 10} more)")
        return

    if not targets:
        print(f"No matching .ply files resolved from {input_path}")
        return

    if pipeline_type == 'optimize':
        m1 = load_m1_model(config, device)
        edgenet = load_edgenet_model(config, device)  # Load EdgeNet for optimization pipeline
    else:
        m1, _ = load_models(config, device)
        edgenet = None  # M2 pipeline doesn't use EdgeNet

    cache_dir = None if args.no_cache else get_inference_cache_dir(str(output_root), config, config['models']['m1']['checkpoint'])
    print(cache_dir)
    if cache_dir:
        save_inference_cache_metadata(cache_dir, config, config['models']['m1']['checkpoint'])

    all_results = list(skipped_inputs)
    for i, target in enumerate(targets):
        ply_path = target['ply_path']
        print(f"\n{'='*60}\n[{i+1}/{len(targets)}]: {ply_path.name}\n{'='*60}")

        output_dir = output_root / target['output_name']
        if output_dir.exists() and not config['inference'].get('overwrite', True):
            print("Skipping (exists)")
            all_results.append({
                'model_name': ply_path.stem,
                'scene_name': target.get('scene_name'),
                'input_path': str(ply_path),
                'output_dir': str(output_dir),
                'status': 'skipped',
                'error': 'output directory already exists and overwrite=false',
            })
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics = process_single_file(ply_path, output_dir, config, m1, device, pipeline_type, cache_dir, edgenet, no_chunk=args.no_chunk)
            metrics['scene_name'] = target.get('scene_name')
            metrics['input_path'] = str(ply_path)
            metrics['output_dir'] = str(output_dir)
            metrics['status'] = 'success'
            all_results.append(metrics)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({
                'model_name': ply_path.stem,
                'scene_name': target.get('scene_name'),
                'input_path': str(ply_path),
                'output_dir': str(output_dir),
                'status': 'failed',
                'error': str(e),
            })

    if len(targets) > 1 or skipped_inputs:
        # Save summary to centralized summary directory
        inference_root = output_root.parent.parent  # outputs/inference/
        summary_dir = inference_root / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique summary name including input path info
        config_name = Path(args.config).stem
        input_name = input_path.name if input_path.is_dir() else input_path.stem

        # Append input_name if different from config_name to avoid redundancy
        if input_name and input_name != config_name:
            summary_name = f"{config_name}_{input_name}"
        else:
            summary_name = config_name

        csv_path = summary_dir / f'{summary_name}_summary.csv'
        save_summary_csv(csv_path, all_results)
        print(f"\nSummary: {csv_path}")


if __name__ == '__main__':
    main()
