"""
Precompute Multi-Scale Patch Features for Global Flip Optimization
预计算多尺度patch特征并缓存到磁盘
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.multi_scale_patch_dataset import MultiScalePatchDataset
from dataset.dataset import estimate_normals_torch
from dataset.transforms import NormalEstimationNormalize
from models.direct_orientation_model import create_direct_orientation_model
from utils.config import load_config, load_and_merge_feat_extraction_config
from utils.metrics import calculate_metrics_inv
from utils.cache_utils import get_global_flip_cache_dir, save_cache_metadata


def compute_patch_features_single_scale(
    coords, gt_normals, patch_list, backbone, config, backbone_config, pooling_method='mean', use_encoder_features=False
):
    """对一个scale的所有patches计算特征

    Args:
        use_encoder_features: bool - 是否使用encoder特征（而非decoder特征）

    Returns:
        patch_centers: (P, 3)
        gt_flip_status: (P,)
        features: (P, 512)
        inv_features: (P, 512) or None - 仅在m2_iterative.enabled=true时提取
        patch_metrics: List[Dict] - 每个patch的准确率统计
    """
    grid_size = config.get('grid_size', 0.02)
    pca_max_nn = config.get('pca_max_nn', 30)

    # 优先读取 patch_iterative，回退到 backbone_config 的 iterative
    patch_iter_config = config.get('patch_iterative', backbone_config.get('iterative', {}))
    num_iterations = patch_iter_config.get('num_iterations', 3)
    use_confidence = patch_iter_config.get('use_confidence', True)

    # M2迭代配置：决定是否提取inv_features
    m2_iter_config = config.get('m2_iterative', {})
    extract_inv_features = m2_iter_config.get('enabled', False)

    batch_size = config.get('batch_size', 8)

    patch_centers_list = []
    gt_flip_status_list = []
    features_list = []
    inv_features_list = [] if extract_inv_features else None
    pred_normals_list = []
    confidence_list = []
    patch_metrics_list = []

    # 预计算所有PCA法向量
    pca_normals_all = []
    for patch_indices in patch_list:
        patch_coords = coords[patch_indices]
        result = estimate_normals_torch(patch_coords.cpu().numpy(), max_nn=pca_max_nn)
        pca_normals = torch.tensor(result[:, 3:6], dtype=torch.float32, device=patch_coords.device)
        pca_normals_all.append(pca_normals)

    # 批处理迭代refinement
    transform = NormalEstimationNormalize()
    num_batches = (len(patch_list) + batch_size - 1) // batch_size

    with torch.no_grad():
        # backbone.cls_mode = True
        pbar_batches = tqdm(range(num_batches), desc="  Refining patches (batched)", leave=False)
        for batch_idx in pbar_batches:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(patch_list))
            batch_patches = patch_list[batch_start:batch_end]

            # 初始化batch数据
            patch_data_list = []
            for i, patch_indices in enumerate(batch_patches):
                patch_coords = coords[patch_indices]
                normals = pca_normals_all[batch_start + i].clone()
                conf = torch.zeros(len(normals), 1, device=coords.device) if use_confidence else None
                patch_data_list.append([patch_coords, normals, conf])

            # 预先transform coords (只需要做一次)
            transformed_coords_list = []
            offsets_list = []
            for coords_i, _, _ in patch_data_list:
                pd = {'coord': coords_i, 'feat': coords_i,
                      'offset': torch.tensor([len(coords_i)], device=coords.device).long(), 'grid_size': grid_size}
                pd, _ = transform(pd)
                transformed_coords_list.append(pd['coord'])
                offsets_list.append(len(coords_i))

            # 预先打包coords和offset (不会变化)
            batch_coords = torch.cat(transformed_coords_list, dim=0)
            batch_offsets = torch.cumsum(torch.tensor(offsets_list, device=coords.device), dim=0).long()

            # 迭代refinement
            try:
                for iter_idx in range(num_iterations):
                    # 组合feats (coords已归一化，只需更新normals和conf部分)
                    batch_feats_list = []
                    start_idx = 0
                    for i, (_, normals_i, conf_i) in enumerate(patch_data_list):
                        end_idx = start_idx + len(normals_i)
                        normalized_coords_i = batch_coords[start_idx:end_idx]
                        if use_confidence:
                            feat = torch.cat([normalized_coords_i, normals_i, conf_i], dim=1)
                        else:
                            feat = torch.cat([normalized_coords_i, normals_i], dim=1)
                        batch_feats_list.append(feat)
                        start_idx = end_idx

                    batch_feats = torch.cat(batch_feats_list, dim=0)

                    # Batch forward
                    logits = backbone({'coord': batch_coords, 'feat': batch_feats, 'offset': batch_offsets, 'grid_size': grid_size})[:, 0]

                    # Update normals
                    flip_prob = torch.sigmoid(logits)  # 保持在GPU上
                    start_idx = 0
                    for i, (coords_i, normals_i, conf_i) in enumerate(patch_data_list):
                        end_idx = start_idx + len(coords_i)
                        flip_mask = flip_prob[start_idx:end_idx] > 0.5
                        normals_i[flip_mask] = -normals_i[flip_mask]
                        if use_confidence:
                            patch_data_list[i][2] = (torch.abs(flip_prob[start_idx:end_idx] - 0.5) * 2).unsqueeze(1)
                        start_idx = end_idx
            except RuntimeError as e:
                print(f"\n[ERROR] RuntimeError in batch {batch_idx}, iteration {iter_idx}")
                print(f"  Batch range: patches [{batch_start}:{batch_end}]")
                print(f"  Number of patches in batch: {len(batch_patches)}")
                print(f"  Patch sizes: {[len(p) for p in batch_patches]}")
                print(f"  Total points in batch: {len(batch_coords)}")
                print(f"  Batch coords shape: {batch_coords.shape}")
                print(f"  Batch feats shape: {batch_feats.shape}")
                print(f"  Batch offsets: {batch_offsets.tolist()}")
                print(f"  Error message: {str(e)}")
                raise

            # 收集结果并计算metrics
            for i, (_, normals, conf) in enumerate(patch_data_list):
                patch_idx = batch_start + i
                patch_indices = patch_list[patch_idx]
                patch_gt_normals = gt_normals[patch_indices]
                patch_size = len(patch_indices)

                pred_normals = normals
                pca_normals = pca_normals_all[patch_idx]

                # Compute GT flip status (per point)
                gt_flip_per_point = ((pca_normals * patch_gt_normals).sum(dim=1) < 0).long()
                pred_flip = ((pred_normals * pca_normals).sum(dim=1) < 0).float()

                # 使用calculate_metrics_inv计算准确率
                accuracy, iou, precision, recall, mean_gt = calculate_metrics_inv(
                    pred_flip.cpu(), gt_flip_per_point.cpu().float()
                )

                # 记录metrics
                patch_metrics_list.append({
                    'patch_idx': patch_idx,
                    'patch_size': patch_size,
                    'accuracy': accuracy,
                    'mean_gt': mean_gt
                })

                # 计算patch级别的GT flip status (majority vote)
                error_count = ((pred_normals * patch_gt_normals).sum(dim=1) < 0).sum()
                gt_flip_status = error_count > (len(pred_normals) / 2)

                pred_normals_list.append(pred_normals)
                if use_confidence:
                    confidence_list.append(conf)
                gt_flip_status_list.append(gt_flip_status.cpu().numpy())

            # Update progress bar with accuracy
            if len(patch_metrics_list) > 0:
                avg_acc = np.mean([m['accuracy'] for m in patch_metrics_list])
                pbar_batches.set_postfix({'avg_acc': f'{100*avg_acc:.1f}%', 'batch': f'{batch_end}/{len(patch_list)}'})

    # 批处理特征提取：使用refinement后的normals
    with torch.no_grad():
        # backbone.cls_mode = True
        pbar_features = tqdm(range(num_batches), desc="  Extracting features (batched)", leave=False)
        for batch_idx in pbar_features:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(patch_list))

            # Transform patches
            transformed_data = []
            for i in range(batch_start, batch_end):
                patch_coords = coords[patch_list[i]]
                normals = pred_normals_list[i]
                conf = confidence_list[i] if use_confidence and confidence_list else None

                feat = torch.cat([patch_coords, normals, conf], dim=1) if use_confidence and conf is not None else torch.cat([patch_coords, normals], dim=1)
                pd = {'coord': patch_coords, 'feat': feat,
                      'offset': torch.tensor([len(patch_coords)], device=coords.device).long(), 'grid_size': grid_size}
                pd, _ = transform(pd)
                transformed_data.append((pd['coord'], pd['feat'], len(patch_coords)))

            # Batch forward
            batch_coords = torch.cat([d[0] for d in transformed_data], dim=0)
            batch_feats = torch.cat([d[1] for d in transformed_data], dim=0)
            batch_offsets = torch.cumsum(torch.tensor([d[2] for d in transformed_data], device=coords.device), dim=0).long()
            backbone_output = backbone.backbone({'coord': batch_coords, 'feat': batch_feats, 'offset': batch_offsets, 'grid_size': grid_size},use_decoder=not use_encoder_features)
            point_features = backbone_output['feat']  # (N_total, 512)

            # 仅在启用M2迭代时提取inv_features
            if extract_inv_features:
                inv_feat = batch_feats.clone()
                inv_feat[:, 3:6] *= -1
                backbone_invoutput = backbone.backbone({'coord': batch_coords, 'feat': inv_feat, 'offset': batch_offsets, 'grid_size': grid_size},use_decoder=not use_encoder_features)
                inv_point_features = backbone_invoutput['feat']
            else:
                inv_point_features = None

            # Split and pool
            start_idx = 0
            if extract_inv_features:
                assert torch.equal(backbone_output['offset'],backbone_invoutput['offset'])
            for i, end_idx in enumerate(backbone_output['offset']):
                # 使用配置的池化方法
                if pooling_method == 'max':
                    patch_feature = point_features[start_idx:end_idx].max(dim=0)[0]  # (512,) - [0]获取值而非索引
                    if extract_inv_features:
                        inv_patch_feature = inv_point_features[start_idx:end_idx].max(dim=0)[0]
                else:  # 'mean' (default)
                    patch_feature = point_features[start_idx:end_idx].mean(dim=0)  # (512,)
                    if extract_inv_features:
                        inv_patch_feature = inv_point_features[start_idx:end_idx].mean(dim=0)

                features_list.append(patch_feature.cpu())
                if extract_inv_features:
                    inv_features_list.append(inv_patch_feature.cpu())

                patch_centers_list.append(coords[patch_list[batch_start + i]].mean(dim=0).cpu())
                start_idx = end_idx

    # 转换为numpy数组
    patch_centers = torch.stack(patch_centers_list).cpu().numpy()  # (P, 3)
    gt_flip_status = np.array(gt_flip_status_list, dtype=np.int64)  # (P,)
    features = torch.stack(features_list).cpu().numpy()  # (P, 512)
    inv_features = torch.stack(inv_features_list).cpu().numpy() if extract_inv_features else None  # (P, 512) or None

    return patch_centers, gt_flip_status, features, inv_features, patch_metrics_list


def compute_patch_features_multiscale(
    coords, gt_normals, patches_by_scale, backbone, config, backbone_config, pooling_method='mean', use_encoder_features=False
):
    """对所有scales计算特征

    Args:
        patches_by_scale: Dict[scale_idx -> List[patch_indices]]
        pooling_method: str - 池化方法 ('mean' or 'max')
        use_encoder_features: bool - 是否使用encoder特征

    Returns:
        features_by_scale: Dict[scale_idx -> {
            'patch_centers': (P, 3),
            'gt_flip_status': (P,),
            'features': (P, 512),
            'inv_features': (P, 512) or None,
            'patch_metrics': List[Dict]
        }]
    """
    features_by_scale = {}

    for scale_idx, patch_list in patches_by_scale.items():
        print(f"  Computing features for scale {scale_idx}: {len(patch_list)} patches")
        features_by_scale[scale_idx] = compute_patch_features_single_scale(
            coords, gt_normals, patch_list, backbone, config, backbone_config, pooling_method, use_encoder_features
        )

    return features_by_scale


def save_multiscale_features(
    save_path,
    features_by_scale,
    scale_configs,
    model_name
):
    """保存多尺度特征到单个.npz文件

    File format:
        num_scales: int
        scale_configs: str (JSON)
        model_name: str
        scale_0_patch_centers: (P0, 3)
        scale_0_gt_flip_status: (P0,)
        scale_0_features: (P0, 512)
        scale_0_num_patches: int
        scale_0_max_points: int
        scale_0_method: str
        ...
    """
    save_dict = {
        'num_scales': len(features_by_scale),
        'scale_configs': json.dumps(scale_configs),
        'model_name': model_name
    }

    for scale_idx, data in features_by_scale.items():
        patch_centers, gt_flip_status, features, inv_features, _ = data
        prefix = f'scale_{scale_idx}_'

        save_dict[prefix + 'patch_centers'] = patch_centers
        save_dict[prefix + 'gt_flip_status'] = gt_flip_status
        save_dict[prefix + 'features'] = features
        if inv_features is not None:
            save_dict[prefix + 'inv_features'] = inv_features
        save_dict[prefix + 'num_patches'] = len(features)
        save_dict[prefix + 'max_points'] = scale_configs[scale_idx]['max_points_per_patch']
        save_dict[prefix + 'method'] = scale_configs[scale_idx]['method']

    np.savez(save_path, **save_dict)


def check_existing_cache(cache_dir, patch_dataset):
    """检查缓存目录中已存在的模型文件

    Returns:
        existing_models: Set[str] - 已存在的模型文件名（不带扩展名）
        action: str - 用户选择的操作 ('overwrite', 'skip', 'cancel')
    """
    if not os.path.exists(cache_dir):
        return set(), 'overwrite'

    # 扫描已存在的.npz文件
    existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
    existing_models = set([Path(f).stem for f in existing_files])

    if len(existing_models) == 0:
        return set(), 'overwrite'

    # 统计状态
    total_models = patch_dataset.get_num_models()
    all_model_names = set([Path(patch_dataset.get_model_name(i)).stem for i in range(total_models)])
    missing_models = all_model_names - existing_models

    # 汇报情况
    print(f"\n{'='*60}")
    print(f"Cache directory already exists: {cache_dir}")
    print(f"{'='*60}")
    print(f"Total models: {total_models}")
    print(f"Cached models: {len(existing_models)} ({100*len(existing_models)/total_models:.1f}%)")
    print(f"Missing models: {len(missing_models)} ({100*len(missing_models)/total_models:.1f}%)")

    if len(missing_models) > 0 and len(missing_models) <= 10:
        print(f"\nMissing model files:")
        for name in sorted(missing_models):
            print(f"  - {name}")
    elif len(missing_models) > 10:
        print(f"\nMissing model files (showing first 10):")
        for name in sorted(missing_models)[:10]:
            print(f"  - {name}")
        print(f"  ... and {len(missing_models)-10} more")

    # 询问用户
    print(f"\nWhat would you like to do?")
    print(f"  [o] Overwrite all - Recompute all models (existing cache will be deleted)")
    print(f"  [s] Skip existing - Only compute missing {len(missing_models)} models")
    print(f"  [c] Cancel - Exit without changes")

    while True:
        choice = input(f"\nYour choice [o/s/c]: ").strip().lower()
        if choice in ['o', 'overwrite']:
            print(f"Will overwrite all cached models.")
            return existing_models, 'overwrite'
        elif choice in ['s', 'skip']:
            if len(missing_models) == 0:
                print(f"All models are already cached. Nothing to do.")
                return existing_models, 'cancel'
            print(f"Will skip {len(existing_models)} existing models and compute {len(missing_models)} missing models.")
            return existing_models, 'skip'
        elif choice in ['c', 'cancel']:
            print(f"Operation cancelled.")
            return existing_models, 'cancel'
        else:
            print(f"Invalid choice. Please enter 'o', 's', or 'c'.")


def precompute_features(config, split='train'):
    """预计算指定split的所有模型的多尺度特征

    Args:
        config: 配置字典
        split: 数据集split ('train', 'val', 'test')
    """
    device = torch.device(config['device'])

    # 1. 加载backbone
    print("Loading backbone...")
    backbone_config_path = config['backbone']['config']
    backbone_config = load_config(backbone_config_path)
    backbone = create_direct_orientation_model(backbone_config)

    checkpoint_path = config['backbone']['checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['model_state_dict'])
    else:
        backbone.load_state_dict(checkpoint)

    backbone = backbone.to(device)
    backbone.eval()
    backbone.freeze_backbone(freeze=True)

    # 读取 patch_iterative 和 m2_iterative 配置
    patch_iter_config = config.get('patch_iterative', backbone_config.get('iterative', {}))
    m2_iter_config = config.get('m2_iterative', {})

    print(f"Backbone loaded from {checkpoint_path}")
    print(f"Patch iterative config: num_iterations={patch_iter_config.get('num_iterations', 3)}, "
          f"use_confidence={patch_iter_config.get('use_confidence', True)}")
    print(f"M2 iterative config: enabled={m2_iter_config.get('enabled', False)}, "
          f"num_iterations={m2_iter_config.get('num_iterations', 1)}")

    # 加载特征提取配置
    feature_config = config.get('feature_extraction', {})
    use_encoder_features = feature_config.get('use_encoder_features', False)
    pooling_method = feature_config.get('pooling_method', 'mean')

    # 验证池化方法
    assert pooling_method in ['mean', 'max'], f"Invalid pooling_method: {pooling_method}. Must be 'mean' or 'max'."

    print(f"\nFeature extraction settings:")
    print(f"  - use_encoder_features: {use_encoder_features} ({'encoder only' if use_encoder_features else 'encoder+decoder'})")
    print(f"  - pooling_method: {pooling_method}")
    print(f"  Note: use_decoder parameter will be passed to forward() dynamically")


    # 2. 创建MultiScalePatchDataset
    print(f"Creating MultiScalePatchDataset for {split}...")
    data_root = os.path.join(config['data']['root'], config['data'][split]['subfolder'])
    scales = config['patch_extraction']['scales']

    patch_dataset = MultiScalePatchDataset(
        data_root=data_root,
        scales=scales,
        device=str(device)
    )
    print(f"Found {patch_dataset.get_num_models()} models with {len(patch_dataset)} patches total")
    print(f"Scales: {len(scales)}")
    for i, scale in enumerate(scales):
        print(f"  Scale {i}: {scale}")

    # 3. 创建缓存目录（基于配置hash自动生成）
    cache_root = get_global_flip_cache_dir(config, backbone_config)
    cache_dir = os.path.join(cache_root, split)

    # 检查已存在的缓存
    existing_models, action = check_existing_cache(cache_dir, patch_dataset)

    if action == 'cancel':
        print(f"Skipping {split} split.")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    print(f"\nCache root: {cache_root}")
    print(f"Cache directory for {split}: {cache_dir}")

    # 4. 遍历所有模型，计算多尺度特征
    print(f"\nPrecomputing multi-scale features for {split} split...")

    all_metrics = []  # 收集所有patch的metrics
    num_processed = 0
    num_skipped = 0

    pbar = tqdm(range(patch_dataset.get_num_models()))
    for model_idx in pbar:
        model_name = patch_dataset.get_model_name(model_idx)
        model_stem = Path(model_name).stem
        save_name = model_stem + '.npz'
        save_path = os.path.join(cache_dir, save_name)

        # 检查是否跳过（如果用户选择skip且文件已存在）
        if action == 'skip' and model_stem in existing_models:
            num_skipped += 1
            pbar.set_postfix({
                'processed': num_processed,
                'skipped': num_skipped,
                'status': 'skip'
            })
            continue

        # 获取模型的所有patches (all scales)
        model_data = patch_dataset.get_model_patches_all_scales(model_idx)
        coords = model_data['coords'].to(device)
        gt_normals = model_data['gt_normals'].to(device)
        patches_by_scale = model_data['patches_by_scale']

        # 计算多尺度特征
        features_by_scale = compute_patch_features_multiscale(
            coords, gt_normals, patches_by_scale, backbone, config, backbone_config, pooling_method, use_encoder_features
        )

        # 保存特征

        save_multiscale_features(save_path, features_by_scale, scales, model_name)
        num_processed += 1

        # 收集metrics (add scale_idx)
        for scale_idx, (_, _, _, _, patch_metrics) in features_by_scale.items():
            for metrics_dict in patch_metrics:
                metrics_dict['model_name'] = model_name
                metrics_dict['model_idx'] = model_idx
                metrics_dict['scale_idx'] = scale_idx
                all_metrics.append(metrics_dict)

        # 计算统计指标并更新进度条
        total_patches = sum(len(p) for p in patches_by_scale.values())
        if len(all_metrics) > 0:
            recent_metrics = all_metrics[-total_patches:] if len(all_metrics) >= total_patches else all_metrics
            avg_acc = np.mean([m['accuracy'] for m in recent_metrics])
        else:
            avg_acc = 0.0

        pbar.set_postfix({
            'processed': num_processed,
            'skipped': num_skipped,
            'acc': f'{100*avg_acc:.1f}%'
        })

    # 打印处理摘要
    print(f"\n{'='*60}")
    print(f"Processing summary for {split}:")
    print(f"  Processed: {num_processed} models")
    print(f"  Skipped: {num_skipped} models")
    print(f"  Total: {patch_dataset.get_num_models()} models")
    print(f"{'='*60}")

    # 保存CSV（只保存本次处理的metrics）
    if len(all_metrics) > 0:
        csv_path = os.path.join(cache_dir, 'patch_metrics.csv')
        df = pd.DataFrame(all_metrics)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to {csv_path}")
    else:
        df = pd.DataFrame()  # Empty dataframe for skip-all case
        print(f"\nNo new models processed, skipping metrics CSV.")

    # 打印统计信息（仅当有新数据时）
    accuracy_stats = {}
    if len(all_metrics) > 0:
        print(f"\n=== Multi-Scale Patch Metrics Statistics (Newly Processed) ===")
        print(f"Total patches: {len(all_metrics)}")
        print(f"Number of scales: {len(scales)}")

        for scale_idx in range(len(scales)):
            scale_metrics = df[df['scale_idx'] == scale_idx]
            if len(scale_metrics) > 0:
                print(f"\nScale {scale_idx} ({scales[scale_idx]['method']}, "
                      f"{scales[scale_idx]['max_points_per_patch']} pts):")
                print(f"  Patches: {len(scale_metrics)}")
                print(f"  Average patch size: {scale_metrics['patch_size'].mean():.1f} points")
                print(f"  Average accuracy: {scale_metrics['accuracy'].mean():.4f}")
                print(f"  Average mean_gt: {scale_metrics['mean_gt'].mean():.4f}")

                accuracy_stats[f'scale_{scale_idx}'] = {
                    'num_patches': len(scale_metrics),
                    'mean_patch_size': float(scale_metrics['patch_size'].mean()),
                    'mean_accuracy': float(scale_metrics['accuracy'].mean()),
                    'mean_gt': float(scale_metrics['mean_gt'].mean())
                }

    print(f"\nPrecomputation complete! Multi-scale features saved to {cache_dir}")

    return accuracy_stats


def main():
    """
    Example usage:
        python dataset/precompute_patch_features_multiscale.py --config configs/global_flip/t2_multi_scale.yaml --gpu 1 --split train
        python dataset/precompute_patch_features_multiscale.py --config configs/global_flip/SceneNN_multi_scale.yaml --gpu 0 --split train,val
    """
    parser = argparse.ArgumentParser(description='Precompute Multi-Scale Patch Features')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to global flip config file (e.g., configs/global_flip/t2_multi_scale.yaml)')
    parser.add_argument('--split', type=str, default='train',
                        help='Which split(s) to precompute. Use comma-separated values (e.g., "train,val,test" or "train,val")')
    parser.add_argument('--grid_size', type=float, default=0.02,
                        help='Grid size for backbone input')
    parser.add_argument('--pca_max_nn', type=int, default=30,
                        help='Max neighbors for PCA normal estimation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for patch processing')
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU device ID to use')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Optional: directly specify data root directory containing ply files. '
                             'If provided, will override config data paths.')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config = load_and_merge_feat_extraction_config(config)
    config['grid_size'] = args.grid_size
    config['pca_max_nn'] = args.pca_max_nn
    config['batch_size'] = args.batch_size

    torch.cuda.set_device(args.gpu)
    config['device'] = f'cuda:{args.gpu}'

    # Load backbone config for metadata
    backbone_config = load_config(config['backbone']['config'])

    # Parse splits
    splits = [s.strip() for s in args.split.split(',')]

    if args.data_root is not None:
        config['data'] = {
            'root': args.data_root,
            **{split: {'subfolder': split} for split in splits}
        }

    # 收集所有split的统计信息
    all_stats = {
        'splits_processed': splits,
        'num_models': {},
        'total_patches': {},
        'accuracy_stats': {}
    }

    # Precompute for each split
    for split in splits:
        if split not in ['train', 'val', 'test']:
            print(f"Warning: Unknown split '{split}'")
            # continue
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print('='*60)
        accuracy_stats = precompute_features(config, split=split)

        # 如果用户取消了这个split，跳过
        if accuracy_stats is None:
            continue

        # 收集统计信息（简单版，可以从返回值中获取）
        all_stats['accuracy_stats'][split] = accuracy_stats

    # 保存缓存元信息到根目录
    cache_root = get_global_flip_cache_dir(config, backbone_config)
    save_cache_metadata(cache_root, config, backbone_config, all_stats)

    print(f"\n{'='*60}")
    print(f"All splits processed successfully!")
    print(f"Cache root: {cache_root}")
    print('='*60)


if __name__ == '__main__':
    main()
