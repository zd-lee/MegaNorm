import os
import sys
import argparse
import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.multi_scale_patch_dataset import MultiScalePatchDataset
from dataset.patch_extractor import extract_patches_unified
from dataset.dataset import estimate_normals_torch
from dataset.transforms import NormalEstimationNormalize
from models.direct_orientation_model import create_direct_orientation_model
from utils.config import load_config, load_and_merge_feat_extraction_config
from utils.metrics import calculate_metrics_inv
from utils.cache_utils import get_global_flip_cache_dir, save_cache_metadata
from utils.convert_checkpoint import load_model_state_dict


def compute_patch_features_single_scale(
    coords,
    gt_normals,
    patch_list,
    backbone,
    config,
    backbone_config,
    pooling_method="mean",
    use_encoder_features=False,
):
    grid_size = config.get("grid_size", 0.02)
    pca_max_nn = config.get("pca_max_nn", 30)
    patch_iter_config = config.get(
        "patch_iterative", backbone_config.get("iterative", {})
    )
    num_iterations = patch_iter_config.get("num_iterations", 3)
    use_confidence = patch_iter_config.get("use_confidence", True)
    extract_inv_features = config.get("use_inverse_features", False)
    batch_size = config.get("batch_size", 8)
    patch_centers_list = []
    gt_flip_status_list = []
    features_list = []
    inv_features_list = [] if extract_inv_features else None
    pred_normals_list = []
    confidence_list = []
    patch_metrics_list = []
    pca_normals_all = []
    for patch_indices in patch_list:
        patch_coords = coords[patch_indices]
        result = estimate_normals_torch(patch_coords, max_nn=pca_max_nn)
        pca_normals = torch.as_tensor(
            result[:, 3:6], dtype=torch.float32, device=patch_coords.device
        )
        pca_normals_all.append(pca_normals)
    transform = NormalEstimationNormalize()
    num_batches = (len(patch_list) + batch_size - 1) // batch_size
    with torch.no_grad():
        pbar_batches = tqdm(
            range(num_batches), desc="  Refining patches (batched)", leave=False
        )
        for batch_idx in pbar_batches:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(patch_list))
            batch_patches = patch_list[batch_start:batch_end]
            patch_data_list = []
            for i, patch_indices in enumerate(batch_patches):
                patch_coords = coords[patch_indices]
                normals = pca_normals_all[batch_start + i].clone()
                conf = (
                    torch.zeros(len(normals), 1, device=coords.device)
                    if use_confidence
                    else None
                )
                patch_data_list.append([patch_coords, normals, conf])
            transformed_coords_list = []
            offsets_list = []
            for coords_i, _, _ in patch_data_list:
                pd = {
                    "coord": coords_i,
                    "feat": coords_i,
                    "offset": torch.tensor(
                        [len(coords_i)], device=coords.device
                    ).long(),
                    "grid_size": grid_size,
                }
                pd, _ = transform(pd)
                transformed_coords_list.append(pd["coord"])
                offsets_list.append(len(coords_i))
            batch_coords = torch.cat(transformed_coords_list, dim=0)
            batch_offsets = torch.cumsum(
                torch.tensor(offsets_list, device=coords.device), dim=0
            ).long()
            try:
                for iter_idx in range(num_iterations):
                    batch_feats_list = []
                    start_idx = 0
                    for i, (_, normals_i, conf_i) in enumerate(patch_data_list):
                        end_idx = start_idx + len(normals_i)
                        normalized_coords_i = batch_coords[start_idx:end_idx]
                        if use_confidence:
                            feat = torch.cat(
                                [normalized_coords_i, normals_i, conf_i], dim=1
                            )
                        else:
                            feat = torch.cat([normalized_coords_i, normals_i], dim=1)
                        batch_feats_list.append(feat)
                        start_idx = end_idx
                    batch_feats = torch.cat(batch_feats_list, dim=0)
                    logits = backbone(
                        {
                            "coord": batch_coords,
                            "feat": batch_feats,
                            "offset": batch_offsets,
                            "grid_size": grid_size,
                        }
                    )[:, 0]
                    flip_prob = torch.sigmoid(logits)
                    start_idx = 0
                    for i, (coords_i, normals_i, conf_i) in enumerate(patch_data_list):
                        end_idx = start_idx + len(coords_i)
                        flip_mask = flip_prob[start_idx:end_idx] > 0.5
                        normals_i[flip_mask] = -normals_i[flip_mask]
                        if use_confidence:
                            patch_data_list[i][2] = (
                                torch.abs(flip_prob[start_idx:end_idx] - 0.5) * 2
                            ).unsqueeze(1)
                        start_idx = end_idx
            except RuntimeError as e:
                print(
                    f"\n[ERROR] RuntimeError in batch {batch_idx}, iteration {iter_idx}"
                )
                print(f"  Batch range: patches [{batch_start}:{batch_end}]")
                print(f"  Number of patches in batch: {len(batch_patches)}")
                print(f"  Patch sizes: {[len(p) for p in batch_patches]}")
                print(f"  Total points in batch: {len(batch_coords)}")
                print(f"  Batch coords shape: {batch_coords.shape}")
                print(f"  Batch feats shape: {batch_feats.shape}")
                print(f"  Batch offsets: {batch_offsets.tolist()}")
                print(f"  Error message: {str(e)}")
                raise
            for i, (_, normals, conf) in enumerate(patch_data_list):
                patch_idx = batch_start + i
                patch_indices = patch_list[patch_idx]
                patch_gt_normals = gt_normals[patch_indices]
                patch_size = len(patch_indices)
                pred_normals = normals
                pca_normals = pca_normals_all[patch_idx]
                gt_flip_per_point = (
                    (pca_normals * patch_gt_normals).sum(dim=1) < 0
                ).long()
                pred_flip = ((pred_normals * pca_normals).sum(dim=1) < 0).float()
                accuracy, iou, precision, recall, mean_gt = calculate_metrics_inv(
                    pred_flip.cpu(), gt_flip_per_point.cpu().float()
                )
                patch_metrics_list.append(
                    {
                        "patch_idx": patch_idx,
                        "patch_size": patch_size,
                        "accuracy": accuracy,
                        "mean_gt": mean_gt,
                    }
                )
                error_count = ((pred_normals * patch_gt_normals).sum(dim=1) < 0).sum()
                gt_flip_status = error_count > (len(pred_normals) / 2)
                pred_normals_list.append(pred_normals)
                if use_confidence:
                    confidence_list.append(conf)
                gt_flip_status_list.append(gt_flip_status.cpu().numpy())
            if len(patch_metrics_list) > 0:
                avg_acc = np.mean([m["accuracy"] for m in patch_metrics_list])
                pbar_batches.set_postfix(
                    {
                        "avg_acc": f"{100*avg_acc:.1f}%",
                        "batch": f"{batch_end}/{len(patch_list)}",
                    }
                )
    with torch.no_grad():
        pbar_features = tqdm(
            range(num_batches), desc="  Extracting features (batched)", leave=False
        )
        for batch_idx in pbar_features:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(patch_list))
            transformed_data = []
            for i in range(batch_start, batch_end):
                patch_coords = coords[patch_list[i]]
                normals = pred_normals_list[i]
                conf = (
                    confidence_list[i] if use_confidence and confidence_list else None
                )
                feat = (
                    torch.cat([patch_coords, normals, conf], dim=1)
                    if use_confidence and conf is not None
                    else torch.cat([patch_coords, normals], dim=1)
                )
                pd = {
                    "coord": patch_coords,
                    "feat": feat,
                    "offset": torch.tensor(
                        [len(patch_coords)], device=coords.device
                    ).long(),
                    "grid_size": grid_size,
                }
                pd, _ = transform(pd)
                transformed_data.append((pd["coord"], pd["feat"], len(patch_coords)))
            batch_coords = torch.cat([d[0] for d in transformed_data], dim=0)
            batch_feats = torch.cat([d[1] for d in transformed_data], dim=0)
            batch_offsets = torch.cumsum(
                torch.tensor([d[2] for d in transformed_data], device=coords.device),
                dim=0,
            ).long()
            backbone_output = backbone.backbone(
                {
                    "coord": batch_coords,
                    "feat": batch_feats,
                    "offset": batch_offsets,
                    "grid_size": grid_size,
                },
                use_decoder=not use_encoder_features,
            )
            point_features = backbone_output["feat"]
            if extract_inv_features:
                inv_feat = batch_feats.clone()
                inv_feat[:, 3:6] *= -1
                backbone_invoutput = backbone.backbone(
                    {
                        "coord": batch_coords,
                        "feat": inv_feat,
                        "offset": batch_offsets,
                        "grid_size": grid_size,
                    },
                    use_decoder=not use_encoder_features,
                )
                inv_point_features = backbone_invoutput["feat"]
            else:
                inv_point_features = None
            start_idx = 0
            if extract_inv_features:
                assert torch.equal(
                    backbone_output["offset"], backbone_invoutput["offset"]
                )
            for i, end_idx in enumerate(backbone_output["offset"]):
                if pooling_method == "max":
                    patch_feature = point_features[start_idx:end_idx].max(dim=0)[0]
                    if extract_inv_features:
                        inv_patch_feature = inv_point_features[start_idx:end_idx].max(
                            dim=0
                        )[0]
                else:
                    patch_feature = point_features[start_idx:end_idx].mean(dim=0)
                    if extract_inv_features:
                        inv_patch_feature = inv_point_features[start_idx:end_idx].mean(
                            dim=0
                        )
                features_list.append(patch_feature.cpu())
                if extract_inv_features:
                    inv_features_list.append(inv_patch_feature.cpu())
                patch_centers_list.append(
                    coords[patch_list[batch_start + i]].mean(dim=0).cpu()
                )
                start_idx = end_idx
    patch_centers = torch.stack(patch_centers_list).cpu().numpy()
    gt_flip_status = np.array(gt_flip_status_list, dtype=np.int64)
    features = torch.stack(features_list).cpu().numpy()
    inv_features = (
        torch.stack(inv_features_list).cpu().numpy() if extract_inv_features else None
    )
    return patch_centers, gt_flip_status, features, inv_features, patch_metrics_list


def compute_patch_features_multiscale(
    coords,
    gt_normals,
    patches_by_scale,
    backbone,
    config,
    backbone_config,
    pooling_method="mean",
    use_encoder_features=False,
):
    features_by_scale = {}
    for scale_idx, patch_list in patches_by_scale.items():
        print(f"  Computing features for scale {scale_idx}: {len(patch_list)} patches")
        features_by_scale[scale_idx] = compute_patch_features_single_scale(
            coords,
            gt_normals,
            patch_list,
            backbone,
            config,
            backbone_config,
            pooling_method,
            use_encoder_features,
        )
    return features_by_scale


def save_multiscale_features(save_path, features_by_scale, scale_configs, model_name):
    save_dict = {
        "num_scales": len(features_by_scale),
        "scale_configs": json.dumps(scale_configs),
        "model_name": model_name,
    }
    for scale_idx, data in features_by_scale.items():
        patch_centers, gt_flip_status, features, inv_features, _ = data
        prefix = f"scale_{scale_idx}_"
        save_dict[prefix + "patch_centers"] = patch_centers
        save_dict[prefix + "gt_flip_status"] = gt_flip_status
        save_dict[prefix + "features"] = features
        if inv_features is not None:
            save_dict[prefix + "inv_features"] = inv_features
        save_dict[prefix + "num_patches"] = len(features)
        save_dict[prefix + "max_points"] = scale_configs[scale_idx][
            "max_points_per_patch"
        ]
        save_dict[prefix + "method"] = scale_configs[scale_idx]["method"]
    np.savez(save_path, **save_dict)


def get_augmentation_config(config):
    default_config = {
        "enabled": False,
        "rotations_z_deg": [0],
        "downsample_rates": [1.0],
        "source_subfolder": "train",
        "output_subfolder": "train_augmented",
    }
    aug_config = config.get("augmentation", {})
    if aug_config is None:
        aug_config = {}
    result = {**default_config, **aug_config}
    result["rotations_z_deg"] = [
        float(v) for v in result.get("rotations_z_deg", [0])
    ]
    result["downsample_rates"] = [
        float(v) for v in result.get("downsample_rates", [1.0])
    ]
    return result


def rotation_matrix_z(degrees, device):
    theta = np.deg2rad(degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return torch.tensor(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def downsample_points(coords, gt_normals, rate, seed):
    if rate <= 0 or rate > 1:
        raise ValueError(f"Downsample rate must be in (0, 1], got {rate}")
    if rate >= 1.0:
        return coords, gt_normals
    num_points = len(coords)
    keep_count = max(2, int(num_points * rate))
    if keep_count >= num_points:
        return coords, gt_normals
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(num_points, generator=generator)[:keep_count]
    indices, _ = torch.sort(indices)
    indices = indices.to(coords.device)
    return coords[indices], gt_normals[indices]


def rotate_points_z(coords, gt_normals, rotation_deg):
    if abs(rotation_deg) < 1e-8:
        return coords, gt_normals
    rot = rotation_matrix_z(rotation_deg, coords.device)
    return coords @ rot.T, gt_normals @ rot.T


def transform_augmented_points(coords, gt_normals, rotation_deg, downsample_rate, seed):
    aug_coords, aug_normals = downsample_points(
        coords, gt_normals, downsample_rate, seed=seed
    )
    return rotate_points_z(aug_coords, aug_normals, rotation_deg)


def format_augmented_model_stem(model_stem, rotation_deg, downsample_rate):
    rot_token = f"{rotation_deg:g}".replace("-", "m").replace(".", "p")
    ds_token = f"{downsample_rate:g}".replace(".", "p")
    return f"{model_stem}_rz{rot_token}_ds{ds_token}"


def extract_patches_for_coords(coords, scales, device):
    patches_by_scale = {}
    coords_device = coords.to(device)
    for scale_idx, scale_config in enumerate(scales):
        method = scale_config["method"]
        num_per_patch = scale_config["max_points_per_patch"]
        k = scale_config.get("k", 10)
        patch_count = scale_config.get("patch_count", None)
        overlap_rate = scale_config.get("overlap_rate", 0.0)
        if patch_count is None:
            patch_count = int(len(coords) / num_per_patch)
            if patch_count == 0:
                patches_by_scale[scale_idx] = (
                    [torch.arange(len(coords), dtype=torch.long)]
                    if len(coords) > 1
                    else []
                )
                continue
        patch_indices_list, _ = extract_patches_unified(
            coords_device,
            method=method,
            num_per_patch=num_per_patch,
            k=k,
            patch_count=patch_count,
            overlap_rate=overlap_rate,
            device=device,
        )
        patches_by_scale[scale_idx] = [
            indices.cpu() for indices in patch_indices_list if len(indices) > 1
        ]
    return patches_by_scale


def check_cache_status(cache_dir, expected_model_stems, cache_action=None):
    if not os.path.exists(cache_dir):
        if cache_action is not None:
            return set(), cache_action
        return set(), "overwrite"
    existing_files = [f for f in os.listdir(cache_dir) if f.endswith(".npz")]
    existing_models = set([Path(f).stem for f in existing_files])
    if len(existing_models) == 0:
        if cache_action is not None:
            return set(), cache_action
        return set(), "overwrite"
    all_model_names = set(expected_model_stems)
    total_models = len(all_model_names)
    missing_models = all_model_names - existing_models
    print(f"\n{'='*60}")
    print(f"Cache directory already exists: {cache_dir}")
    print(f"{'='*60}")
    print(f"Total models: {total_models}")
    print(
        f"Cached models: {len(existing_models)} ({100*len(existing_models)/total_models:.1f}%)"
    )
    print(
        f"Missing models: {len(missing_models)} ({100*len(missing_models)/total_models:.1f}%)"
    )
    if len(missing_models) > 0 and len(missing_models) <= 10:
        print(f"\nMissing model files:")
        for name in sorted(missing_models):
            print(f"  - {name}")
    elif len(missing_models) > 10:
        print(f"\nMissing model files (showing first 10):")
        for name in sorted(missing_models)[:10]:
            print(f"  - {name}")
        print(f"  ... and {len(missing_models)-10} more")
    if cache_action is not None:
        print(f"\nCache action: {cache_action}")
        if cache_action == "skip" and len(missing_models) == 0:
            print(f"All models are already cached. Nothing to do.")
            return existing_models, "cancel"
        return existing_models, cache_action
    print(f"\nWhat would you like to do?")
    print(
        f"  [o] Overwrite all - Recompute all models (existing cache will be deleted)"
    )
    print(f"  [s] Skip existing - Only compute missing {len(missing_models)} models")
    print(f"  [c] Cancel - Exit without changes")
    while True:
        choice = input(f"\nYour choice [o/s/c]: ").strip().lower()
        if choice in ["o", "overwrite"]:
            print(f"Will overwrite all cached models.")
            return existing_models, "overwrite"
        elif choice in ["s", "skip"]:
            if len(missing_models) == 0:
                print(f"All models are already cached. Nothing to do.")
                return existing_models, "cancel"
            print(
                f"Will skip {len(existing_models)} existing models and compute {len(missing_models)} missing models."
            )
            return existing_models, "skip"
        elif choice in ["c", "cancel"]:
            print(f"Operation cancelled.")
            return existing_models, "cancel"
        else:
            print(f"Invalid choice. Please enter 'o', 's', or 'c'.")


def check_existing_cache(cache_dir, patch_dataset, cache_action=None):
    expected_model_stems = [
        Path(patch_dataset.get_model_name(i)).stem
        for i in range(patch_dataset.get_num_models())
    ]
    return check_cache_status(cache_dir, expected_model_stems, cache_action)


def precompute_features(
    config,
    split="train",
    shard_index=0,
    num_shards=1,
    augment=False,
    cache_action=None,
):
    device = torch.device(config["device"])
    print("Loading backbone...")
    backbone_config_path = config["backbone"]["config"]
    backbone_config = load_config(backbone_config_path)
    backbone = create_direct_orientation_model(backbone_config)
    checkpoint_path = config["backbone"]["checkpoint"]
    backbone.load_state_dict(load_model_state_dict(checkpoint_path))
    backbone = backbone.to(device)
    backbone.eval()
    backbone.freeze_backbone(freeze=True)
    patch_iter_config = config.get(
        "patch_iterative", backbone_config.get("iterative", {})
    )
    print(f"Backbone loaded from {checkpoint_path}")
    print(
        f"Patch iterative config: num_iterations={patch_iter_config.get('num_iterations', 3)}, "
        f"use_confidence={patch_iter_config.get('use_confidence', True)}"
    )
    print(f"Use inverse features: {config.get('use_inverse_features', False)}")
    feature_config = config.get("feature_extraction", {})
    use_encoder_features = feature_config.get("use_encoder_features", False)
    pooling_method = feature_config.get("pooling_method", "mean")
    assert pooling_method in [
        "mean",
        "max",
    ], f"Invalid pooling_method: {pooling_method}. Must be 'mean' or 'max'."
    print(f"\nFeature extraction settings:")
    print(
        f"  - use_encoder_features: {use_encoder_features} ({'encoder only' if use_encoder_features else 'encoder+decoder'})"
    )
    print(f"  - pooling_method: {pooling_method}")
    print(f"  Note: use_decoder parameter will be passed to forward() dynamically")
    aug_config = get_augmentation_config(config)
    augment_split = augment and split == "train"
    if augment and split != "train":
        print(f"Augmentation requested but split is '{split}', using normal precompute.")
    print(f"Preparing dataset for {split}...")
    split_subfolder = config["data"][split]["subfolder"]
    if augment_split:
        split_subfolder = aug_config.get("source_subfolder", "train")
    data_root = os.path.join(config["data"]["root"], split_subfolder)
    scales = config["patch_extraction"]["scales"]
    if augment_split:
        model_files = sorted([f for f in os.listdir(data_root) if f.endswith(".ply")])
        source_patch_dataset = MultiScalePatchDataset(
            data_root=data_root, scales=scales, device=str(device)
        )
        patch_dataset = None
        print(f"Found {len(model_files)} source models for augmentation")
    else:
        patch_dataset = MultiScalePatchDataset(
            data_root=data_root, scales=scales, device=str(device)
        )
        model_files = [
            patch_dataset.get_model_name(i) for i in range(patch_dataset.get_num_models())
        ]
        print(
            f"Found {patch_dataset.get_num_models()} models with {len(patch_dataset)} patches total"
        )
    print(f"Scales: {len(scales)}")
    for i, scale in enumerate(scales):
        print(f"  Scale {i}: {scale}")
    cache_root = get_global_flip_cache_dir(config, backbone_config)
    cache_subdir = (
        aug_config.get("output_subfolder", "train_augmented")
        if augment_split
        else split
    )
    cache_dir = os.path.join(cache_root, cache_subdir)
    if augment_split:
        rotations = aug_config["rotations_z_deg"]
        downsample_rates = aug_config["downsample_rates"]
        expected_model_stems = []
        for model_idx in range(len(model_files)):
            model_stem = Path(model_files[model_idx]).stem
            for rotation_deg in rotations:
                for downsample_rate in downsample_rates:
                    expected_model_stems.append(
                        format_augmented_model_stem(
                            model_stem, rotation_deg, downsample_rate
                        )
                    )
        existing_models, action = check_cache_status(
            cache_dir, expected_model_stems, cache_action=cache_action
        )
        print("\nAugmentation settings:")
        print(f"  - output_subfolder: {cache_subdir}")
        print(f"  - rotations_z_deg: {rotations}")
        print(f"  - downsample_rates: {downsample_rates}")
        print(f"  - variants_per_model: {len(rotations) * len(downsample_rates)}")
    else:
        existing_models, action = check_existing_cache(
            cache_dir, patch_dataset, cache_action=cache_action
        )
    if action == "cancel":
        print(f"Skipping {split} split.")
        return None
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\nCache root: {cache_root}")
    print(f"Cache directory for {split}: {cache_dir}")
    print(f"\nPrecomputing multi-scale features for {split} split...")
    all_metrics = []
    num_processed = 0
    num_skipped = 0
    num_models = len(model_files)
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    model_indices = [
        model_idx
        for model_idx in range(num_models)
        if model_idx % num_shards == shard_index
    ]
    if num_shards > 1:
        print(
            f"Shard {shard_index}/{num_shards}: processing "
            f"{len(model_indices)} of {num_models} models"
        )
    pbar = tqdm(model_indices)
    for model_idx in pbar:
        model_name = model_files[model_idx]
        model_stem = Path(model_name).stem
        if augment_split:
            model_data = source_patch_dataset.get_model_patches_all_scales(model_idx)
            coords_cpu = model_data["coords"]
            gt_normals_cpu = model_data["gt_normals"]
        else:
            model_data = patch_dataset.get_model_patches_all_scales(model_idx)
            coords_cpu = model_data["coords"]
            gt_normals_cpu = model_data["gt_normals"]
        if augment_split:
            model_variants = []
            variant_idx = 0
            downsample_cache = {}
            for rotation_deg in rotations:
                for downsample_rate in downsample_rates:
                    variant_stem = format_augmented_model_stem(
                        model_stem, rotation_deg, downsample_rate
                    )
                    model_variants.append(
                        (variant_idx, variant_stem, rotation_deg, downsample_rate)
                    )
                    variant_idx += 1
        else:
            model_variants = [(0, model_stem, 0.0, 1.0)]
        recent_model_metrics = []
        for variant_idx, save_stem, rotation_deg, downsample_rate in model_variants:
            save_path = os.path.join(cache_dir, save_stem + ".npz")
            if action == "skip" and save_stem in existing_models:
                num_skipped += 1
                pbar.set_postfix(
                    {
                        "processed": num_processed,
                        "skipped": num_skipped,
                        "status": "skip",
                    }
                )
                continue
            if augment_split:
                downsample_key = f"{downsample_rate:g}"
                if downsample_key not in downsample_cache:
                    if downsample_rate >= 1.0:
                        ds_coords_cpu = coords_cpu
                        ds_gt_normals_cpu = gt_normals_cpu
                        patches_by_scale_for_ds = (
                            source_patch_dataset.get_model_patches_all_scales(
                                model_idx
                            )["patches_by_scale"]
                        )
                    else:
                        seed = (
                            config.get("experiment", {}).get("seed", 42)
                            + model_idx * 1009
                            + int(round(downsample_rate * 1000))
                        )
                        ds_coords_cpu, ds_gt_normals_cpu = downsample_points(
                            coords_cpu, gt_normals_cpu, downsample_rate, seed
                        )
                        patches_by_scale_for_ds = extract_patches_for_coords(
                            ds_coords_cpu, scales, str(device)
                        )
                    downsample_cache[downsample_key] = (
                        ds_coords_cpu,
                        ds_gt_normals_cpu,
                        patches_by_scale_for_ds,
                    )
                ds_coords_cpu, ds_gt_normals_cpu, patches_by_scale = downsample_cache[
                    downsample_key
                ]
                coords_aug_cpu, gt_normals_aug_cpu = rotate_points_z(
                    ds_coords_cpu, ds_gt_normals_cpu, rotation_deg
                )
                coords = coords_aug_cpu.to(device)
                gt_normals = gt_normals_aug_cpu.to(device)
                variant_model_name = f"{model_stem}.ply"
            else:
                coords = coords_cpu.to(device)
                gt_normals = gt_normals_cpu.to(device)
                patches_by_scale = model_data["patches_by_scale"]
                variant_model_name = model_name
            features_by_scale = compute_patch_features_multiscale(
                coords,
                gt_normals,
                patches_by_scale,
                backbone,
                config,
                backbone_config,
                pooling_method,
                use_encoder_features,
            )
            save_multiscale_features(
                save_path, features_by_scale, scales, variant_model_name
            )
            num_processed += 1
            for scale_idx, (_, _, _, _, patch_metrics) in features_by_scale.items():
                for metrics_dict in patch_metrics:
                    metrics_dict["model_name"] = variant_model_name
                    metrics_dict["cache_stem"] = save_stem
                    metrics_dict["model_idx"] = model_idx
                    metrics_dict["scale_idx"] = scale_idx
                    if augment_split:
                        metrics_dict["rotation_z_deg"] = rotation_deg
                        metrics_dict["downsample_rate"] = downsample_rate
                    all_metrics.append(metrics_dict)
                    recent_model_metrics.append(metrics_dict)
        if len(recent_model_metrics) > 0:
            avg_acc = np.mean([m["accuracy"] for m in recent_model_metrics])
        else:
            avg_acc = 0.0
        pbar.set_postfix(
            {
                "processed": num_processed,
                "skipped": num_skipped,
                "acc": f"{100*avg_acc:.1f}%",
            }
        )
    print(f"\n{'='*60}")
    print(f"Processing summary for {split}:")
    print(f"  Processed: {num_processed} models")
    print(f"  Skipped: {num_skipped} models")
    print(f"  Total: {len(model_indices)} models in this run")
    if num_shards > 1:
        print(f"  Full split total: {num_models} models")
    print(f"{'='*60}")
    if len(all_metrics) > 0:
        csv_name = (
            f"patch_metrics_shard{shard_index:02d}.csv"
            if num_shards > 1
            else "patch_metrics.csv"
        )
        csv_path = os.path.join(cache_dir, csv_name)
        df = pd.DataFrame(all_metrics)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to {csv_path}")
    else:
        df = pd.DataFrame()
        print(f"\nNo new models processed, skipping metrics CSV.")
    accuracy_stats = {}
    if len(all_metrics) > 0:
        print(f"\n=== Multi-Scale Patch Metrics Statistics (Newly Processed) ===")
        print(f"Total patches: {len(all_metrics)}")
        print(f"Number of scales: {len(scales)}")
        for scale_idx in range(len(scales)):
            scale_metrics = df[df["scale_idx"] == scale_idx]
            if len(scale_metrics) > 0:
                print(
                    f"\nScale {scale_idx} ({scales[scale_idx]['method']}, "
                    f"{scales[scale_idx]['max_points_per_patch']} pts):"
                )
                print(f"  Patches: {len(scale_metrics)}")
                print(
                    f"  Average patch size: {scale_metrics['patch_size'].mean():.1f} points"
                )
                print(f"  Average accuracy: {scale_metrics['accuracy'].mean():.4f}")
                print(f"  Average mean_gt: {scale_metrics['mean_gt'].mean():.4f}")
                accuracy_stats[f"scale_{scale_idx}"] = {
                    "num_patches": len(scale_metrics),
                    "mean_patch_size": float(scale_metrics["patch_size"].mean()),
                    "mean_accuracy": float(scale_metrics["accuracy"].mean()),
                    "mean_gt": float(scale_metrics["mean_gt"].mean()),
                }
    print(f"\nPrecomputation complete! Multi-scale features saved to {cache_dir}")
    return accuracy_stats


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Multi-Scale Patch Features"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to global flip config file (e.g., configs/global_flip/t2_multi_scale.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help='Which split(s) to precompute. Use comma-separated values (e.g., "train,val,test" or "train,val")',
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.02, help="Grid size for backbone input"
    )
    parser.add_argument(
        "--pca_max_nn",
        type=int,
        default=30,
        help="Max neighbors for PCA normal estimation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for patch processing"
    )
    parser.add_argument("--gpu", type=int, default=3, help="GPU device ID to use")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="For train split, precompute deterministic augmented variants into the configured augmented cache subfolder.",
    )
    parser.add_argument(
        "--cache_action",
        type=str,
        choices=["overwrite", "skip", "cancel"],
        default=None,
        help="Optional non-interactive cache action. Use 'skip' for sharded/multi-GPU runs.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of disjoint model-index shards to split this run into",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index for this run; processes model_idx %% num_shards == shard_index",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional: directly specify data root directory containing ply files. "
        "If provided, will override config data paths.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = load_and_merge_feat_extraction_config(config)
    config["grid_size"] = args.grid_size
    config["pca_max_nn"] = args.pca_max_nn
    config["batch_size"] = args.batch_size
    torch.cuda.set_device(args.gpu)
    config["device"] = f"cuda:{args.gpu}"
    backbone_config = load_config(config["backbone"]["config"])
    splits = [s.strip() for s in args.split.split(",")]
    if args.data_root is not None:
        config["data"] = {
            "root": args.data_root,
            **{split: {"subfolder": split} for split in splits},
        }
    all_stats = {
        "splits_processed": splits,
        "num_models": {},
        "total_patches": {},
        "accuracy_stats": {},
    }
    for split in splits:
        if split not in ["train", "val", "test"]:
            print(f"Warning: Unknown split '{split}'")
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print("=" * 60)
        accuracy_stats = precompute_features(
            config,
            split=split,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            augment=args.augment,
            cache_action=args.cache_action,
        )
        if accuracy_stats is None:
            continue
        all_stats["accuracy_stats"][split] = accuracy_stats
    cache_root = get_global_flip_cache_dir(config, backbone_config)
    if args.num_shards == 1 or args.shard_index == 0:
        save_cache_metadata(cache_root, config, backbone_config, all_stats)
    else:
        print(
            f"Skipping shared cache metadata write for shard {args.shard_index}; "
            "shard 0 owns metadata."
        )
    print(f"\n{'='*60}")
    print(f"All splits processed successfully!")
    print(f"Cache root: {cache_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
