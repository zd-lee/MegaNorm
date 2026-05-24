#!/usr/bin/env python
"""
Direct Orientation Prediction Training Script
直接方向预测训练脚本 - 使用PTv3 + MLP进行二分类
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
import os
os.environ["TIMM_DISABLE_HF_HUB"] = "1"

from models.direct_orientation_model import create_direct_orientation_model
from dataset.multi_scale_patch_dataset import MultiScalePatchDataset
from dataset.dataset import estimate_normals_torch
from dataset.transforms import get_normal_estimation_train_transforms
from dataset.collate_fn import NormalEstimationCollator
from utils.experiment_manager import ExperimentManager
from utils.loss import create_loss_function
from utils.config import load_config
from utils.scheduler import create_scheduler
from utils.metrics import calculate_metrics_inv

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


def compute_pca_normals(coords, offset, pca_max_nn, device):
    """Compute PCA normals for each sample in batch"""
    pred_normals_list = []
    start_idx = 0

    for end_idx in offset:
        sample_coords = coords[start_idx:end_idx].cpu()
        result = estimate_normals_torch(sample_coords, max_nn=pca_max_nn)
        pred_normal = torch.from_numpy(result[:, 3:6]).float().to(device)
        pred_normals_list.append(pred_normal)
        start_idx = end_idx

    return torch.cat(pred_normals_list, dim=0)


def compute_flip_gt(current_normals, gt_normals):
    """Compute GT flip status: 1 if angle > 90 degrees (dot < 0)"""
    dot_products = (current_normals * gt_normals).sum(dim=1)
    return (dot_products < 0).long()


def create_soft_labels_mixup(current_normals, gt_normals, hard_labels):
    """Create soft labels using mixup strategy"""
    dot_prod = torch.sum(current_normals * gt_normals, dim=1).clamp(-1, 1).abs()
    soft_labels = hard_labels.clone().float()
    soft_labels[hard_labels.bool()] = 0.5 + dot_prod[hard_labels.bool()] / 2
    soft_labels[~hard_labels.bool()] = 0.5 - dot_prod[~hard_labels.bool()] / 2
    return soft_labels


def create_data_loaders(config, include_test=False):
    """Create multi-scale patch dataloaders"""
    base_root = config['data']['root']
    collate_fn = NormalEstimationCollator()

    # Get scale configurations
    scales = config['patch_extraction']['scales']

    loaders = {}
    splits = ['train', 'val']
    if include_test:
        splits.append('test')

    for split in splits:
        split_config = config['data'][split]
        data_root = os.path.join(base_root, split_config['subfolder'])

        # Create transforms based on split-specific augmentation config
        transform_list = []

        # Add random rotation if enabled in augmentation config
        aug_config = split_config.get('augmentation', {})
        rot_config = aug_config.get('random_rotation', {})
        if rot_config.get('enabled', False):
            from dataset.transforms import RandomRotation, NormalEstimationCompose
            max_angle = rot_config.get('max_angle', 15.0)
            axes = rot_config.get('axes', 'xyz')
            transform_list.append(RandomRotation(max_angle=max_angle, axes=axes))

        # Always add normalization
        from dataset.transforms import NormalEstimationNormalize, NormalEstimationCompose
        transform_list.append(NormalEstimationNormalize(method='unit_sphere', center=True))

        # Compose transforms
        transforms = NormalEstimationCompose(transform_list)

        # Use MultiScalePatchDataset
        dataset = MultiScalePatchDataset(
            data_root=data_root,
            scales=scales,
            device=config.get('device', 'cuda'),
            grid_size=split_config['grid_size'],  # Pass grid_size for PTv3
            transform=transforms  # Pass transforms for normalization
        )

        print(f"{split} dataset: {dataset.get_num_models()} models, "
              f"{len(dataset)} patches, {dataset.get_num_scales()} scales")

        loaders[split] = DataLoader(
            dataset,
            batch_size=split_config['batch_size'],
            shuffle=split_config.get('shuffle', split == 'train'),
            num_workers=split_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )

    if include_test:
        return loaders['train'], loaders['val'], loaders['test']
    return loaders['train'], loaders['val']

def run_epoch(epoch, total_epochs, data_loader, model, criterion, optimizer,
              device, writer, config, is_train=True):
    """Run one epoch (train or val), returns avg_metrics dict"""
    phase = 'Train' if is_train else 'Val'
    pca_max_nn = config['data'][phase.lower()].get('pca_max_nn', 30)
    num_iterations = config.get('iterative', {}).get('num_iterations', 3)
    use_confidence = config.get('iterative', {}).get('use_confidence', True)
    use_mixup = config.get('training', {}).get('mixup', {}).get('enabled', False)

    model.train() if is_train else model.eval()

    epoch_metrics = {k: 0.0 for k in ['Loss', 'Accuracy', 'IoU', 'Precision', 'Recall', 'MeanGT']}
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_loader, desc=f'{phase} {epoch}/{total_epochs}')
        for batch_idx, (point_data, gt_normal) in enumerate(pbar):

            # Move to device
            for k in point_data:
                if isinstance(point_data[k], torch.Tensor):
                    point_data[k] = point_data[k].to(device)
            gt_normal = gt_normal.to(device)

            # Compute PCA normals (once per batch)
            coords = point_data['coord']
            offset = point_data['batch_offsets']
            pca_normals = compute_pca_normals(coords, offset, pca_max_nn, device)

            # Initialize iterative state
            x_old = pca_normals.clone()  # (N, 3)
            if use_confidence:
                conf_old = torch.zeros(x_old.shape[0], 1, device=device)  # (N, 1)

            # Iterative refinement loop with per-iteration logging
            iter_losses = []
            iter_accs = []

            fail = False
            for _ in range(num_iterations):
                # Compute GT flip status for current normals
                gt_flip_status = compute_flip_gt(x_old, gt_normal)

                # Update features: xyz + normals (+ confidence if enabled)
                if use_confidence:
                    point_data['feat'] = torch.cat([coords, x_old, conf_old], dim=1)  # (N, 7)
                else:
                    point_data['feat'] = torch.cat([coords, x_old], dim=1)  # (N, 6)

                # Forward pass
                try:
                    logits = model(point_data)[:, 0]  # (N,)
                except Exception as e:
                    print(f"Exception at batch {batch_idx}, iteration {_}: {e}")
                    logits = torch.zeros(x_old.shape[0], device=device)
                    fail = True
                    break

                # Loss
                if use_mixup:
                    soft_labels = create_soft_labels_mixup(x_old, gt_normal, gt_flip_status)
                    loss, _ = criterion(logits, None, soft_labels, offset=offset)
                else:
                    loss, _ = criterion(logits, None, gt_flip_status.float(), offset=offset)

                # Record per-iteration metrics
                iter_losses.append(loss.item())
                with torch.no_grad():
                    iter_acc,_,_,_,_ = calculate_metrics_inv(logits, gt_flip_status.float(), offset)
                iter_accs.append(iter_acc)

                # Backward pass (train only)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_norm'])
                    optimizer.step()

                # Apply flip operation
                flip_prob = torch.sigmoid(logits)  # (N,)
                flip_mask = flip_prob > 0.5
                x_new = x_old.clone()
                x_new[flip_mask] = -x_new[flip_mask]

                # Update confidence (if enabled)
                if use_confidence:
                    conf_new = (torch.abs(flip_prob - 0.5) * 2).unsqueeze(1)  # (N, 1)

                # Update for next iteration
                x_old = x_new.detach()
                if use_confidence:
                    conf_old = conf_new.detach()

            if fail:
                continue
            # Calculate final metrics
            accuracy, iou, precision, recall, mean_gt = calculate_metrics_inv(
                logits, gt_flip_status.float(), offset
            )

            # Update epoch statistics
            epoch_metrics['Loss'] += loss.item()
            epoch_metrics['Accuracy'] += accuracy
            epoch_metrics['IoU'] += iou
            epoch_metrics['Precision'] += precision
            epoch_metrics['Recall'] += recall
            epoch_metrics['MeanGT'] += mean_gt
            num_batches += 1

            # Progress bar: show iteration improvement
            pbar.set_postfix({
                'loss': f'{iter_losses[-1]:.3f}',
                'acc': f'{100*iter_accs[0]:.0f}→{100*iter_accs[-1]:.0f}%'
            })

    # Epoch averages
    if num_batches == 0:
        # All batches were skipped
        avg_metrics = {k: 0.0 for k in epoch_metrics.keys()}
    else:
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

    # Log epoch averages
    if writer:
        for name, value in avg_metrics.items():
            writer.add_scalar(f'{phase}_Epoch/{name}', value, epoch)

    return avg_metrics


def run_test_mode(model, test_loader, criterion, device, experiment_manager, config):
    """Run evaluation on test set and save results"""
    logger = experiment_manager.logger
    logger.info("="*80)
    logger.info("Running in TEST mode...")
    logger.info("="*80)
    test_metrics = run_epoch(
        epoch=0, total_epochs=1, data_loader=test_loader,
        model=model, criterion=criterion, optimizer=None,
        device=device, writer=experiment_manager.writer,
        config=config, is_train=False
    )
    logger.info("="*80)
    logger.info("TEST RESULTS:")
    logger.info("="*80)
    logger.info(f"  Loss:           {test_metrics['Loss']:.4f}")
    logger.info(f"  Accuracy:       {100*test_metrics['Accuracy']:.2f}%")
    logger.info(f"  IoU:            {test_metrics['IoU']:.4f}")
    logger.info(f"  Precision:      {test_metrics['Precision']:.4f}")
    logger.info(f"  Recall:         {test_metrics['Recall']:.4f}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train Direct Orientation Prediction Model')
    parser.add_argument('--config', type=str, default='configs/direct_orientation/iterative_multiscale.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (skip training, only evaluate on test set)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])

    # Device configuration
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if config['device'] == 'cuda':
        torch.cuda.set_device(args.gpu)

    # Initialize experiment manager (handles logging)
    experiment_name = Path(args.config).stem + config['experiment']['name']
    experiment_manager = ExperimentManager(config, experiment_name=experiment_name)
    logger = experiment_manager.logger

    # Create model
    model = create_direct_orientation_model(config)
    model = model.to(device)

    # Multi-GPU support (optional)
    if config.get('distributed', False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=config['gpu_ids'])

    # Create data loaders
    if args.test:
        train_loader, val_loader, test_loader = create_data_loaders(config, include_test=True)
    else:
        train_loader, val_loader = create_data_loaders(config, include_test=False)

    # Loss function
    criterion = create_loss_function(config)

    # Optimizer
    if config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']))

    # Learning rate scheduler
    total_epochs = config['training']['epochs']
    scheduler = create_scheduler(optimizer, config['training'], total_epochs)

    # Resume from checkpoint if specified
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training tracking
    best_val = 0.0
    val_metrics = None  # Initialize to handle non-eval epochs

    # Test mode: run evaluation and exit
    if args.test:
        if not args.resume:
            logger.error("Test mode requires a checkpoint. Please provide --resume <checkpoint_path>")
            experiment_manager.cleanup()
            return

        run_test_mode(model, test_loader, criterion, device, experiment_manager, config)
        experiment_manager.cleanup()
        logger.info("Test completed!")
        return

    # Training loop
    for epoch in range(start_epoch, total_epochs):

        # Train
        train_metrics = run_epoch(
            epoch=epoch + 1, total_epochs=total_epochs, data_loader=train_loader,
            model=model, criterion=criterion, optimizer=optimizer,
            device=device, writer=experiment_manager.writer, config=config, is_train=True
        )

        # Validate (only on eval_freq epochs)
        if (epoch + 1) % config["training"]["eval_freq"] == 0:
            val_metrics = run_epoch(
                epoch=epoch + 1, total_epochs=total_epochs, data_loader=val_loader,
                model=model, criterion=criterion, optimizer=None,
                device=device, writer=experiment_manager.writer, config=config, is_train=False
            )
        else:
            val_metrics = None
        # Step scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch+1}: LR = {current_lr:.2e}")

            # Log LR to TensorBoard
            if experiment_manager.writer:
                experiment_manager.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)

        # Always save latest checkpoint after every epoch
        save_metrics = val_metrics if val_metrics is not None else train_metrics
        experiment_manager.save_checkpoint(
            epoch=epoch + 1, model=model, optimizer=optimizer,
            metrics=save_metrics, is_best=False, scheduler=scheduler
        )

        # Save best model based on validation accuracy (only when val_metrics available)
        if val_metrics is not None and val_metrics['Accuracy'] > best_val:
            best_val = val_metrics['Accuracy']
            experiment_manager.save_checkpoint(
                epoch=epoch + 1, model=model, optimizer=optimizer,
                metrics=val_metrics, is_best=True, scheduler=scheduler
            )
            logger.info(f"Saved best model: Accuracy={best_val:.4f}")

    # Save the last epoch checkpoint
    logger.info(f"Saving final epoch checkpoint: epoch_{total_epochs}")
    final_metrics = val_metrics if val_metrics is not None else train_metrics
    experiment_manager.save_checkpoint(
        epoch=total_epochs, model=model, optimizer=optimizer,
        metrics=final_metrics, is_best=False, scheduler=scheduler
    )

    # Training completed
    experiment_manager.cleanup()
    logger.info("Training completed!")

if __name__ == '__main__':
    main()
