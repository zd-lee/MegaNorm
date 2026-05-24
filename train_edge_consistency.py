#!/usr/bin/env python
"""
Edge Consistency Prediction Training Script
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
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataset.edge_consistency_dataset import EdgeConsistencyDataset, edge_consistency_collate
from models.edge_consistency_mlp import create_edge_consistency_mlp
from utils.experiment_manager import ExperimentManager
from utils.config import load_config, load_and_merge_feat_extraction_config
from utils.scheduler import create_scheduler
from utils.metrics import calculate_metrics
from utils.loss import create_loss_function
from utils.cache_utils import get_global_flip_cache_dir, load_and_verify_cache_metadata


def run_epoch(epoch, total_epochs, data_loader, model, criterion, optimizer,
              device, writer, config, is_train=True):
    phase = 'Train' if is_train else 'Val'
    model.train() if is_train else model.eval()

    epoch_metrics = {'loss': 0.0, 'accuracy': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_loader, desc=f'{phase} {epoch}/{total_epochs}')
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            feat_A = batch['features_A'].to(device)
            feat_B = batch['features_B'].to(device)
            labels = batch['edge_labels'].to(device)
            center_A = batch['patch_centers_A'].to(device)
            center_B = batch['patch_centers_B'].to(device)
            batch_offsets = batch['batch_offsets']

            logits = model(feat_A, feat_B, center_A, center_B, batch_offsets)

            loss, _ = criterion(logits.squeeze(), None, labels, offset=batch_offsets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip_norm'])
                optimizer.step()

            accuracy, iou, precision, recall, _ = calculate_metrics(
                logits.squeeze(), labels, batch_offsets
            )

            epoch_metrics['loss'] += loss.item()
            epoch_metrics['accuracy'] += accuracy
            epoch_metrics['iou'] += iou
            epoch_metrics['precision'] += precision
            epoch_metrics['recall'] += recall
            num_batches += 1

            if writer:
                global_step = (epoch - 1) * len(data_loader) + batch_idx
                writer.add_scalar(f'{phase}/Loss', loss.item(), global_step)
                writer.add_scalar(f'{phase}/Accuracy', accuracy, global_step)

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100*accuracy:.1f}%'
            })

    avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

    if writer:
        for name, value in avg_metrics.items():
            writer.add_scalar(f'{phase}_Epoch/{name}', value, epoch)

    return avg_metrics


def create_data_loaders(config, include_test=False):
    # 合并特征提取配置
    config = load_and_merge_feat_extraction_config(config)

    backbone_config = load_config(config['backbone']['config'])
    cache_root = get_global_flip_cache_dir(config, backbone_config)

    print(f"\\nLoading cache from: {cache_root}")
    is_valid, metadata = load_and_verify_cache_metadata(
        cache_root, config, backbone_config, verbose=True
    )

    if not is_valid:
        config_path = 'YOUR_CONFIG.yaml'
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                break
        raise ValueError(
            f"Cache validation failed! Please regenerate cache with (P, 2, 512) features.\\n"
            f"Ensure precompute saves both pos/neg features."
        )

    loaders = {}
    splits = ['train', 'val']
    if include_test:
        splits.append('test')

    k = config['graph'].get('k', 10)
    use_inv_features = config.get('m2_iterative', {}).get('enabled', True)

    for split in splits:
        split_config = config['data'][split]

        dataset = EdgeConsistencyDataset(
            cache_root=cache_root,
            cache_subdir=split_config['subfolder'],
            k=k,
            use_inv_features=use_inv_features
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=split_config['batch_size'],
            shuffle=split_config.get('shuffle', split == 'train'),
            num_workers=split_config['num_workers'],
            collate_fn=edge_consistency_collate,
            pin_memory=True
        )

    print(f"\\nDatasets loaded successfully!")
    for split in loaders:
        print(f"  {split}: {len(loaders[split].dataset)} samples")

    if include_test:
        return loaders['train'], loaders['val'], loaders['test']
    return loaders['train'], loaders['val']


def run_test_mode(model, test_loader, criterion, device, experiment_manager, config):
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
    logger.info(f"  Loss:           {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy:       {100*test_metrics['accuracy']:.2f}%")
    logger.info(f"  IoU:            {test_metrics['iou']:.4f}")
    logger.info(f"  Precision:      {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:         {test_metrics['recall']:.4f}")
    logger.info("="*80)


def main():
    """
    Train Edge Consistency MLP

    Example usage:
        python train_edge_consistency.py --config configs/edge_consistency/scannet_edge.yaml --gpu 0
    """
    parser = argparse.ArgumentParser(description='Train Edge Consistency MLP')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode')
    args = parser.parse_args()

    config = load_config(args.config)

    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    if config['device'] == 'cuda':
        torch.cuda.set_device(args.gpu)

    experiment_name = Path(args.config).stem + '_' + config['experiment']['name']
    experiment_manager = ExperimentManager(config, experiment_name=experiment_name)
    logger = experiment_manager.logger

    model = create_edge_consistency_mlp(config)
    model = model.to(device)

    if config.get('distributed', False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=config.get('gpu_ids', [0]))

    if args.test:
        train_loader, val_loader, test_loader = create_data_loaders(config, include_test=True)
    else:
        train_loader, val_loader = create_data_loaders(config, include_test=False)

    criterion = create_loss_function(config)

    if config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']))

    total_epochs = config['training']['epochs']
    scheduler = create_scheduler(optimizer, config['training'], total_epochs)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    if args.test:
        if not args.resume:
            logger.error("Test mode requires a checkpoint. Please provide --resume <checkpoint_path>")
            experiment_manager.cleanup()
            return

        run_test_mode(model, test_loader, criterion, device, experiment_manager, config)
        experiment_manager.cleanup()
        logger.info("Test completed!")
        return

    best_val_iou = 0.0

    for epoch in range(start_epoch, total_epochs):

        train_metrics = run_epoch(
            epoch=epoch + 1, total_epochs=total_epochs, data_loader=train_loader,
            model=model, criterion=criterion, optimizer=optimizer,
            device=device, writer=experiment_manager.writer, config=config, is_train=True
        )

        val_metrics = run_epoch(
            epoch=epoch + 1, total_epochs=total_epochs, data_loader=val_loader,
            model=model, criterion=criterion, optimizer=None,
            device=device, writer=experiment_manager.writer, config=config, is_train=False
        )

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch+1}: LR = {current_lr:.2e}")

            if experiment_manager.writer:
                experiment_manager.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)

        if (epoch + 1) % config['training']['save_freq'] == 0:
            experiment_manager.save_checkpoint(
                epoch=epoch + 1, model=model, optimizer=optimizer,
                metrics=val_metrics, is_best=False, scheduler=scheduler
            )
            logger.info(f"Saved checkpoint: epoch_{epoch+1:03d}.pth")

        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            experiment_manager.save_checkpoint(
                epoch=epoch + 1, model=model, optimizer=optimizer,
                metrics=val_metrics, is_best=True, scheduler=scheduler
            )
            logger.info(f"Saved best model: IoU={best_val_iou:.4f}")

    logger.info(f"Saving final epoch checkpoint: epoch_{total_epochs}")
    experiment_manager.save_checkpoint(
        epoch=total_epochs, model=model, optimizer=optimizer,
        metrics=val_metrics, is_best=False, scheduler=scheduler
    )

    experiment_manager.cleanup()
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
