import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import Dict, Any


from models.direct_orientation_model import create_direct_orientation_model
from dataset.dataset import estimate_normals_torch
from utils.metrics import calculate_metrics_inv
from utils.loss import create_loss_function


class DirectOrientationModule(LightningModule):
    def __init__(
        self,
        backbone: dict,
        mlp_head: dict,
        iterative: dict,
        loss_params: dict,
        grad_clip_norm: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Reconstruct the config for legacy factory functions
        # Map 'loss_type' back to 'type' for create_loss_function
        if "loss_type" in loss_params:
            loss_params["type"] = loss_params.pop("loss_type")

        full_config = {
            "backbone": backbone,
            "mlp_head": mlp_head,
            "iterative": iterative,
            "training": {"loss": loss_params, "grad_clip_norm": grad_clip_norm},
        }
        self.model = create_direct_orientation_model(full_config)
        self.criterion = create_loss_function(full_config)

        self.num_iterations = iterative.get("num_iterations", 3)
        self.use_confidence = iterative.get("use_confidence", True)
        self.use_mixup = iterative.get("use_mixup", False)
        self.grad_clip_norm = grad_clip_norm

        # Manual optimization because parameters are updated inside the iterative loop
        self.automatic_optimization = False

    def forward(self, point_data):
        return self.model(point_data)

    def _compute_pca_normals(self, coords, offset, pca_max_nn):
        """Compute PCA normals for each sample in batch"""
        pred_normals_list = []
        start_idx = 0

        for end_idx in offset:
            sample_coords = coords[start_idx:end_idx].cpu()
            result = estimate_normals_torch(sample_coords, max_nn=pca_max_nn)
            pred_normal = torch.from_numpy(result[:, 3:6]).float().to(self.device)
            pred_normals_list.append(pred_normal)
            start_idx = end_idx
        
        return torch.cat(pred_normals_list, dim=0)

    def _compute_flip_gt(self, current_normals, gt_normals):
        """Compute GT flip status: 1 if angle > 90 degrees (dot < 0)"""
        dot_products = (current_normals * gt_normals).sum(dim=1)
        return (dot_products < 0).long()

    def _create_soft_labels_mixup(self, current_normals, gt_normals, hard_labels):
        """Create soft labels using mixup strategy"""
        dot_prod = torch.sum(current_normals * gt_normals, dim=1).clamp(-1, 1).abs()
        soft_labels = hard_labels.clone().float()
        soft_labels[hard_labels.bool()] = 0.5 + dot_prod[hard_labels.bool()] / 2
        soft_labels[~hard_labels.bool()] = 0.5 - dot_prod[~hard_labels.bool()] / 2
        return soft_labels

    def _iterative_step(self, batch, batch_idx, pca_max_nn, is_train=True):
        point_data, gt_normal = batch
        coords = point_data["coord"]
        offset = point_data["batch_offsets"]

        # Compute PCA normals
        pca_normals = self._compute_pca_normals(coords, offset, pca_max_nn)

        x_old = pca_normals.clone()
        if self.use_confidence:
            conf_old = torch.zeros(x_old.shape[0], 1, device=self.device)

        optimizer = self.optimizers() if is_train else None

        last_loss = None
        last_logits = None
        last_gt_flip_status = None

        for i in range(self.num_iterations):
            gt_flip_status = self._compute_flip_gt(x_old, gt_normal)

            if self.use_confidence:
                point_data["feat"] = torch.cat([coords, x_old, conf_old], dim=1)
            else:
                point_data["feat"] = torch.cat([coords, x_old], dim=1)

            logits = self.forward(point_data)[:, 0]
            if self.use_mixup:
                soft_labels = self._create_soft_labels_mixup(x_old, gt_normal, gt_flip_status)
                loss, _ = self.criterion(logits, None, soft_labels, offset=offset)
            else:
                loss, _ = self.criterion(logits, None, gt_flip_status.float(), offset=offset)

            if is_train:
                optimizer.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=self.grad_clip_norm,
                    gradient_clip_algorithm="norm",
                )
                optimizer.step()

            # Apply flip
            flip_prob = torch.sigmoid(logits)
            flip_mask = flip_prob > 0.5
            x_new = x_old.clone()
            x_new[flip_mask] = -x_new[flip_mask]

            if self.use_confidence:
                conf_new = (torch.abs(flip_prob - 0.5) * 2).unsqueeze(1)

            x_old = x_new.detach()
            if self.use_confidence:
                conf_old = conf_new.detach()

            last_loss = loss
            last_logits = logits
            last_gt_flip_status = gt_flip_status

        # Calculate metrics on the last iteration
        metrics = calculate_metrics_inv(
            last_logits, last_gt_flip_status.float(), offset
        )
        return last_loss, metrics

    def training_step(self, batch, batch_idx):
        if batch[0]["coord"].shape[0] < 10:
            return None

        try:
            pca_max_nn = self.trainer.datamodule.pca_max_nn_train
            loss, metrics = self._iterative_step(
                batch, batch_idx, pca_max_nn, is_train=True
            )

            acc, iou, prec, rec, mean_gt = metrics
            batch_size = batch[0]["coord"].shape[0]
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/acc", acc, sync_dist=True)
            self.log("train/iou", iou, sync_dist=True)
            self.log("train/precision", prec, sync_dist=True)
            self.log("train/recall", rec, sync_dist=True)
            self.log("train/mean_gt", mean_gt, sync_dist=True)

            return loss
        except Exception as e:
            print(f"Error in training_step (batch {batch_idx}): {e}")
            return None

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch[0]["coord"].shape[0] < 10:
            return None

        try:
            pca_max_nn = self.trainer.datamodule.pca_max_nn_val
            loss, metrics = self._iterative_step(
                batch, batch_idx, pca_max_nn, is_train=False
            )

            acc, iou, prec, rec, mean_gt = metrics
            batch_size = batch[0]["coord"].shape[0]

            val_subfolder = self.trainer.datamodule.val_subfolders[dataloader_idx]
            dataset_name = val_subfolder.replace('/', '_')

            self.log(f"val/acc_{dataset_name}", acc, sync_dist=True, batch_size=batch_size, add_dataloader_idx=False)

            return loss
        except Exception as e:
            print(f"Error in validation_step (batch {batch_idx}): {e}")
            return None

    def test_step(self, batch, batch_idx):
        if batch[0]["coord"].shape[0] < 2:
            return None

        try:
            pca_max_nn = self.trainer.datamodule.pca_max_nn_test
            loss, metrics = self._iterative_step(
                batch, batch_idx, pca_max_nn, is_train=False
            )

            acc, iou, prec, rec, mean_gt = metrics
            batch_size = batch[0]["coord"].shape[0]
            self.log("test/loss", loss, sync_dist=True, batch_size=batch_size)
            self.log("test/acc", acc, sync_dist=True, batch_size=batch_size)
            self.log("test/iou", iou, sync_dist=True, batch_size=batch_size)
            self.log("test/precision", prec, sync_dist=True, batch_size=batch_size)
            self.log("test/recall", rec, sync_dist=True, batch_size=batch_size)
            self.log("test/mean_gt", mean_gt, sync_dist=True, batch_size=batch_size)

            return loss
        except Exception as e:
            print(f"Error in test_step (batch {batch_idx}): {e}")
            return None

    def on_train_epoch_end(self):
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            if not isinstance(lr_schedulers, list):
                lr_schedulers = [lr_schedulers]
            for sch in lr_schedulers:
                if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step()

    def on_validation_epoch_end(self):
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            if not isinstance(lr_schedulers, list):
                lr_schedulers = [lr_schedulers]
            for sch in lr_schedulers:
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = self.trainer.callback_metrics.get("val/loss")
                    if val_loss is not None:
                        sch.step(val_loss)

    def on_train_start(self):
        """Warmup GPU memory by finding and processing the largest batch"""
        if self.trainer.global_rank == 0:
            print("Warming up GPU memory...")

        train_loader = self.trainer.train_dataloader
        max_batch = self._find_largest_batch(train_loader)

        if max_batch is not None:
            max_batch = self._move_batch_to_device(max_batch)

            optimizer = self.optimizers()
            optimizer.zero_grad()
            _ = self.training_step(max_batch, 0)

        if self.trainer.global_rank == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory reserved: {reserved:.2f} GB (allocated: {allocated:.2f} GB)")

    def _move_batch_to_device(self, batch):
        """Move batch data to current device"""
        point_data, gt_normal = batch

        point_data_gpu = {}
        for key, value in point_data.items():
            if isinstance(value, torch.Tensor):
                point_data_gpu[key] = value.to(self.device)
            else:
                point_data_gpu[key] = value

        gt_normal_gpu = gt_normal.to(self.device) if isinstance(gt_normal, torch.Tensor) else gt_normal

        return point_data_gpu, gt_normal_gpu

    def _find_largest_batch(self, dataloader, sample_size=100):
        """Sample dataloader to find the largest batch"""
        max_batch = None
        max_points = 0

        for i, batch in enumerate(dataloader):
            if i >= sample_size:
                break

            point_data, gt_normal = batch
            num_points = len(point_data['coord'])

            if num_points > max_points:
                max_points = num_points
                max_batch = batch

        if self.trainer.global_rank == 0 and max_batch is not None:
            print(f"Found largest batch with {max_points} points")

        return max_batch
