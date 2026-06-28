import torch
from lightning.pytorch import LightningModule

from dataset.dataset import estimate_normals_torch
from models.direct_orientation_model import create_direct_orientation_model
from utils.loss import create_loss_function
from utils.metrics import calculate_metrics_inv


class DirectOrientationModule(LightningModule):
    def __init__(
        self,
        backbone,
        mlp_head,
        iterative,
        loss_params,
        grad_clip_norm=10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        loss_config = dict(loss_params)
        if "loss_type" in loss_config:
            loss_config["type"] = loss_config.pop("loss_type")
        config = {
            "backbone": backbone,
            "mlp_head": mlp_head,
            "iterative": iterative,
            "training": {
                "loss": loss_config,
                "grad_clip_norm": grad_clip_norm,
            },
        }
        self.model = create_direct_orientation_model(config)
        self.criterion = create_loss_function(config)
        self.num_iterations = iterative.get("num_iterations", 3)
        self.use_confidence = iterative.get("use_confidence", True)
        self.use_mixup = iterative.get("use_mixup", False)
        self.grad_clip_norm = grad_clip_norm
        self.automatic_optimization = False

    def _ensure_offset(self, point_data):
        if "offset" not in point_data and "batch_offsets" in point_data:
            point_data["offset"] = point_data["batch_offsets"]
        elif "batch_offsets" not in point_data and "offset" in point_data:
            point_data["batch_offsets"] = point_data["offset"]
        return point_data["offset"]

    def forward(self, point_data):
        self._ensure_offset(point_data)
        return self.model(point_data)

    def _compute_pca_normals(self, coords, offset, pca_max_nn):
        normals = []
        start = 0
        for end in offset.detach().cpu().tolist():
            sample_coords = coords[start:end].detach().cpu()
            result = estimate_normals_torch(sample_coords, max_nn=pca_max_nn)
            if isinstance(result, torch.Tensor):
                normals.append(result[:, 3:6].float().to(self.device))
            else:
                normals.append(torch.from_numpy(result[:, 3:6]).float().to(self.device))
            start = end
        return torch.cat(normals, dim=0)

    def _compute_flip_gt(self, current_normals, gt_normals):
        return ((current_normals * gt_normals).sum(dim=1) < 0).long()

    def _create_soft_labels_mixup(self, current_normals, gt_normals, hard_labels):
        dot = torch.sum(current_normals * gt_normals, dim=1).clamp(-1, 1).abs()
        labels = hard_labels.clone().float()
        labels[hard_labels.bool()] = 0.5 + dot[hard_labels.bool()] / 2
        labels[~hard_labels.bool()] = 0.5 - dot[~hard_labels.bool()] / 2
        return labels

    def _iterative_step(self, batch, pca_max_nn, is_train):
        point_data, gt_normal = batch
        coords = point_data["coord"]
        offset = self._ensure_offset(point_data)
        pca_normals = self._compute_pca_normals(coords, offset, pca_max_nn)
        x_old = pca_normals.clone()
        conf_old = (
            torch.zeros(x_old.shape[0], 1, device=self.device)
            if self.use_confidence
            else None
        )
        optimizer = self.optimizers() if is_train else None
        last_loss = None
        last_logits = None
        last_gt_flip = None
        for _ in range(self.num_iterations):
            gt_flip = self._compute_flip_gt(x_old, gt_normal)
            if self.use_confidence:
                point_data["feat"] = torch.cat([coords, x_old, conf_old], dim=1)
            else:
                point_data["feat"] = torch.cat([coords, x_old], dim=1)
            logits = self.forward(point_data)[:, 0]
            targets = (
                self._create_soft_labels_mixup(x_old, gt_normal, gt_flip)
                if self.use_mixup
                else gt_flip.float()
            )
            loss, _ = self.criterion(logits, None, targets, offset=offset)
            if is_train:
                optimizer.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=self.grad_clip_norm,
                    gradient_clip_algorithm="norm",
                )
                optimizer.step()
            flip_prob = torch.sigmoid(logits)
            flip_mask = flip_prob > 0.5
            x_new = x_old.clone()
            x_new[flip_mask] = -x_new[flip_mask]
            x_old = x_new.detach()
            if self.use_confidence:
                conf_old = (torch.abs(flip_prob - 0.5) * 2).unsqueeze(1).detach()
            last_loss = loss
            last_logits = logits
            last_gt_flip = gt_flip
        metrics = calculate_metrics_inv(last_logits, last_gt_flip.float(), offset)
        return last_loss, metrics

    def training_step(self, batch, batch_idx):
        if batch[0]["coord"].shape[0] < 10:
            return None
        pca_max_nn = self.trainer.datamodule.pca_max_nn_train
        loss, metrics = self._iterative_step(batch, pca_max_nn, is_train=True)
        self._log_metrics("train", loss, metrics)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch[0]["coord"].shape[0] < 10:
            return None
        pca_max_nn = self.trainer.datamodule.pca_max_nn_val
        loss, metrics = self._iterative_step(batch, pca_max_nn, is_train=False)
        dataset_name = self.trainer.datamodule.val_subfolders[dataloader_idx].replace(
            "/", "_"
        )
        if dataloader_idx == 0:
            self._log_metrics("val", loss, metrics, batch)
            self.log(
                "val/acc_val",
                metrics[0],
                sync_dist=True,
                batch_size=batch[0]["coord"].shape[0],
                add_dataloader_idx=False,
            )
        self.log(
            f"val/loss_{dataset_name}",
            loss,
            sync_dist=True,
            batch_size=batch[0]["coord"].shape[0],
            add_dataloader_idx=False,
        )
        if dataset_name != "val":
            self.log(
                f"val/acc_{dataset_name}",
                metrics[0],
                sync_dist=True,
                batch_size=batch[0]["coord"].shape[0],
                add_dataloader_idx=False,
            )
        return loss

    def test_step(self, batch, batch_idx):
        if batch[0]["coord"].shape[0] < 2:
            return None
        pca_max_nn = self.trainer.datamodule.pca_max_nn_test
        loss, metrics = self._iterative_step(batch, pca_max_nn, is_train=False)
        self._log_metrics("test", loss, metrics, batch)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        point_data = batch[0] if isinstance(batch, (tuple, list)) else batch
        coords = point_data["coord"]
        offset = self._ensure_offset(point_data)
        pca_max_nn = getattr(self.trainer.datamodule, "pca_max_nn_test", 30)
        x_old = self._compute_pca_normals(coords, offset, pca_max_nn)
        conf_old = (
            torch.zeros(x_old.shape[0], 1, device=self.device)
            if self.use_confidence
            else None
        )
        last_logits = None
        for _ in range(self.num_iterations):
            if self.use_confidence:
                point_data["feat"] = torch.cat([coords, x_old, conf_old], dim=1)
            else:
                point_data["feat"] = torch.cat([coords, x_old], dim=1)
            logits = self.forward(point_data)[:, 0]
            flip_prob = torch.sigmoid(logits)
            flip_mask = flip_prob > 0.5
            x_new = x_old.clone()
            x_new[flip_mask] = -x_new[flip_mask]
            x_old = x_new
            if self.use_confidence:
                conf_old = (torch.abs(flip_prob - 0.5) * 2).unsqueeze(1)
            last_logits = logits
        return {
            "normals": x_old,
            "logits": last_logits,
            "offset": offset,
            "filenames": point_data.get("filenames"),
            "ply_paths": point_data.get("ply_paths"),
        }

    def _log_metrics(self, prefix, loss, metrics, batch=None):
        acc, iou, precision, recall, mean_gt = metrics
        kwargs = {"sync_dist": True, "add_dataloader_idx": False}
        if batch is not None:
            kwargs["batch_size"] = batch[0]["coord"].shape[0]
        self.log(f"{prefix}/loss", loss, prog_bar=prefix == "train", **kwargs)
        self.log(f"{prefix}/acc", acc, prog_bar=prefix == "train", **kwargs)
        self.log(f"{prefix}/iou", iou, **kwargs)
        self.log(f"{prefix}/precision", precision, **kwargs)
        self.log(f"{prefix}/recall", recall, **kwargs)
        self.log(f"{prefix}/mean_gt", mean_gt, **kwargs)

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

    def on_validation_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = self.trainer.callback_metrics.get("val/loss")
                if val_loss is not None:
                    scheduler.step(val_loss)
