import os
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from dataset.multi_scale_patch_dataset import MultiScalePatchDataset
from dataset.collate_fn import NormalEstimationCollator

class DirectOrientationDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "data/t2_training_with_normals",
        scales: list = None,
        train_subfolder: str = "train",
        val_subfolder: str = "val",
        val_subfolders: list = None,
        test_subfolder: str = "test",
        batch_size: int = 1,
        num_workers: int = 4,
        grid_size: float = 0.02,
        pca_max_nn_train: int = 10,
        pca_max_nn_val: int = 30,
        pca_max_nn_test: int = 30,
        device: str = "cuda",
        train_augmentation: dict = None,
        val_augmentation: dict = None,
        test_augmentation: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.scales = scales or [
            {"max_points_per_patch": 10000, "method": "fps", "k": 10, "patch_count": None, "overlap_rate": 0.0},
            {"max_points_per_patch": 80000, "method": "fps", "k": 10, "patch_count": None, "overlap_rate": 0.0},
            {"max_points_per_patch": 30000, "method": "fps", "k": None, "patch_count": None, "overlap_rate": 0.0},
        ]
        self.train_subfolder = train_subfolder
        self.val_subfolder = val_subfolder
        self.val_subfolders = val_subfolders if val_subfolders is not None else [val_subfolder]
        self.test_subfolder = test_subfolder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grid_size = grid_size
        self.pca_max_nn_train = pca_max_nn_train
        self.pca_max_nn_val = pca_max_nn_val
        self.pca_max_nn_test = pca_max_nn_test
        self.device = device
        self.train_augmentation = train_augmentation or {}
        self.val_augmentation = val_augmentation or {}
        self.test_augmentation = test_augmentation or {}
        self.collate_fn = NormalEstimationCollator()

    def _create_transforms(self, augmentation_config: dict):
        """Create transform pipeline based on augmentation config"""
        from dataset.transforms import (
            RandomRotation,
            RandomDownsample,
            GaussianNoise,
            NormalEstimationNormalize,
            NormalEstimationCompose
        )

        transform_list = []

        rot_config = augmentation_config.get('random_rotation', {})
        if rot_config.get('enabled', False):
            max_angle = rot_config.get('max_angle', 15.0)
            axes = rot_config.get('axes', 'xyz')
            transform_list.append(RandomRotation(max_angle=max_angle, axes=axes))

        downsample_config = augmentation_config.get('random_downsample', {})
        if downsample_config.get('enabled', False):
            min_ratio = downsample_config.get('min_ratio', 0.8)
            min_pts   = downsample_config.get('min_pts', 100)
            seed      = downsample_config.get('seed', None)
            transform_list.append(RandomDownsample(min_ratio=min_ratio, min_pts=min_pts, seed=seed))

        noise_config = augmentation_config.get('gaussian_noise', {})
        if noise_config.get('enabled', False):
            mean    = noise_config.get('mean', 0.0)
            max_std = noise_config.get('max_std', 0.005)
            transform_list.append(GaussianNoise(mean=mean, max_std=max_std))

        transform_list.append(NormalEstimationNormalize(method='unit_sphere', center=True))

        return NormalEstimationCompose(transform_list)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            train_transforms = self._create_transforms(self.train_augmentation)
            val_transforms = self._create_transforms(self.val_augmentation)

            self.train_dataset = MultiScalePatchDataset(
                data_root=os.path.join(self.root, self.train_subfolder),
                scales=self.scales,
                device=self.device,
                grid_size=self.grid_size,
                transform=train_transforms
            )
            self.val_datasets = []
            for val_subfolder in self.val_subfolders:
                val_dataset = MultiScalePatchDataset(
                    data_root=os.path.join(self.root, val_subfolder),
                    scales=self.scales,
                    device=self.device,
                    grid_size=self.grid_size,
                    transform=val_transforms
                )
                self.val_datasets.append(val_dataset)

        if stage == "test" or stage is None:
            test_transforms = self._create_transforms(self.test_augmentation)

            self.test_dataset = MultiScalePatchDataset(
                data_root=os.path.join(self.root, self.test_subfolder),
                scales=self.scales,
                device=self.device,
                grid_size=self.grid_size,
                transform=test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
            for val_dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
