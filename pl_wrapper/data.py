import os
import math

import torch
import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Sampler

from dataset.collate_fn import NormalEstimationCollator
from dataset.multi_scale_patch_dataset import MultiScalePatchDataset


class PatchSizeDistributedSampler(Sampler):
    def __init__(
        self,
        dataset,
        *,
        shuffle=True,
        seed=42,
        largest_first_epoch0=True,
        drop_last=False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.largest_first_epoch0 = largest_first_epoch0
        self.drop_last = drop_last
        self.epoch = 0
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        dataset_len = len(self.dataset)
        if self.drop_last and dataset_len % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (dataset_len - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(dataset_len / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.patch_sizes = [
            len(metadata["patch_indices"]) for metadata in self.dataset.patch_metadata
        ]

    def __iter__(self):
        if self.epoch == 0 and self.largest_first_epoch0:
            indices = sorted(
                range(len(self.dataset)),
                key=lambda idx: self.patch_sizes[idx],
                reverse=True,
            )
        elif self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            indices = indices[: self.total_size]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DirectOrientationDataModule(LightningDataModule):
    def __init__(
        self,
        root="data/SceneNN_part",
        scales=None,
        train_subfolder="train",
        val_subfolder="val",
        val_subfolders=None,
        test_subfolder="test",
        batch_size=1,
        num_workers=4,
        grid_size=0.02,
        pca_max_nn_train=10,
        pca_max_nn_val=30,
        pca_max_nn_test=30,
        device="cuda",
        train_augmentation=None,
        val_augmentation=None,
        test_augmentation=None,
        train_largest_first_epoch0=False,
        train_shuffle_after_epoch0=True,
        sampler_seed=42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.scales = scales or [
            {
                "max_points_per_patch": 10000,
                "method": "fps",
                "k": 10,
                "patch_count": None,
                "overlap_rate": 0.0,
            }
        ]
        self.train_subfolder = train_subfolder
        self.val_subfolder = val_subfolder
        self.val_subfolders = val_subfolders or [val_subfolder]
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
        self.train_largest_first_epoch0 = train_largest_first_epoch0
        self.train_shuffle_after_epoch0 = train_shuffle_after_epoch0
        self.sampler_seed = sampler_seed
        self.collate_fn = NormalEstimationCollator()

    def _create_transforms(self, augmentation_config):
        from dataset.transforms import (
            GaussianNoise,
            NormalEstimationCompose,
            NormalEstimationNormalize,
            RandomDownsample,
            RandomRotation,
        )

        transforms = []
        rotation = augmentation_config.get("random_rotation", {})
        if rotation.get("enabled", False):
            transforms.append(
                RandomRotation(
                    max_angle=rotation.get("max_angle", 15.0),
                    axes=rotation.get("axes", "xyz"),
                )
            )
        downsample = augmentation_config.get("random_downsample", {})
        if downsample.get("enabled", False):
            transforms.append(
                RandomDownsample(
                    min_ratio=downsample.get("min_ratio", 0.8),
                    min_pts=downsample.get("min_pts", 100),
                    seed=downsample.get("seed"),
                )
            )
        noise = augmentation_config.get("gaussian_noise", {})
        if noise.get("enabled", False):
            transforms.append(
                GaussianNoise(
                    mean=noise.get("mean", 0.0),
                    max_std=noise.get("max_std", 0.005),
                )
            )
        transforms.append(NormalEstimationNormalize(method="unit_sphere", center=True))
        return NormalEstimationCompose(transforms)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(
                self.train_subfolder, self.train_augmentation
            )
            self.val_datasets = [
                self._make_dataset(subfolder, self.val_augmentation)
                for subfolder in self.val_subfolders
            ]
        if stage in (None, "test", "predict"):
            self.test_dataset = self._make_dataset(
                self.test_subfolder, self.test_augmentation
            )

    def _make_dataset(self, subfolder, augmentation):
        return MultiScalePatchDataset(
            data_root=os.path.join(self.root, subfolder),
            scales=self.scales,
            device=self.device,
            grid_size=self.grid_size,
            transform=self._create_transforms(augmentation),
        )

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.train_largest_first_epoch0:
            sampler = PatchSizeDistributedSampler(
                self.train_dataset,
                shuffle=self.train_shuffle_after_epoch0,
                seed=self.sampler_seed,
                largest_first_epoch0=True,
            )
            shuffle = False
            preview_indices = list(iter(sampler))[: min(8, len(sampler))]
            preview_sizes = [sampler.patch_sizes[idx] for idx in preview_indices]
            print(
                "Train sampler: largest_first_epoch0=True, "
                f"rank={sampler.rank}/{sampler.num_replicas}, "
                f"first_patch_sizes={preview_sizes}"
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
