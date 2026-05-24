"""Point Cloud Data Transforms"""

import numpy as np
import torch
from typing import Dict, Tuple

class NormalEstimationNormalize:
    """Normalize point cloud for normal estimation task"""

    def __init__(self, method='unit_sphere', center=True):
        self.method = method
        self.center = center

    def direct_call(self, xyz):
        """Direct normalization for xyz tensor"""
        min_coords = xyz.min(dim=0)[0]
        max_coords = xyz.max(dim=0)[0]
        center_coords = (min_coords + max_coords) / 2

        if self.center:
            xyz = xyz - center_coords
        if self.method == 'unit_sphere':
            scale = torch.max(torch.abs(xyz))
            if scale > 0:
                xyz = xyz / scale
                return xyz, center_coords, scale
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def __call__(self,
                 point_data: Dict[str, torch.Tensor],
                 gt_normal: torch.Tensor = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        coords = point_data['coord'].clone()
        feat = point_data['feat'].clone()

        min_coords = coords.min(dim=0)[0]
        max_coords = coords.max(dim=0)[0]
        center_coords = (min_coords + max_coords) / 2

        feat_has_coords = feat.shape[1] >= 3

        if self.center:
            coords = coords - center_coords
            if feat_has_coords:
                feat[:, :3] = feat[:, :3] - center_coords

        scale = 1.0
        if self.method == 'unit_sphere':
            scale = torch.max(torch.abs(coords))
            if scale > 0:
                coords = coords / scale
                if feat_has_coords:
                    feat[:, :3] = feat[:, :3] / scale

        elif self.method == 'zero_one':
            coord_range = max_coords - min_coords
            scale = torch.max(coord_range)
            if scale > 0:
                coords = (coords + (max_coords - min_coords) / 2) / scale
                if feat_has_coords:
                    feat[:, :3] = (feat[:, :3] + (max_coords - min_coords) / 2) / scale

        elif self.method == 'unit_cube':
            coord_range = max_coords - min_coords
            scale = torch.max(coord_range)
            if scale > 0:
                coords = coords / scale
                if feat_has_coords:
                    feat[:, :3] = feat[:, :3] / scale

        norm_params = {
            'min_coords': min_coords,
            'max_coords': max_coords,
            'center_coords': center_coords,
            'scale': scale,
            'method': self.method
        }

        point_data['coord'] = coords
        point_data['feat'] = feat
        point_data['norm_params'] = norm_params

        if gt_normal is not None:
            gt_normal_normalized = torch.nn.functional.normalize(gt_normal, p=2, dim=1)
        else:
            gt_normal_normalized = None

        return point_data, gt_normal_normalized


class NormalEstimationCompose:
    """Compose multiple transforms for normal estimation"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self,
                 point_data: Dict[str, torch.Tensor],
                 gt_normal: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        for transform in self.transforms:
            point_data, gt_normal = transform(point_data, gt_normal)
        return point_data, gt_normal


def get_normal_estimation_train_transforms():
    """Get training transforms for normal estimation"""
    return NormalEstimationCompose([
        NormalEstimationNormalize(method='unit_sphere', center=True),
    ])


class RandomRotation:
    """Random rotation augmentation for point clouds and normals"""

    def __init__(self, max_angle: float = 15.0, axes: str = 'xyz'):
        self.max_angle_rad = np.deg2rad(max_angle)
        self.axes = axes

    def _generate_rotation_matrix(self) -> np.ndarray:
        import scipy.spatial.transform as transform

        if self.axes == 'z':
            angle = np.random.uniform(-self.max_angle_rad, self.max_angle_rad)
            rot = transform.Rotation.from_euler('z', angle)
        elif self.axes == 'xy':
            angles = np.random.uniform(-self.max_angle_rad, self.max_angle_rad, 2)
            rot = transform.Rotation.from_euler('xy', angles)
        else:  # 'xyz'
            angles = np.random.uniform(-self.max_angle_rad, self.max_angle_rad, 3)
            rot = transform.Rotation.from_euler('xyz', angles)

        return rot.as_matrix().astype(np.float32)

    def __call__(self,
                 point_data: Dict[str, torch.Tensor],
                 gt_normal: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        rot_matrix = self._generate_rotation_matrix()
        rot_matrix_torch = torch.from_numpy(rot_matrix).to(point_data['coord'].device)

        coords = point_data['coord']
        feat = point_data['feat']

        rotated_coords = torch.matmul(coords, rot_matrix_torch.T)

        feat_has_coords = feat.shape[1] >= 3
        if feat_has_coords:
            rotated_feat = feat.clone()
            rotated_feat[:, :3] = torch.matmul(feat[:, :3], rot_matrix_torch.T)
        else:
            rotated_feat = feat

        point_data['coord'] = rotated_coords
        point_data['feat'] = rotated_feat

        if gt_normal is not None:
            rotated_normal = torch.matmul(gt_normal, rot_matrix_torch.T)
            rotated_normal = torch.nn.functional.normalize(rotated_normal, p=2, dim=1)
        else:
            rotated_normal = None

        return point_data, rotated_normal


class RandomDownsample:
    """
    Random downsampling for point clouds.

    IMPORTANT: Apply BEFORE batch collation in Dataset.__getitem__().
    """

    def __init__(self, min_ratio: float, seed: int = None, min_pts: int = 100):
        if not 0 < min_ratio <= 1:
            raise ValueError(f"min_ratio must be in (0, 1], got {min_ratio}")
        self.min_ratio = min_ratio
        self.seed      = seed
        self.min_pts   = min_pts

    def __call__(self,
                 point_data: Dict[str, torch.Tensor],
                 gt_normal: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if 'batch_offsets' in point_data or 'offsets' in point_data:
            raise RuntimeError(
                "RandomDownsample detected 'batch_offsets'. "
                "Apply this transform BEFORE batch collation."
            )

        N = len(point_data['coord'])
        if N < self.min_pts:
            return point_data, gt_normal

        ratio       = np.random.uniform(self.min_ratio, 1.0)
        num_samples = max(2, int(N * ratio))

        if self.seed is not None:
            torch.manual_seed(self.seed)

        indices = torch.randperm(N)[:num_samples]
        indices = indices.sort()[0]

        point_data['coord'] = point_data['coord'][indices]
        point_data['feat']  = point_data['feat'][indices]
        if 'batch' in point_data:
            point_data['batch'] = point_data['batch'][indices]
        if 'grid_coord' in point_data:
            point_data['grid_coord'] = point_data['grid_coord'][indices]

        downsampled_normal = gt_normal[indices] if gt_normal is not None else None

        return point_data, downsampled_normal


class GaussianNoise:
    """Add Gaussian noise to point cloud coordinates (not normals)"""

    def __init__(self, mean: float = 0.0, max_std: float = 0.005):
        self.mean    = mean
        self.max_std = max_std

    def __call__(self,
                 point_data: Dict[str, torch.Tensor],
                 gt_normal: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        coords = point_data['coord']
        feat   = point_data['feat']

        std          = np.random.uniform(0, self.max_std)
        noise        = torch.randn_like(coords) * std + self.mean
        noisy_coords = coords + noise

        feat_has_coords = feat.shape[1] >= 3
        if feat_has_coords:
            noisy_feat = feat.clone()
            noisy_feat[:, :3] = noisy_feat[:, :3] + noise
        else:
            noisy_feat = feat

        point_data['coord'] = noisy_coords
        point_data['feat']  = noisy_feat

        return point_data, gt_normal

