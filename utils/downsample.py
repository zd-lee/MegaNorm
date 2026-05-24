"""
Point Cloud Downsampling Utilities

Provides efficient random downsampling for point clouds with optional normal preservation.
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.pointcloud_io import write_ply_ascii





def normalize_point_cloud(pcd):
    '''
    归一化点云数据,将点云数据归一化到单位立方体中心
    :param xyz: 点云数据
    :return: 归一化后的点云数据
    '''
    shape_scale = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    shape_center = pcd.get_center()
    pcd = pcd.translate(-shape_center)
    pcd = pcd.scale(1 / shape_scale, center=(0, 0, 0))
    return pcd,shape_scale,shape_center

 
def voxel_downsampling(pcd,voxel_size):
    """
    Downsample a point cloud using voxel downsampling.
    下采样前会先对点云进行归一化;下采样结束后再将点云还原
    """
    _,scale,center = normalize_point_cloud(pcd)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_downsampled.scale(scale,[0,0,0])
    pcd_downsampled.translate(center)
    return np.asarray(pcd_downsampled.points)


def rate_voxel_downsampling(pcd,rate,offset=0.10,verbose=False):
    """_summary_
    多次调用voxel_downsampling函数,直到点云的数量在rate*原点云数量左右
    Args:
        rate (float): 
        offset (float): 允许的误差范围
    """
    count = len(pcd.points)
    longest_axis = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    # 二分查找
    low = 0
    high = longest_axis/np.cbrt(count*rate) + 0.001
    times = 0
    while True:
        mid = (low + high) / 2
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=mid)
        if len(pcd_downsampled.points) > count*rate*(1+offset):
            low = mid
        elif len(pcd_downsampled.points) < count*rate*(1-offset):
            high = mid
        else:
            break
        print("times: {}, mid: {}, rate: {}".format(times,mid,len(pcd_downsampled.points)/count))
        times += 1
        if times > 100:
            print("downsampling failed after 100 times")
            break
    
    print("after {} times downsampling, choose voxel size: {}, get rate: {}".format(times,mid,len(pcd_downsampled.points)/count))
    return pcd_downsampled


def random_downsample(
    coords: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    ratio: Optional[float] = None,
    target_points: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Randomly downsample a point cloud.

    Args:
        coords: (N, 3) point coordinates
        normals: (N, 3) point normals (optional)
        colors: (N, 3) point colors (optional)
        ratio: Downsampling ratio in (0, 1], e.g., 0.5 to keep 50% of points
        target_points: Target number of points (alternative to ratio)
        seed: Random seed for reproducibility

    Returns:
        downsampled_coords: (M, 3) downsampled coordinates
        downsampled_normals: (M, 3) downsampled normals (if normals provided)
        downsampled_colors: (M, 3) downsampled colors (if colors provided)

    Note:
        Either ratio or target_points must be specified, but not both.
    """
    N = len(coords)

    if ratio is None and target_points is None:
        raise ValueError("Either ratio or target_points must be specified")
    if ratio is not None and target_points is not None:
        raise ValueError("Cannot specify both ratio and target_points")

    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        num_samples = max(1, int(N * ratio))
    else:
        if target_points <= 0:
            raise ValueError(f"target_points must be positive, got {target_points}")
        if target_points > N:
            print(f"Warning: target_points ({target_points}) > num_points ({N}), keeping all points")
            return coords, normals, colors
        num_samples = target_points

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.choice(N, size=num_samples, replace=False)

    downsampled_coords = coords[indices]
    downsampled_normals = normals[indices] if normals is not None else None
    downsampled_colors = colors[indices] if colors is not None else None

    return downsampled_coords, downsampled_normals, downsampled_colors


def downsample_ply_file(
    input_path: str,
    output_path: str,
    ratio: Optional[float] = None,
    target_points: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    method: str = "random"
) -> dict:
    """
    Downsample a single PLY file.

    Args:
        input_path: Input PLY file path
        output_path: Output PLY file path
        ratio: Downsampling ratio
        target_points: Target number of points
        seed: Random seed
        verbose: Print progress information
        method: Downsampling method ("random" or "voxel")

    Returns:
        stats: Dictionary with statistics
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for reading PLY files")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if verbose:
        print(f"Processing: {input_path.name}")

    pcd = o3d.io.read_point_cloud(str(input_path))
    coords = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None

    if colors is not None:
        colors = (colors * 255).astype(np.uint8)

    original_num_points = len(coords)

    if method == "voxel":
        if ratio is None:
            raise ValueError("voxel method requires ratio parameter")
        pcd_downsampled = rate_voxel_downsampling(pcd, ratio, verbose=verbose)
        downsampled_coords = np.asarray(pcd_downsampled.points, dtype=np.float32)
        downsampled_normals = np.asarray(pcd_downsampled.normals, dtype=np.float32) if pcd_downsampled.has_normals() else None
        downsampled_colors = np.asarray(pcd_downsampled.colors, dtype=np.float32) if pcd_downsampled.has_colors() else None
        if downsampled_colors is not None:
            downsampled_colors = (downsampled_colors * 255).astype(np.uint8)
    else:
        downsampled_coords, downsampled_normals, downsampled_colors = random_downsample(
            coords, normals, colors, ratio, target_points, seed
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply_ascii(str(output_path), downsampled_coords, colors=downsampled_colors, normals=downsampled_normals)

    stats = {
        'input_file': str(input_path),
        'output_file': str(output_path),
        'original_points': original_num_points,
        'downsampled_points': len(downsampled_coords),
        'ratio_actual': len(downsampled_coords) / original_num_points,
        'has_normals': normals is not None,
        'has_colors': colors is not None
    }

    if verbose:
        print(f"  Original: {original_num_points} points")
        print(f"  Downsampled: {len(downsampled_coords)} points ({stats['ratio_actual']*100:.1f}%)")
        print(f"  Saved to: {output_path}")

    return stats


def batch_downsample_directory(
    input_dir: str,
    output_dir: str,
    ratio: Optional[float] = None,
    target_points: Optional[int] = None,
    seed: Optional[int] = None,
    pattern: str = "*.ply",
    method: str = "random"
) -> list:
    """
    Batch downsample all PLY files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        ratio: Downsampling ratio
        target_points: Target number of points
        seed: Random seed
        pattern: File pattern to match (default: "*.ply")
        method: Downsampling method ("random" or "voxel")

    Returns:
        all_stats: List of statistics for each file
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ply_files = sorted(input_dir.glob(pattern))

    if len(ply_files) == 0:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return []

    print(f"Found {len(ply_files)} file(s) matching '{pattern}'")
    print(f"Output directory: {output_dir}")
    print()

    all_stats = []
    for i, ply_path in enumerate(ply_files, 1):
        print(f"[{i}/{len(ply_files)}]", end=" ")

        output_path = output_dir / ply_path.name

        try:
            stats = downsample_ply_file(
                str(ply_path),
                str(output_path),
                ratio=ratio,
                target_points=target_points,
                seed=seed,
                verbose=True,
                method=method
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        print()

    print(f"Total: {len(all_stats)}/{len(ply_files)} files processed successfully")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Efficient point cloud downsampling tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Downsample to 50% of points
  python -m utils.downsample --input data/model.ply --output data/model_half.ply --ratio 0.5

  # Downsample to 10000 points
  python -m utils.downsample --input data/model.ply --output data/model_10k.ply --target 10000

  # Batch process directory
  python -m utils.downsample --input data/input_dir/ --output data/output_dir/ --ratio 0.3

  # With random seed for reproducibility
  python -m utils.downsample --input data/model.ply --output data/model_half.ply --ratio 0.5 --seed 42
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input PLY file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output PLY file or directory')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ratio', '-r', type=float, help='Downsampling ratio (0, 1], e.g., 0.5 for 50%%')
    group.add_argument('--target', '-t', type=int, help='Target number of points')

    parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--pattern', '-p', default='*.ply', help='File pattern for batch processing (default: *.ply)')
    parser.add_argument('--method', '-m', default='random', choices=['random', 'voxel'], help='Downsampling method (default: random)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        downsample_ply_file(
            str(input_path),
            args.output,
            ratio=args.ratio,
            target_points=args.target,
            seed=args.seed,
            verbose=True,
            method=args.method
        )
    elif input_path.is_dir():
        batch_downsample_directory(
            str(input_path),
            args.output,
            ratio=args.ratio,
            target_points=args.target,
            seed=args.seed,
            pattern=args.pattern,
            method=args.method
        )
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
