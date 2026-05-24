import os
import time
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cpp_alg import extract_largest_connected_component, process_ply_file


def process_file(input_path, output_path, k):
    """
    处理单个 PLY 文件

    Returns:
        dict: 包含处理统计信息
    """
    t0 = time.time()

    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points).astype(np.float32)
    N_in = len(points)

    mask, timing = extract_largest_connected_component(points, k=k, return_timing=True)
    mask_np = mask.astype(bool)

    # 统计连通分量信息
    from sklearn.neighbors import KDTree
    tree = KDTree(points)
    _, nn = tree.query(points, k=k + 1)
    neighbors = nn[:, 1:]

    component_id = np.full(N_in, -1, dtype=np.int32)
    component_sizes = []
    next_id = 0

    for i in range(N_in):
        if component_id[i] == -1:
            queue = [i]
            component_id[i] = next_id
            size = 1
            qpos = 0

            while qpos < len(queue):
                curr = queue[qpos]
                qpos += 1
                for nb in neighbors[curr]:
                    if 0 <= nb < N_in and component_id[nb] == -1:
                        component_id[nb] = next_id
                        queue.append(nb)
                        size += 1

            component_sizes.append(size)
            next_id += 1

    num_components = len(component_sizes)
    component_sizes_sorted = sorted(component_sizes, reverse=True)
    second_largest = component_sizes_sorted[1] if num_components > 1 else 0

    filtered_points = points[mask_np]
    N_out = len(filtered_points)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(filtered_points)

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)[mask_np]
        pcd_out.normals = o3d.utility.Vector3dVector(normals)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[mask_np]
        pcd_out.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(output_path), pcd_out)

    elapsed = time.time() - t0

    return {
        'filename': input_path.name,
        'input_points': N_in,
        'output_points': N_out,
        'num_components': num_components,
        'retention_ratio': N_out / N_in if N_in > 0 else 0,
        'second_largest_points': second_largest,
        'kdtree_build_time': timing['kdtree_build'],
        'knn_query_time': timing['knn_query'],
        'bfs_time': timing['bfs'],
        'cpp_total_time': timing['total'],
        'total_time': elapsed
    }


def process_file_cpp(input_path, output_path, k):
    """
    C++ 加速版本（I/O 也在 C++）

    Returns:
        dict: 包含处理统计信息
    """
    t0 = time.time()

    stats, timing = process_ply_file(
        str(input_path),
        str(output_path),
        k=k,
        return_timing=True
    )

    total_time = time.time() - t0

    return {
        'filename': input_path.name,
        'input_points': stats['input_points'],
        'output_points': stats['output_points'],
        'num_components': stats['num_components'],
        'retention_ratio': stats['retention_ratio'],
        'second_largest_points': stats['second_largest_points'],
        'read_time': timing['read'],
        'kdtree_build_time': timing['kdtree_build'],
        'knn_query_time': timing['knn_query'],
        'bfs_time': timing['bfs'],
        'filter_time': timing['filter'],
        'write_time': timing['write'],
        'cpp_total_time': timing['total'],
        'total_time': total_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--log_csv', type=str, default=None)
    parser.add_argument('--use_cpp_io', action='store_true',
                        help='Use C++ I/O (10x+ faster)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_csv is None:
        args.log_csv = str(output_dir / 'processing_log.csv')

    ply_files = sorted(input_dir.rglob('**/*.ply'))

    if not ply_files:
        print(f"未找到 PLY 文件: {input_dir}")
        return

    io_mode = "C++ I/O" if args.use_cpp_io else "Python I/O (open3d)"
    print(f"处理 {len(ply_files)} 个文件，k={args.k}，I/O 模式: {io_mode}")

    process_fn = process_file_cpp if args.use_cpp_io else process_file
    results = []
    for ply_file in tqdm(ply_files):
        rel_path = ply_file.relative_to(input_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = process_fn(ply_file, output_path, args.k)
            result['relative_path'] = str(rel_path)
            results.append(result)
        except Exception as e:
            print(f"\n处理失败 {rel_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(args.log_csv, index=False)
    print(f"\n日志已保存: {args.log_csv}")

    if df.empty or 'retention_ratio' not in df.columns:
        print("\n所有文件处理失败，无统计信息")
        return

    print(f"\n平均保留率: {df['retention_ratio'].mean():.4f}")
    print(f"平均分量数: {df['num_components'].mean():.2f}")
    print(f"平均总耗时: {df['total_time'].mean():.2f}s")

    if args.use_cpp_io:
        print(f"  - PLY读取: {df['read_time'].mean():.3f}s")
    print(f"  - KD-tree构建: {df['kdtree_build_time'].mean():.3f}s")
    print(f"  - KNN查询: {df['knn_query_time'].mean():.3f}s")
    print(f"  - BFS: {df['bfs_time'].mean():.3f}s")
    if args.use_cpp_io:
        print(f"  - 数据过滤: {df['filter_time'].mean():.3f}s")
        print(f"  - PLY写入: {df['write_time'].mean():.3f}s")
    print(f"  - C++总计: {df['cpp_total_time'].mean():.3f}s")


if __name__ == '__main__':
    main()
