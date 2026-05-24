import socket
import numpy as np
import json
import argparse
import glob
import os


def calculate_unoriented_accuracy(pred_normals, gt_normals):
    """计算无方向准确率（0-90度）"""
    # 归一化
    pred_normals = pred_normals / (np.linalg.norm(pred_normals, axis=1, keepdims=True) + 1e-8)
    gt_normals = gt_normals / (np.linalg.norm(gt_normals, axis=1, keepdims=True) + 1e-8)

    # 点积
    dot_product = np.sum(pred_normals * gt_normals, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 角度（使用abs处理无方向）
    angles_rad = np.arccos(np.abs(dot_product))
    angles_deg = angles_rad * 180.0 / np.pi

    # 统计
    mean_angle = np.mean(angles_deg)
    median_angle = np.median(angles_deg)
    rmse = np.sqrt(np.mean(angles_deg ** 2))
    pgp5 = np.mean(angles_deg < 5.0) * 100
    pgp10 = np.mean(angles_deg < 10.0) * 100
    pgp30 = np.mean(angles_deg < 30.0) * 100

    return {
        'mean_angle': mean_angle,
        'median_angle': median_angle,
        'rmse': rmse,
        'pgp5': pgp5,
        'pgp10': pgp10,
        'pgp30': pgp30
    }


def calculate_oriented_accuracy(pred_normals, gt_normals):
    """计算有方向准确率（0-180度）"""
    # 归一化
    pred_normals = pred_normals / (np.linalg.norm(pred_normals, axis=1, keepdims=True) + 1e-8)
    gt_normals = gt_normals / (np.linalg.norm(gt_normals, axis=1, keepdims=True) + 1e-8)

    # 点积
    dot_product = np.sum(pred_normals * gt_normals, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 角度（不使用abs，允许0-180度）
    angles_rad = np.arccos(dot_product)
    angles_deg = angles_rad * 180.0 / np.pi

    # 统计
    mean_angle = np.mean(angles_deg)
    median_angle = np.median(angles_deg)
    rmse = np.sqrt(np.mean(angles_deg ** 2))
    pgp5 = np.mean(angles_deg < 5.0) * 100
    pgp10 = np.mean(angles_deg < 10.0) * 100
    pgp30 = np.mean(angles_deg < 30.0) * 100

    return {
        'mean_angle': mean_angle,
        'median_angle': median_angle,
        'rmse': rmse,
        'pgp5': pgp5,
        'pgp10': pgp10,
        'pgp30': pgp30
    }


def test_single_file(ply_path, host, port):
    """测试单个PLY文件"""
    import open3d as o3d

    # 1. 加载PLY文件
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points).astype(np.float32)
    gt_normals = np.asarray(pcd.normals).astype(np.float32)

    # 2. 连接socket服务器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        # 3. 发送请求
        request = json.dumps({
            "data_size": len(xyz),
            "function_name": "direct_orientation",
            "function_config": {}
        })
        s.sendall(request.encode())

        # 4. 接收ACK
        ack = s.recv(1024)
        response = json.loads(ack.decode())
        if response.get("status") != "OK":
            raise Exception("Server returned error")

        # 5. 发送点云数据
        data = xyz.astype(np.float64).tobytes()
        s.sendall(data)

        # 6. 接收结果
        result_size = len(xyz) * 6 * 8  # 6 channels, 8 bytes per float64
        result_data = b''
        while len(result_data) < result_size:
            chunk = s.recv(result_size - len(result_data))
            if not chunk:
                break
            result_data += chunk

        result = np.frombuffer(result_data, dtype=np.float64).reshape(-1, 6)
        pred_normals = result[:, 3:6].astype(np.float32)

    return pred_normals, gt_normals, xyz


def calculate_summary(all_results, total_points):
    """计算整体统计信息"""
    valid_results = [r for r in all_results if 'error' not in r]

    if not valid_results:
        return {'error': 'No valid results'}

    # 加权平均（按点数加权）
    weighted_mean = sum(r['mean_angle'] * r['num_points'] for r in valid_results) / total_points
    weighted_pgp10 = sum(r['pgp10'] * r['num_points'] for r in valid_results) / total_points

    summary = {
        'total_files': len(all_results),
        'successful_files': len(valid_results),
        'failed_files': len(all_results) - len(valid_results),
        'total_points': total_points,
        'avg_mean_angle': weighted_mean,
        'avg_median_angle': np.mean([r['median_angle'] for r in valid_results]),
        'avg_rmse': np.mean([r['rmse'] for r in valid_results]),
        'avg_pgp5': np.mean([r['pgp5'] for r in valid_results]),
        'avg_pgp10': weighted_pgp10,
        'avg_pgp30': np.mean([r['pgp30'] for r in valid_results]),
    }

    return summary


def test_directory(data_dir, host, port, metric_type='unoriented'):
    """测试目录中所有PLY文件"""
    # 查找所有PLY文件
    ply_files = sorted(glob.glob(os.path.join(data_dir, '*.ply')))
    print(f"Found {len(ply_files)} PLY files")

    all_results = []
    total_points = 0

    for i, ply_path in enumerate(ply_files):
        filename = os.path.basename(ply_path)
        print(f"[{i+1}/{len(ply_files)}] Testing {filename}...")

        try:
            # 测试单个文件
            pred_normals, gt_normals, xyz = test_single_file(ply_path, host, port)

            # 计算准确率
            if metric_type == 'unoriented':
                metrics = calculate_unoriented_accuracy(pred_normals, gt_normals)
            else:
                metrics = calculate_oriented_accuracy(pred_normals, gt_normals)

            # 记录结果
            result = {
                'filename': filename,
                'num_points': len(xyz),
                **metrics
            }
            all_results.append(result)
            total_points += len(xyz)

            print(f"  Points: {len(xyz)}, Mean Error: {metrics['mean_angle']:.2f}°, "
                  f"PGP10: {metrics['pgp10']:.1f}%")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'filename': filename,
                'error': str(e)
            })

    # 计算整体统计
    summary = calculate_summary(all_results, total_points)

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='Test socket API for normal estimation')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing PLY files')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8999, help='Server port')
    parser.add_argument('--metric', type=str, default='unoriented',
                        choices=['unoriented', 'oriented'], help='Accuracy metric type')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')

    args = parser.parse_args()

    print(f"Testing API at {args.host}:{args.port}")
    print(f"Data directory: {args.data_dir}")
    print(f"Metric type: {args.metric}")
    print()

    # 测试目录
    all_results, summary = test_directory(args.data_dir, args.host, args.port, args.metric)

    # 打印结果
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # 保存结果
    if args.output:
        output_data = {
            'summary': summary,
            'details': all_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
