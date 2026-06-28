import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.dataset import estimate_normals_torch
from dataset.patch_extractor import extract_patches_unified
from dataset.transforms import NormalEstimationNormalize
from inference.optimization import FlipOptimizer, compute_objective
from models.direct_orientation_model import create_direct_orientation_model
from models.edge_consistency_mlp import create_edge_consistency_mlp
from utils.config import load_config
from utils.convert_checkpoint import load_model_state_dict
from utils.metrics import calculate_edge_accuracy, calculate_normal_metrics
from utils.pointcloud_io import write_ply_ascii


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def infer_scene_name_from_ply(ply_path):
    parent_name = ply_path.parent.name
    return parent_name if ply_path.stem.startswith(parent_name) else None


def find_scene_ply(scene_dir, pattern):
    matches = sorted(p for p in scene_dir.glob(pattern) if p.is_file())
    if not matches:
        return None, f"no files matching '{pattern}'"
    preferred_name = f"{scene_dir.name}_raw_pointcloud.ply"
    preferred = [p for p in matches if p.name == preferred_name]
    if len(preferred) == 1:
        return preferred[0], None
    if len(matches) == 1:
        return matches[0], None
    return None, f"multiple files matching '{pattern}'"


def build_inference_target(ply_path, source_mode, scene_name=None):
    scene_name = scene_name or infer_scene_name_from_ply(ply_path)
    output_name = scene_name or ply_path.stem
    return {
        "ply_path": ply_path,
        "source_mode": source_mode,
        "scene_name": scene_name,
        "output_name": output_name,
    }


def resolve_input_targets(
    input_path, input_mode="auto", pattern="*_raw_pointcloud.ply", recursive=False
):
    input_path = Path(input_path)
    targets = []
    skipped = []
    if input_mode == "file" or (input_mode == "auto" and input_path.is_file()):
        if input_path.suffix.lower() != ".ply":
            skipped.append(
                {
                    "input_path": str(input_path),
                    "status": "skipped",
                    "error": "input is not a .ply file",
                }
            )
        else:
            targets.append(build_inference_target(input_path, "file"))
        return targets, skipped, "file"
    if not input_path.is_dir():
        return (
            targets,
            [
                {
                    "input_path": str(input_path),
                    "status": "skipped",
                    "error": "input does not exist",
                }
            ],
            input_mode,
        )
    if input_mode == "scene_dir" or (input_mode == "auto" and not recursive):
        ply_path, error = find_scene_ply(input_path, pattern)
        if ply_path is not None:
            return (
                [build_inference_target(ply_path, "scene_dir", input_path.name)],
                skipped,
                "scene_dir",
            )
        if input_mode == "scene_dir":
            return (
                targets,
                [{"input_path": str(input_path), "status": "skipped", "error": error}],
                "scene_dir",
            )
    search_dirs = [p for p in sorted(input_path.iterdir()) if p.is_dir()]
    for scene_dir in search_dirs:
        ply_path, error = find_scene_ply(scene_dir, pattern)
        if ply_path is None:
            skipped.append(
                {
                    "input_path": str(scene_dir),
                    "scene_name": scene_dir.name,
                    "status": "skipped",
                    "error": error,
                }
            )
        else:
            targets.append(
                build_inference_target(ply_path, "scan_root", scene_dir.name)
            )
    if not targets and not skipped:
        for ply_path in sorted(input_path.glob("*.ply")):
            targets.append(build_inference_target(ply_path, "flat_dir"))
    return (
        targets,
        skipped,
        "scan_root" if recursive or input_mode == "scan_root" else "auto_dir",
    )


def load_patchnet(config, device):
    model_config = load_config(config["models"]["patchnet"]["config"])
    model = create_direct_orientation_model(model_config)
    model.load_state_dict(load_model_state_dict(config["models"]["patchnet"]["checkpoint"]))
    return model.to(device).eval()


def load_edgenet(config, device):
    model_config = load_config(config["models"]["edgenet"]["config"])
    model = create_edge_consistency_mlp(model_config)
    model.load_state_dict(load_model_state_dict(config["models"]["edgenet"]["checkpoint"]))
    return model.to(device).eval()


def run_patchnet(patches, coords, patchnet, config, device):
    transform = NormalEstimationNormalize()
    grid_size_value = config["inference"]["grid_size"]
    pca_max_nn = config["inference"]["pca_max_nn"]
    batch_size = config["inference"].get("batch_size", 1)
    num_iterations = config.get("iterative", {}).get("num_iterations", 3)
    use_confidence = config.get("iterative", {}).get("use_confidence", True)
    patch_normals = []
    patch_features = []
    patch_centers = []
    pca_time = 0.0
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(patches), batch_size), desc="PatchNet"):
            batch_patches = patches[batch_start : batch_start + batch_size]
            patch_data = []
            for patch_indices in batch_patches:
                indices = patch_indices.detach().cpu().numpy()
                patch_coords = torch.from_numpy(coords[indices]).float()
                t0 = time.time()
                pca_normals = torch.from_numpy(
                    estimate_normals_torch(patch_coords.numpy(), max_nn=pca_max_nn)[
                        :, 3:6
                    ]
                ).float()
                pca_time += time.time() - t0
                conf = torch.zeros(len(patch_coords), 1) if use_confidence else None
                patch_data.append([patch_coords, pca_normals, conf])
            for _ in range(num_iterations):
                batch_coords = []
                batch_feats = []
                batch_sizes = []
                for patch_coords, normals, conf in patch_data:
                    feat = (
                        torch.cat([patch_coords, normals, conf], dim=1)
                        if use_confidence
                        else torch.cat([patch_coords, normals], dim=1)
                    )
                    point_data = {
                        "coord": patch_coords.to(device),
                        "feat": feat.to(device),
                        "offset": torch.tensor(
                            [len(patch_coords)], device=device
                        ).long(),
                        "grid_size": grid_size_value,
                    }
                    point_data, _ = transform(point_data)
                    batch_coords.append(point_data["coord"])
                    batch_feats.append(point_data["feat"])
                    batch_sizes.append(len(patch_coords))
                coords_batch = torch.cat(batch_coords, dim=0)
                feats_batch = torch.cat(batch_feats, dim=0)
                offsets = torch.cumsum(
                    torch.tensor(batch_sizes, device=device), dim=0
                ).long()
                logits = patchnet(
                    {
                        "coord": coords_batch,
                        "feat": feats_batch,
                        "offset": offsets,
                        "grid_size": grid_size_value,
                    }
                )[:, 0]
                flip_prob = torch.sigmoid(logits).cpu()
                start = 0
                for idx, (_, normals, conf) in enumerate(patch_data):
                    end = start + len(normals)
                    flip_mask = flip_prob[start:end] > 0.5
                    normals[flip_mask] = -normals[flip_mask]
                    if use_confidence:
                        patch_data[idx][2] = (
                            torch.abs(flip_prob[start:end] - 0.5) * 2
                        ).unsqueeze(1)
                    start = end
            for patch_coords, normals, conf in patch_data:
                feat = (
                    torch.cat([patch_coords, normals, conf], dim=1)
                    if use_confidence
                    else torch.cat([patch_coords, normals], dim=1)
                )
                point_data = {
                    "coord": patch_coords.to(device),
                    "feat": feat.to(device),
                    "offset": torch.tensor([len(patch_coords)], device=device).long(),
                    "grid_size": grid_size_value,
                }
                point_data, _ = transform(point_data)
                feature = (
                    patchnet.backbone(point_data, use_decoder=False)["feat"]
                    .max(dim=0)[0]
                    .cpu()
                )
                patch_normals.append(normals.cpu())
                patch_features.append(feature)
                patch_centers.append(patch_coords.mean(dim=0))
    return patch_normals, patch_features, patch_centers, pca_time


def build_knn_graph(patch_centers, k):
    centers = torch.stack(patch_centers).numpy()
    count = len(centers)
    if count < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(k, count - 1)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    sources = np.repeat(np.arange(count), k)
    return np.stack([sources, neighbors.reshape(-1)], axis=1).astype(np.int64)


def build_consistency_matrices(edgenet, patch_features, patch_centers, k, device):
    edges = build_knn_graph(patch_centers, k)
    count = len(patch_features)
    same = np.zeros((count, count), dtype=np.float32)
    opposite = np.zeros((count, count), dtype=np.float32)
    if len(edges) == 0:
        return same, opposite, edges
    features = torch.stack(patch_features).to(device)
    centers = torch.stack(patch_centers).to(device)
    edge_tensor = torch.from_numpy(edges).long().to(device)
    with torch.no_grad():
        logits = edgenet(
            features[edge_tensor[:, 0]],
            features[edge_tensor[:, 1]],
            centers[edge_tensor[:, 0]],
            centers[edge_tensor[:, 1]],
        ).squeeze()
        prob_opposite = torch.sigmoid(logits).cpu().numpy()
    for edge_idx, (src, dst) in enumerate(edges):
        opposite_prob = float(prob_opposite[edge_idx])
        same_prob = 1.0 - opposite_prob
        same[src, dst] = same[dst, src] = same_prob
        opposite[src, dst] = opposite[dst, src] = opposite_prob
    return same, opposite, edges


def compose_point_normals(num_points, patches, patch_normals, flip_decisions=None):
    final_normals = np.zeros((num_points, 3), dtype=np.float32)
    if flip_decisions is None:
        flip_decisions = np.zeros(len(patches), dtype=np.int8)
    for patch_idx, patch_indices in enumerate(patches):
        indices = patch_indices.detach().cpu().numpy()
        normals = patch_normals[patch_idx].detach().cpu().numpy()
        if flip_decisions[patch_idx] == 1:
            normals = -normals
        final_normals[indices] = normals
    return final_normals


def calculate_patch_accuracy_stats(patches, patch_normals, gt_normals):
    patch_accuracies = []
    patch_sizes = []
    raw_patch_accuracies = []
    for patch_indices, normals in zip(patches, patch_normals):
        indices = patch_indices.detach().cpu().numpy()
        pred = normals.detach().cpu().numpy()
        gt = gt_normals[indices]
        correct = (pred * gt).sum(axis=1) > 0
        raw_acc = float(correct.mean())
        patch_acc = max(raw_acc, 1.0 - raw_acc)
        raw_patch_accuracies.append(raw_acc)
        patch_accuracies.append(patch_acc)
        patch_sizes.append(int(len(indices)))
    if not patch_accuracies:
        return None, {
            "patch_accuracies": [],
            "raw_patch_accuracies": [],
            "patch_sizes": [],
            "mean": None,
            "weighted_mean": None,
            "min": None,
            "max": None,
        }
    weights = np.asarray(patch_sizes, dtype=np.float64)
    accs = np.asarray(patch_accuracies, dtype=np.float64)
    stats = {
        "patch_accuracies": patch_accuracies,
        "raw_patch_accuracies": raw_patch_accuracies,
        "patch_sizes": patch_sizes,
        "mean": float(accs.mean()),
        "weighted_mean": float(np.average(accs, weights=weights)),
        "min": float(accs.min()),
        "max": float(accs.max()),
    }
    return stats["mean"], stats


def calculate_patch_flip_labels(patches, patch_normals, gt_normals):
    labels = []
    for patch_indices, normals in zip(patches, patch_normals):
        indices = patch_indices.detach().cpu().numpy()
        pred = normals.detach().cpu().numpy()
        gt = gt_normals[indices]
        labels.append(int(((pred * gt).sum(axis=1) < 0).mean() > 0.5))
    return np.asarray(labels, dtype=np.int8)


def load_ply(input_path):
    plydata = PlyData.read(input_path)
    vertex = plydata["vertex"]
    coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )
    names = vertex.data.dtype.names
    normals = None
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(
            np.float32
        )
    return coords, normals


def process_single_file(
    input_path, output_dir, config, patchnet, edgenet, device, no_chunk=False
):
    model_name = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    start_total = time.time()
    coords, gt_normals = load_ply(input_path)
    print(f"Loaded {len(coords)} points from {input_path}")
    t0 = time.time()
    if no_chunk:
        patches = [torch.arange(len(coords), device=device)]
    else:
        patch_config = config["patch_extraction"]
        patches, _ = extract_patches_unified(
            torch.from_numpy(coords).to(device),
            method=patch_config.get("method", "fps_bfs"),
            num_per_patch=patch_config.get("max_points_per_patch", 2000),
            patch_count=patch_config.get("patch_count"),
            overlap_rate=patch_config.get("overlap_rate", 0.5),
            k=patch_config.get("k", 10),
            device=device,
        )
        patches = sorted(patches, key=lambda patch: len(patch), reverse=True)
    timings["patch_extraction"] = time.time() - t0
    print(f"Extracted {len(patches)} patches")
    t0 = time.time()
    patch_normals, patch_features, patch_centers, pca_time = run_patchnet(
        patches, coords, patchnet, config, device
    )
    timings["pca"] = pca_time
    timings["patchnet"] = time.time() - t0 - pca_time
    patchnet_normals = compose_point_normals(len(coords), patches, patch_normals)
    t0 = time.time()
    same, opposite, edges = build_consistency_matrices(
        edgenet,
        patch_features,
        patch_centers,
        config.get("edge_consistency", {}).get("k", 10),
        device,
    )
    timings["edgenet"] = time.time() - t0
    t0 = time.time()
    opt_config = config.get("optimization", {})
    optimizer = FlipOptimizer(
        method=opt_config.get("method", "greedy"),
        max_iterations=opt_config.get("max_iterations", 20),
        time_limit=opt_config.get("time_limit", 300.0),
        miqp_server_host=opt_config.get("miqp_server_host", "192.168.8.19"),
        miqp_server_port=opt_config.get("miqp_server_port", 11111),
        miqp_timeout=opt_config.get(
            "miqp_timeout", opt_config.get("time_limit", 3000.0)
        ),
        socket_fallback=opt_config.get("socket_fallback"),
    )
    objective_before = compute_objective(
        np.zeros(len(patches), dtype=np.int8), same, opposite
    )
    flip_decisions, opt_stats = optimizer.solve(same, opposite)
    objective_after = compute_objective(flip_decisions, same, opposite)
    timings["optimization"] = time.time() - t0
    final_normals = compose_point_normals(
        len(coords), patches, patch_normals, flip_decisions
    )
    write_ply_ascii(
        str(output_dir / f"{model_name}.ply"), coords, normals=final_normals
    )
    accuracy = None
    patchnet_accuracy = None
    patchnet_patch_stats = None
    normal_metrics = None
    patchnet_normal_metrics = None
    edge_accuracy = None
    flip_accuracy = None
    if gt_normals is not None:
        patchnet_accuracy, patchnet_patch_stats = calculate_patch_accuracy_stats(
            patches, patch_normals, gt_normals
        )
        patchnet_correct = (patchnet_normals * gt_normals).sum(axis=1) > 0
        patchnet_point_accuracy = max(
            float(patchnet_correct.mean()), 1.0 - float(patchnet_correct.mean())
        )
        patchnet_oriented = (
            -patchnet_normals if patchnet_correct.mean() < 0.5 else patchnet_normals
        )
        patchnet_normal_metrics = calculate_normal_metrics(
            torch.from_numpy(patchnet_oriented), torch.from_numpy(gt_normals)
        )
        print(
            f"PatchNet patch-mean accuracy: {patchnet_accuracy * 100:.2f}% "
            f"(point={patchnet_point_accuracy * 100:.2f}%)"
        )
        gt_patch_flip = calculate_patch_flip_labels(patches, patch_normals, gt_normals)
        edge_accuracy_percent, edge_count = calculate_edge_accuracy(
            same, opposite, gt_patch_flip
        )
        if edge_count > 0:
            edge_accuracy = edge_accuracy_percent / 100.0
        if len(gt_patch_flip) == len(flip_decisions) and len(gt_patch_flip) > 0:
            raw_flip_accuracy = float(
                (flip_decisions.astype(np.int8) == gt_patch_flip).mean()
            )
            flip_accuracy = max(raw_flip_accuracy, 1.0 - raw_flip_accuracy)
        if edge_accuracy is not None:
            print(f"Edge accuracy: {edge_accuracy * 100:.2f}%")
        if flip_accuracy is not None:
            print(f"Flip accuracy: {flip_accuracy * 100:.2f}%")
        correct = (final_normals * gt_normals).sum(axis=1) > 0
        accuracy = max(float(correct.mean()), 1.0 - float(correct.mean()))
        oriented = -final_normals if correct.mean() < 0.5 else final_normals
        normal_metrics = calculate_normal_metrics(
            torch.from_numpy(oriented), torch.from_numpy(gt_normals)
        )
        print(f"Orientation accuracy: {accuracy * 100:.2f}%")
    timings["total"] = time.time() - start_total
    metrics = {
        "model_name": model_name,
        "num_points": len(coords),
        "num_patches": len(patches),
        "num_edges": len(edges),
        "patchnet_accuracy": patchnet_accuracy,
        "patchnet_patch_stats": patchnet_patch_stats,
        "edge_accuracy": edge_accuracy,
        "flip_accuracy": flip_accuracy,
        "accuracy": accuracy,
        "patchnet_normal_metrics": patchnet_normal_metrics,
        "normal_metrics": normal_metrics,
        "timing": timings,
        "optimization": {
            "objective_before": objective_before,
            "objective_after": objective_after,
            **opt_stats,
        },
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(convert_to_serializable(metrics), handle, indent=2)
    print(f"Done in {timings['total']:.2f}s")
    return metrics


def save_summary_csv(csv_path, results):
    fieldnames = [
        "model_name",
        "scene_name",
        "input_path",
        "output_dir",
        "status",
        "num_points",
        "num_patches",
        "patchnet_accuracy",
        "edge_accuracy",
        "flip_accuracy",
        "accuracy",
        "time_total",
        "error",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model_name": result.get("model_name"),
                    "scene_name": result.get("scene_name"),
                    "input_path": result.get("input_path"),
                    "output_dir": result.get("output_dir"),
                    "status": result.get("status"),
                    "num_points": result.get("num_points"),
                    "num_patches": result.get("num_patches"),
                    "patchnet_accuracy": result.get("patchnet_accuracy"),
                    "edge_accuracy": result.get("edge_accuracy"),
                    "flip_accuracy": result.get("flip_accuracy"),
                    "accuracy": result.get("accuracy"),
                    "time_total": result.get("timing", {}).get("total"),
                    "error": result.get("error"),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference/base_config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--input-mode",
        default="auto",
        choices=["auto", "file", "scene_dir", "scan_root"],
    )
    parser.add_argument("--pattern", default="*_raw_pointcloud.ply")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-chunk", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing-metrics", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    config = load_config(args.config)
    if args.input:
        config["input"] = args.input
    input_path = Path(config["input"])
    output_root = (
        Path(args.output)
        if args.output
        else Path("outputs/inference") / Path(args.config).stem
    )
    targets, skipped, mode = resolve_input_targets(
        input_path, args.input_mode, args.pattern, args.recursive
    )
    print(f"Input mode: {mode}")
    print(f"Discovered {len(targets)} file(s)")
    if args.num_shards > 1:
        targets = [
            target
            for target_idx, target in enumerate(targets)
            if target_idx % args.num_shards == args.shard_index
        ]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(targets)} file(s)"
        )
    if args.dry_run:
        for target in targets:
            print(target["ply_path"])
        return
    device_name = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        and config["inference"].get("device", "cuda") == "cuda"
        else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        memory_fraction = config.get("inference", {}).get("gpu_memory_fraction")
        if memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device)
            print(f"CUDA memory fraction limit: {float(memory_fraction):.3f}")
    patchnet = load_patchnet(config, device)
    edgenet = load_edgenet(config, device)
    results = list(skipped)
    for idx, target in enumerate(targets, 1):
        output_dir = output_root / target["output_name"]
        if args.skip_existing_metrics and (output_dir / "metrics.json").exists():
            results.append({**target, "status": "skipped", "error": "metrics exists"})
            continue
        if output_dir.exists() and not config["inference"].get("overwrite", True):
            results.append({**target, "status": "skipped", "error": "output exists"})
            continue
        print(f"\n[{idx}/{len(targets)}] {target['ply_path']}")
        try:
            metrics = process_single_file(
                target["ply_path"],
                output_dir,
                config,
                patchnet,
                edgenet,
                device,
                args.no_chunk,
            )
            results.append(
                {
                    **metrics,
                    "scene_name": target.get("scene_name"),
                    "input_path": str(target["ply_path"]),
                    "output_dir": str(output_dir),
                    "status": "success",
                }
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "model_name": target["ply_path"].stem,
                    "scene_name": target.get("scene_name"),
                    "input_path": str(target["ply_path"]),
                    "output_dir": str(output_dir),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    if len(results) > 1 and not args.no_summary:
        output_root.mkdir(parents=True, exist_ok=True)
        save_summary_csv(output_root / "summary.csv", results)


if __name__ == "__main__":
    main()
