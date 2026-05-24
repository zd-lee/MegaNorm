import socket
import numpy as np
import json
import torch
import threading
import queue
import time
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.direct_orientation_model import create_direct_orientation_model
from dataset.dataset import estimate_normals_torch
from utils.config import load_config


@dataclass
class QueueEntry:
    """Queue entry containing client request data"""
    conn: socket.socket
    addr: tuple
    xyz_data: np.ndarray
    function_name: str
    function_config: dict
    timestamp: float


def load_model(checkpoint_path, config_path, device):
    """Load model from checkpoint"""
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    print("Creating model...")
    model = create_direct_orientation_model(config)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.to(device)

    test_config = config['data']['test']
    print(f"Model loaded successfully on {device}")
    return model, config


def process_batch(queue_entries, model, config, device):
    """Process a batch of requests with iterative refinement"""
    # Validate all requests use supported method
    for entry in queue_entries:
        if entry.function_name != 'direct_orientation':
            print(f"Warning: Unsupported method {entry.function_name} for {entry.addr}")

    num_iterations = config['iterative']['num_iterations']
    pca_max_nn = config['data']['train'].get('pca_max_nn', 30)

    all_point_data = []

    for entry in queue_entries:
        # Build initial point_data dict
        point_data = {
            'coord': torch.from_numpy(entry.xyz_data).float(),
            'feat': torch.from_numpy(entry.xyz_data).float(),
        }

        # PCA normal estimation
        xyz_with_normals = estimate_normals_torch(point_data['coord'], max_nn=pca_max_nn)
        normals = torch.from_numpy(xyz_with_normals[:, 3:6]).float()

        point_data['normals'] = normals
        all_point_data.append(point_data)

    # Step 2: Build batch with offset tensor
    all_coords = [pd['coord'] for pd in all_point_data]
    all_normals = [pd['normals'] for pd in all_point_data]

    coords_batch = torch.cat(all_coords, dim=0).to(device)
    normals_batch = torch.cat(all_normals, dim=0).to(device)

    offsets = []
    cumsum = 0
    for coords in all_coords:
        cumsum += len(coords)
        offsets.append(cumsum)
    offsets_tensor = torch.tensor(offsets, dtype=torch.long, device=device)

    # Step 3: Iterative refinement
    x_old = normals_batch.clone()
    conf_old = torch.zeros(x_old.shape[0], 1, device=device)

    with torch.no_grad():
        for _ in range(num_iterations):
            # Build 7D features: [xyz(3) + normals(3) + confidence(1)]
            feat = torch.cat([coords_batch, x_old, conf_old], dim=1)

            # Prepare point_data dict
            model_input = {
                'coord': coords_batch,
                'feat': feat,
                'offset': offsets_tensor,
                'grid_size': config['data']['test']['grid_size']
            }


            # Forward pass
            logits = model(model_input)[:, 0]

            # Apply flip operation
            flip_prob = torch.sigmoid(logits)
            flip_mask = flip_prob > 0.5
            x_new = x_old.clone()
            x_new[flip_mask] = -x_new[flip_mask]

            # Update confidence
            conf_new = (torch.abs(flip_prob - 0.5) * 2).unsqueeze(1)

            # Update for next iteration
            x_old = x_new
            conf_old = conf_new

    # Step 4: Split results back to individual samples
    results = []
    start_idx = 0
    for i, entry in enumerate(queue_entries):
        end_idx = offsets[i]
        xyz = entry.xyz_data
        normals = x_old[start_idx:end_idx].cpu().numpy()
        result = np.concatenate([xyz, normals], axis=1)  # (N, 6)
        results.append((entry.conn, result))
        start_idx = end_idx

    return results


def send_result(conn, result):
    """Send result to client"""
    try:
        conn.sendall(result.astype(np.float64).tobytes())
    except Exception as e:
        print(f"Error sending result: {e}")
    finally:
        conn.close()


def send_error(conn):
    """Send error response to client"""
    try:
        conn.sendall(json.dumps({"status": "ERROR"}).encode())
    except:
        pass
    finally:
        conn.close()


def batch_processor_thread(request_queue, model, config, device, batch_size, timeout):
    """Monitor queue and process batches"""
    print(f"Batch processor started (batch_size={batch_size}, timeout={timeout}s)")

    while True:
        batch = []
        deadline = time.time() + timeout

        # Collect batch entries
        while len(batch) < batch_size:
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break

            try:
                entry = request_queue.get(timeout=remaining_time)
                batch.append(entry)
            except queue.Empty:
                break

        # Process batch if not empty
        if batch:
            print(f"Processing batch of {len(batch)} requests")
            try:
                results = process_batch(batch, model, config, device)

                # Send results back to clients
                for conn, result in results:
                    send_result(conn, result)

                print(f"Batch processed successfully")

            except Exception as e:
                print(f"Batch processing error: {e}")
                import traceback
                traceback.print_exc()

                # Send error to all clients in batch
                for entry in batch:
                    send_error(entry.conn)


def handle_client(conn, addr, request_queue):
    """Handle client connection and enqueue request"""
    print(f"Connected by {addr}")
    try:
        # 1. Receive metadata
        req = conn.recv(1024)
        req = json.loads(req.decode())
        print(f"Request from {addr}: {req}")

        # Extract fields (compatible with socket_example.py protocol)
        data_size = req['data_size']
        function_name = req.get('function_name', 'direct_orientation')
        function_config = req.get('function_config', {})

        # Validate function_name
        if function_name != 'direct_orientation':
            print(f"Warning: Unsupported function '{function_name}', using 'direct_orientation'")

        # 2. Send ACK
        conn.sendall(json.dumps({"status": "OK"}).encode())

        # 3. Receive point cloud data
        data_buffer_size = data_size * 24  # float64, 3 channels
        data = b''
        data_recv = 0

        while data_recv < data_buffer_size:
            chunk = conn.recv(data_buffer_size - data_recv)
            if not chunk:
                break
            data_recv += len(chunk)
            data += chunk

        print(f"Received {len(data)} bytes from {addr}")

        if len(data) != data_buffer_size:
            print(f"Data size mismatch. Expected {data_buffer_size}, got {len(data)}")
            send_error(conn)
            return

        # 4. Parse XYZ data
        xyz_data = np.frombuffer(data, dtype=np.float64).reshape(-1, 3)

        # 5. Enqueue request
        entry = QueueEntry(
            conn=conn,
            addr=addr,
            xyz_data=xyz_data,
            function_name=function_name,
            function_config=function_config,
            timestamp=time.time()
        )
        request_queue.put(entry)
        print(f"Request from {addr} queued ({len(xyz_data)} points, method: {function_name})")

        # Note: Connection stays open, result will be sent by batch processor

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
        import traceback
        traceback.print_exc()
        send_error(conn)


def main():
    parser = argparse.ArgumentParser(description='Socket server for normal estimation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/direct_orientation/iterative.yaml',
                        help='Path to config file')
    parser.add_argument('--port', type=int, default=8999, help='Server port')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--timeout', type=float, default=2.0, help='Timeout in seconds')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    args = parser.parse_args()

    # Setup device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)

    # Create request queue
    request_queue = queue.Queue()

    # Start batch processor thread
    processor = threading.Thread(
        target=batch_processor_thread,
        args=(request_queue, model, config, device, args.batch_size, args.timeout),
        daemon=True
    )
    processor.start()

    # Listen for connections
    HOST = '0.0.0.0'
    PORT = args.port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            # Spawn thread to handle connection
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, request_queue),
                daemon=False
            )
            t.start()


if __name__ == "__main__":
    main()
