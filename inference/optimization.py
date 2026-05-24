"""
Optimization Module

Implements Step 6 of DACPO algorithm:
- Solve 0-1 optimization to find optimal patch flip labels
- Maximize consistency between overlapping patches
"""

from pickletools import optimize
import numpy as np
import logging
import socket
import json
import struct
from typing import Tuple, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)



class FlipOptimizer:
    """
    Solves the 0-1 optimization problem to determine patch flip labels.
    """

    def __init__(
        self,
        miqp_server_host: str = '192.168.8.19',
        miqp_server_port: int = 11111,
        miqp_timeout: float = 3000.0
    ):
        """
        Initialize the FlipOptimizer.

        Args:
            method: Optimization method ('miqp', 'greedy', 'spectral', 'connected_components')
            max_time: Maximum time for optimization (seconds)
            miqp_server_host: MIQP server host address
            miqp_server_port: MIQP server port
            miqp_timeout: Timeout for MIQP server connection (seconds)
        """
        self.miqp_server_host = miqp_server_host
        self.miqp_server_port = miqp_server_port
        self.miqp_timeout = miqp_timeout
        logger.info(f"MIQP server: {miqp_server_host}:{miqp_server_port}")

    def solve(
        self,
        consistency_matrix: np.ndarray,
        inconsistency_matrix: np.ndarray = None,
        method: str = 'miqp',
        init_values: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve the flip optimization problem.

        The objective is to assign binary labels x_i ∈ {0, 1} to each patch i
        such that we maximize:
            Σ(i,j) consistency[i,j] * (1 - |x_i - x_j|)

        This means:
        - If x_i == x_j, we get consistency[i,j] (they agree)
        - If x_i != x_j, we get 0 (they disagree)

        Args:
            consistency_matrix: Consistency scores (N_q, N_q)
            inconsistency_matrix: Inconsistency scores (N_q, N_q), optional

        Returns:
            patch_flip_labels: Binary flip labels for patches (N_q,)
            stats: Dictionary with optimization statistics
        """
        N_q = consistency_matrix.shape[0]

        logger.info(f"Solving flip optimization for {N_q} patches using {method}")

        if method == 'miqp':
            labels, stats = self._solve_miqp(consistency_matrix, inconsistency_matrix, init_values)
        elif method == 'trival':
            labels, stats = self._solve_trival(consistency_matrix, inconsistency_matrix)
        else:
            logger.warning(f"Unknown method {method}, using trival")
            labels, stats = self._solve_trival(consistency_matrix, inconsistency_matrix)

        logger.info(f"Optimization completed: {stats}")

        return labels, stats

    def _solve_trival(
        self,
        consistency_matrix: np.ndarray,
        inconsistency_matrix: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Trivial solver: Assign all patches to label 0.

        This is a fallback method when no optimization is performed.
        """
        N_q = consistency_matrix.shape[0]
        labels = np.random.randint(0, 2, size=N_q)

        objective = compute_objective(labels, consistency_matrix, inconsistency_matrix)

        stats = {
            'method': 'trival',
            'objective': objective
        }

        return labels, stats

    def _solve_miqp(
        self,
        consistency_matrix: np.ndarray,
        inconsistency_matrix: np.ndarray = None,
        init_values: np.ndarray = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        MIQP solver: Send optimization problem to MIQP server using Gurobi.

        The server expects two matrices:
        - A[i,j]: cost when patch i and j have the same flip state
        - B[i,j]: cost when patch i and j have different flip states

        Protocol:
        1. Send metadata (data size)
        2. Receive acknowledgment
        3. Send flattened data: [A.flatten(), B.flatten()]
        4. Receive binary result array
        """
        N_q = consistency_matrix.shape[0]

        # Prepare matrices A and B
        # A[i,j] = consistency when x[i] == x[j] (same flip state)
        A = consistency_matrix.copy()

        # B[i,j] = consistency when x[i] != x[j] (different flip state)
        if inconsistency_matrix is not None:
            B = inconsistency_matrix.copy()
        else:
            # If no inconsistency matrix provided, assume B = 0
            B = np.zeros_like(A)

        # Set diagonal to zero (no self-consistency)
        np.fill_diagonal(A, 0)
        np.fill_diagonal(B, 0)

        # Convert to sparse format
        sparse_edges, num_nonzero = _convert_to_sparse_edges(A, B)
        logger.info(f"Using sparse format: {num_nonzero} edges ({num_nonzero/(N_q*N_q)*100:.2f}% density)")

        logger.info(f"Connecting to MIQP server at {self.miqp_server_host}:{self.miqp_server_port}")

        try:
            # Create socket connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.miqp_timeout)
                s.connect((self.miqp_server_host, self.miqp_server_port))

                # Step 1: Send metadata
                metadata = {
                    'matrix_format': 'sparse',
                    'matrix_dim': N_q,
                    'num_nonzero': num_nonzero,
                    'data_size': num_nonzero * 6,
                    'has_init_values': init_values is not None
                }
                metadata_json = json.dumps(metadata).encode()
                s.sendall(metadata_json)
                logger.debug(f"Sent metadata: {metadata}")

                # Step 2: Receive acknowledgment
                ack = s.recv(1024)
                ack_data = json.loads(ack.decode())
                logger.debug(f"Received acknowledgment: {ack_data}")

                if ack_data.get('status') != 'OK':
                    raise RuntimeError(f"Server returned error status: {ack_data}")

                # Step 3: Send sparse edge data
                data_bytes = sparse_edges.tobytes()
                total_sent = 0
                while total_sent < len(data_bytes):
                    sent = s.send(data_bytes[total_sent:])
                    if sent == 0:
                        raise RuntimeError("Socket connection broken")
                    total_sent += sent
                logger.info(f"Sent {total_sent} bytes of sparse data")

                # Step 3.5: Send initialization values if provided
                if init_values is not None:
                    init_values_int32 = init_values.astype(np.int32)
                    init_bytes = init_values_int32.tobytes()
                    total_sent = 0
                    while total_sent < len(init_bytes):
                        sent = s.send(init_bytes[total_sent:])
                        if sent == 0:
                            raise RuntimeError("Socket connection broken")
                        total_sent += sent
                    logger.info(f"Sent {total_sent} bytes of initialization values")

                # Step 4: Receive timing info length (4 bytes)
                timing_len_bytes = b''
                while len(timing_len_bytes) < 4:
                    chunk = s.recv(4 - len(timing_len_bytes))
                    if not chunk:
                        raise RuntimeError("Connection closed before receiving timing length")
                    timing_len_bytes += chunk
                timing_len = struct.unpack('I', timing_len_bytes)[0]
                logger.debug(f"Timing JSON length: {timing_len}")

                # Receive and parse timing JSON
                timing_json_bytes = b''
                while len(timing_json_bytes) < timing_len:
                    chunk = s.recv(timing_len - len(timing_json_bytes))
                    if not chunk:
                        raise RuntimeError("Connection closed before receiving timing info")
                    timing_json_bytes += chunk
                response_data = json.loads(timing_json_bytes.decode())
                logger.debug(f"Received timing info: {response_data.get('timing')}")

                # Get result size from response
                result_size = response_data['result_size']
                logger.debug(f"Expecting {result_size} result values")

                # Step 5: Receive result
                result_bytes = b''
                expected_size = result_size * 4  # int32 = 4 bytes per element
                while len(result_bytes) < expected_size:
                    chunk = s.recv(expected_size - len(result_bytes))
                    if not chunk:
                        raise RuntimeError("Connection closed before receiving full result")
                    result_bytes += chunk

                # Parse result
                labels = np.frombuffer(result_bytes, dtype=np.int32)
                logger.info(f"Received solution from MIQP server: {len(labels)} labels")

                # Compute objective value
                objective = compute_objective(labels, consistency_matrix,inconsistency_matrix)

                stats = {
                    'method': 'miqp',
                    'objective': objective,
                    'server': f"{self.miqp_server_host}:{self.miqp_server_port}"
                }

                return labels, stats

        except socket.timeout:
            logger.error(f"MIQP server connection timeout after {self.miqp_timeout}s")
            logger.warning("Falling back to greedy solver")
            return self._solve_greedy(consistency_matrix, inconsistency_matrix)

        except (socket.error, ConnectionRefusedError) as e:
            logger.error(f"Failed to connect to MIQP server: {e}")
            logger.warning("Falling back to greedy solver")
            return self._solve_greedy(consistency_matrix, inconsistency_matrix)

        except Exception as e:
            logger.error(f"MIQP solver error: {e}")
            logger.warning("Falling back to greedy solver")
            return self._solve_greedy(consistency_matrix, inconsistency_matrix)

    def _compute_objective(
        self,
        labels: np.ndarray,
        consistency_matrix: np.ndarray,
        inconsistency_matrix: np.ndarray
    ):
        return compute_objective(labels, consistency_matrix, inconsistency_matrix)

def compute_objective(
    labels: np.ndarray,
    consistency_matrix: np.ndarray,
    inconsistency_matrix: np.ndarray
) -> float:
    """
    Compute the objective function value (vectorized).

    Objective: Σ(i,j) consistency[i,j] * (1 - |x_i - x_j|)

    Vectorized implementation for faster computation on large matrices.
    """
    # Create boolean mask: True where labels are equal (N_q x N_q)
    labels_equal = labels[:, None] == labels[None, :]

    # Select from consistency or inconsistency matrix based on label equality
    selected_values = np.where(labels_equal, consistency_matrix, inconsistency_matrix)

    # Extract upper triangle (excluding diagonal) to avoid double counting
    # triu with k=1 gives upper triangle excluding diagonal
    upper_triangle_mask = np.triu(np.ones_like(selected_values, dtype=bool), k=1)
    objective = selected_values[upper_triangle_mask].sum()

    return float(objective)


def _convert_to_sparse_edges(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Convert dense matrices A and B to sparse edge list.
    Only extracts upper triangle (undirected graph).

    Args:
        A: Consistency matrix (N x N)
        B: Inconsistency matrix (N x N)

    Returns:
        sparse_edges: Structured array with fields (i, j, w, inv_w)
        num_edges: Number of edges
    """
    # Extract upper triangle indices where A is non-zero
    rows, cols = np.where(np.triu(A, k=1) != 0)

    # Create structured array
    sparse_dtype = np.dtype([
        ('i', np.uint32),
        ('j', np.uint32),
        ('w', np.float64),
        ('inv_w', np.float64)
    ])

    num_edges = len(rows)
    sparse_edges = np.zeros(num_edges, dtype=sparse_dtype)

    sparse_edges['i'] = rows.astype(np.uint32)
    sparse_edges['j'] = cols.astype(np.uint32)
    sparse_edges['w'] = A[rows, cols]
    sparse_edges['inv_w'] = B[rows, cols]

    return sparse_edges, num_edges

