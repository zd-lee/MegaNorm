import json
import socket
import struct

import numpy as np

from typing import Dict, Tuple

SOCKET_METHODS = {"miqp", "gurobi_socket", "socket_gurobi", "remote_gurobi"}


def compute_objective(
    labels: np.ndarray, same: np.ndarray, opposite: np.ndarray
) -> float:
    labels = labels.astype(np.int8)
    labels_equal = labels[:, None] == labels[None, :]
    selected = np.where(labels_equal, same, opposite)
    return float(selected[np.triu(np.ones_like(selected, dtype=bool), k=1)].sum())


class FlipOptimizer:
    def __init__(
        self,
        method: str = "greedy",
        max_iterations: int = 20,
        time_limit: float = 300.0,
        miqp_server_host: str = "192.168.8.19",
        miqp_server_port: int = 11111,
        miqp_timeout: float = 3000.0,
        socket_fallback: str = None,
    ):
        self.method = str(method).lower()
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.miqp_server_host = miqp_server_host
        self.miqp_server_port = int(miqp_server_port)
        self.miqp_timeout = miqp_timeout
        self.socket_fallback = socket_fallback

    def solve(self, same: np.ndarray, opposite: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.method in SOCKET_METHODS:
            try:
                return self._solve_miqp_socket(same, opposite)
            except Exception as exc:
                fallback = self._normalize_fallback(self.socket_fallback)
                if fallback is None:
                    raise
                if fallback == "ortools":
                    labels, stats = self._solve_ortools(same, opposite)
                elif fallback == "greedy":
                    labels, stats = self._solve_greedy(same, opposite)
                else:
                    raise ValueError(f"Unsupported socket fallback: {self.socket_fallback}")
                stats["requested_method"] = self.method
                stats["socket_fallback"] = fallback
                stats["socket_error"] = str(exc)
                return labels, stats
        if self.method == "gurobi":
            try:
                return self._solve_gurobi(same, opposite)
            except Exception as exc:
                if "Model too large for size-limited license" not in str(exc):
                    raise
                labels, stats = self._solve_ortools(same, opposite)
                stats["requested_method"] = "gurobi"
                stats["gurobi_fallback"] = "size_limited"
                stats["gurobi_error"] = str(exc)
                return labels, stats
        if self.method in {"ortools", "or-tools", "cp-sat", "cpsat"}:
            try:
                return self._solve_ortools(same, opposite)
            except ImportError:
                return self._solve_greedy(
                    same, opposite, fallback="ortools_not_installed"
                )
        return self._solve_greedy(same, opposite)

    @staticmethod
    def _normalize_fallback(fallback: str):
        if fallback is None:
            return None
        fallback = str(fallback).lower()
        if fallback in {"", "none", "raise", "error", "false"}:
            return None
        if fallback in {"ortools", "or-tools", "cp-sat", "cpsat"}:
            return "ortools"
        if fallback == "greedy":
            return "greedy"
        return fallback

    def _solve_greedy(
        self, same: np.ndarray, opposite: np.ndarray, fallback: str = None
    ) -> Tuple[np.ndarray, Dict]:
        labels = np.zeros(same.shape[0], dtype=np.int8)
        current = compute_objective(labels, same, opposite)
        sweeps = 0
        for sweeps in range(1, self.max_iterations + 1):
            improved = False
            for idx in range(len(labels)):
                labels[idx] ^= 1
                candidate = compute_objective(labels, same, opposite)
                if candidate > current:
                    current = candidate
                    improved = True
                else:
                    labels[idx] ^= 1
            if not improved:
                break
        stats = {"method": "greedy", "objective": current, "sweeps": sweeps}
        if fallback:
            stats["fallback"] = fallback
        return labels, stats

    def _solve_gurobi(
        self, same: np.ndarray, opposite: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        import gurobipy as gp
        from gurobipy import GRB

        n_nodes = same.shape[0]
        model = gp.Model("meganorm_flip")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.time_limit
        if n_nodes == 0:
            return np.zeros(0, dtype=np.int8), {
                "method": "gurobi",
                "objective": 0.0,
                "status": int(GRB.OPTIMAL),
            }
        flips = model.addVars(n_nodes, vtype=GRB.BINARY, name="flip")
        model.addConstr(flips[0] == 0)
        objective_terms = []
        rows, cols = np.where(np.triu((same + opposite) > 0, k=1))
        for src, dst in zip(rows, cols):
            different = flips[src] + flips[dst] - 2 * flips[src] * flips[dst]
            objective_terms.append(
                same[src, dst] * (1 - different) + opposite[src, dst] * different
            )
        model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi did not produce a solution, status={model.Status}")
        labels = np.array(
            [int(round(flips[idx].X)) for idx in range(n_nodes)], dtype=np.int8
        )
        objective = compute_objective(labels, same, opposite)
        return labels, {
            "method": "gurobi",
            "objective": objective,
            "status": int(model.Status),
        }

    def _solve_miqp_socket(
        self, same: np.ndarray, opposite: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        n_nodes = same.shape[0]
        if n_nodes == 0:
            return np.zeros(0, dtype=np.int8), {
                "method": "gurobi_socket",
                "objective": 0.0,
                "server": f"{self.miqp_server_host}:{self.miqp_server_port}",
                "num_edges": 0,
            }
        same = same.copy()
        opposite = opposite.copy()
        np.fill_diagonal(same, 0)
        np.fill_diagonal(opposite, 0)
        sparse_edges, num_edges = _convert_to_sparse_edges(same, opposite)
        if num_edges == 0:
            labels = np.zeros(n_nodes, dtype=np.int8)
            return labels, {
                "method": "gurobi_socket",
                "objective": 0.0,
                "server": f"{self.miqp_server_host}:{self.miqp_server_port}",
                "num_edges": 0,
            }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.miqp_timeout)
            sock.connect((self.miqp_server_host, self.miqp_server_port))
            metadata = {
                "matrix_format": "sparse",
                "matrix_dim": n_nodes,
                "num_nonzero": num_edges,
                "data_size": num_edges * 6,
                "has_init_values": False,
            }
            sock.sendall(json.dumps(metadata).encode())
            ack = json.loads(sock.recv(1024).decode())
            if ack.get("status") != "OK":
                raise RuntimeError(f"MIQP socket server returned error: {ack}")
            sock.sendall(sparse_edges.tobytes())
            timing_len = struct.unpack("I", _recv_exact(sock, 4))[0]
            response = json.loads(_recv_exact(sock, timing_len).decode())
            result_size = int(response["result_size"])
            result_bytes = _recv_exact(sock, result_size * 4)
            labels = np.frombuffer(result_bytes, dtype=np.int32).astype(np.int8)
        if len(labels) != n_nodes:
            raise RuntimeError(
                f"MIQP socket server returned {len(labels)} labels for {n_nodes} nodes"
            )
        if labels[0] == 1:
            labels = 1 - labels
        objective = compute_objective(labels, same, opposite)
        stats = {
            "method": "gurobi_socket",
            "requested_method": self.method,
            "objective": objective,
            "server": f"{self.miqp_server_host}:{self.miqp_server_port}",
            "num_edges": num_edges,
        }
        if "timing" in response:
            stats["server_timing"] = response["timing"]
        if "status" in response:
            stats["status"] = response["status"]
        return labels, stats

    def _solve_ortools(
        self, same: np.ndarray, opposite: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        from ortools.sat.python import cp_model

        n_nodes = same.shape[0]
        rows, cols = np.where(np.triu((same + opposite) > 0, k=1))
        if n_nodes == 0:
            return np.zeros(0, dtype=np.int8), {
                "method": "ortools",
                "objective": 0.0,
                "status": "empty",
            }
        model = cp_model.CpModel()
        flips = [model.new_bool_var(f"flip_{idx}") for idx in range(n_nodes)]
        model.add(flips[0] == 0)
        greedy_labels, _ = self._solve_greedy(same, opposite)
        if greedy_labels[0] == 1:
            greedy_labels = 1 - greedy_labels
        for var, value in zip(flips, greedy_labels):
            model.add_hint(var, int(value))
        different_vars = []
        weights = []
        scale = 1_000_000
        for edge_idx, (src, dst) in enumerate(zip(rows, cols)):
            different = model.new_bool_var(f"different_{edge_idx}")
            model.add(flips[src] + flips[dst] == 1).only_enforce_if(different)
            model.add(flips[src] == flips[dst]).only_enforce_if(different.Not())
            different_vars.append(different)
            weights.append(int(round((opposite[src, dst] - same[src, dst]) * scale)))
        model.maximize(sum(w * var for w, var in zip(weights, different_vars)))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = min(max(self.max_iterations, 1), 32)
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._solve_greedy(
                same, opposite, fallback=f"ortools_status_{status}"
            )
        labels = np.array([solver.value(var) for var in flips], dtype=np.int8)
        objective = compute_objective(labels, same, opposite)
        constant = float(same[rows, cols].sum())
        return labels, {
            "method": "ortools",
            "objective": objective,
            "status": solver.status_name(status),
            "best_objective_bound": constant + solver.best_objective_bound / scale,
            "wall_time": solver.wall_time,
        }


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("Socket connection closed before receiving full payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _convert_to_sparse_edges(same: np.ndarray, opposite: np.ndarray) -> Tuple[np.ndarray, int]:
    rows, cols = np.where(np.triu((same + opposite) > 0, k=1))
    sparse_dtype = np.dtype(
        [
            ("i", np.uint32),
            ("j", np.uint32),
            ("w", np.float64),
            ("inv_w", np.float64),
        ]
    )
    sparse_edges = np.zeros(len(rows), dtype=sparse_dtype)
    sparse_edges["i"] = rows.astype(np.uint32)
    sparse_edges["j"] = cols.astype(np.uint32)
    sparse_edges["w"] = same[rows, cols]
    sparse_edges["inv_w"] = opposite[rows, cols]
    return sparse_edges, len(rows)
