"""
Orientation Solver Module

Implements ForestFlip + VoteFlip algorithms for consistent normal orientation.
Based on graph-based optimization using spanning tree forests.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import deque


# === Data Structures ===

class Node:
    """Flipable node with orientation state."""

    def __init__(self, node_id: int):
        self.id = node_id
        self.inv_time = 0  # Flip counter

    def is_flipped(self) -> bool:
        """Check if currently flipped."""
        return self.inv_time % 2 == 1

    def flip(self):
        """Toggle flip state."""
        self.inv_time += 1


class Edge:
    """Edge with orientation-dependent weights."""

    def __init__(self, start: Node, end: Node, weight: float, inv_weight: float):
        self.start = start
        self.end = end
        self.weight = weight          # Cost when nodes have same orientation
        self.inv_weight = inv_weight  # Cost when nodes have different orientations

    def get_current_weight(self) -> float:
        """Get active weight based on current node states."""
        same_orientation = (self.start.is_flipped() == self.end.is_flipped())
        return self.weight if same_orientation else self.inv_weight


class Tree:
    """Tree node for spanning tree forest."""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.children: List[Tree] = []

    def add_child(self, child: 'Tree'):
        """Add child tree."""
        self.children.append(child)

    def get_all_nodes(self) -> List[int]:
        """Get all node IDs in this subtree."""
        result = [self.node_id]
        for child in self.children:
            result.extend(child.get_all_nodes())
        return result


class UnionFind:
    """Union-Find data structure for connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if actually merged."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


# === OrientationGraph ===

class OrientationGraph:
    """Graph for orientation optimization."""

    def __init__(self, n_nodes: int, consistency_matrix: np.ndarray, inconsistency_matrix: np.ndarray):
        """
        Initialize graph from consistency matrices.

        Args:
            n_nodes: Number of nodes
            consistency_matrix: (N, N) agreement counts (A[i,j])
            inconsistency_matrix: (N, N) disagreement counts (B[i,j])
        """
        self.n_nodes = n_nodes
        self.nodes = [Node(i) for i in range(n_nodes)]

        # Build edge list
        self.edges: List[Edge] = []
        self.adj_matrix = {}  # (i, j) -> Edge

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue

                # Create edge: weight = cost for same orientation
                # In consistency matrix, higher value means they should be same
                # So cost = inconsistency value (we want to minimize)
                weight = inconsistency_matrix[i, j]
                inv_weight = consistency_matrix[i, j]

                if weight > 0 or inv_weight > 0:
                    edge = Edge(self.nodes[i], self.nodes[j], weight, inv_weight)
                    self.edges.append(edge)
                    self.adj_matrix[(i, j)] = edge

    def reset_flips(self):
        """Reset all nodes to unflipped state."""
        for node in self.nodes:
            node.inv_time = 0

    def apply_flips(self, flip_labels: np.ndarray):
        """Apply flip labels to nodes."""
        for node, should_flip in zip(self.nodes, flip_labels):
            if should_flip != node.is_flipped():
                node.flip()

    def get_flip_states(self) -> np.ndarray:
        """Get current flip states."""
        return np.array([node.is_flipped() for node in self.nodes])

    def calc_total_weight(self) -> float:
        """Sum of all active edge weights."""
        return sum(edge.get_current_weight() for edge in self.edges)

    def get_connected_components(self) -> List[List[int]]:
        """Find connected components where bidirectional edges exist."""
        uf = UnionFind(self.n_nodes)

        # Only connect nodes with bidirectional edges
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if (i, j) in self.adj_matrix and (j, i) in self.adj_matrix:
                    uf.union(i, j)

        # Group by root
        components = {}
        for i in range(self.n_nodes):
            root = uf.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        return list(components.values())

    def get_random_spanning_forest(self) -> List[Tree]:
        """Generate random spanning tree for each connected component."""
        components = self.get_connected_components()
        forest = []

        for component in components:
            if len(component) == 1:
                # Single node component
                forest.append(Tree(component[0]))
                continue

            # Get edges within this component
            comp_edges = []
            for i in component:
                for j in component:
                    if i != j and (i, j) in self.adj_matrix:
                        comp_edges.append((i, j, self.adj_matrix[(i, j)]))

            # Randomized Kruskal
            random.shuffle(comp_edges)

            uf = UnionFind(self.n_nodes)
            tree_edges = []

            for i, j, edge in comp_edges:
                if uf.union(i, j):
                    tree_edges.append((i, j))
                    if len(tree_edges) == len(component) - 1:
                        break

            # Build tree structure from edges
            # Choose random root
            root_id = random.choice(component)
            tree = self._build_tree_from_edges(root_id, tree_edges, component)
            forest.append(tree)

        return forest

    def _build_tree_from_edges(self, root_id: int, edges: List[Tuple[int, int]], nodes: List[int]) -> Tree:
        """Build tree structure from edge list via BFS."""
        # Build adjacency list (undirected)
        adj = {node: [] for node in nodes}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # BFS from root to build tree
        visited = set([root_id])
        queue = deque([root_id])
        tree_nodes = {root_id: Tree(root_id)}

        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    tree_nodes[neighbor] = Tree(neighbor)
                    tree_nodes[curr].add_child(tree_nodes[neighbor])

        return tree_nodes[root_id]


# === ForestFlipSolver ===

class ForestFlipSolver:
    """Forest-based flip solver using spanning trees."""

    def flip(self, graph: OrientationGraph) -> np.ndarray:
        """Run ForestFlip algorithm once."""
        # Get random spanning forest
        trees = graph.get_random_spanning_forest()

        # Post-order traverse and align
        for tree in trees:
            self._post_order_align(tree, graph)

        # Return flip states
        return graph.get_flip_states()

    def _post_order_align(self, tree: Tree, graph: OrientationGraph):
        """Post-order traversal with alignment."""
        # Recurse to children first
        for child in tree.children:
            self._post_order_align(child, graph)

        # Align children to root
        if len(tree.children) > 0:
            self._align_children_to_root(tree, graph)

    def _align_children_to_root(self, tree: Tree, graph: OrientationGraph):
        """Align child subtrees to minimize weight."""
        # Group nodes: [root] + [child1_nodes] + [child2_nodes] + ...
        groups = [[tree.node_id]] + [child.get_all_nodes() for child in tree.children]

        # Brute force solve mini problem
        best_flips = self._brute_force_groups(groups, graph)

        # Apply flips
        for group_idx, should_flip in enumerate(best_flips):
            if should_flip:
                for node_id in groups[group_idx]:
                    graph.nodes[node_id].flip()

    def _brute_force_groups(self, groups: List[List[int]], graph: OrientationGraph) -> List[bool]:
        """Brute force search for best group flips."""
        n_groups = len(groups)
        best_weight = float('inf')
        best_flips = [False] * n_groups

        # Try all 2^n combinations
        for mask in range(2 ** n_groups):
            # Apply flips
            flips = [(mask >> i) & 1 for i in range(n_groups)]
            for group_idx, should_flip in enumerate(flips):
                if should_flip:
                    for node_id in groups[group_idx]:
                        graph.nodes[node_id].flip()

            # Calculate weight
            weight = self._calc_inter_group_weight(groups, graph)

            # Update best
            if weight < best_weight:
                best_weight = weight
                best_flips = flips.copy()

            # Restore state
            for group_idx, should_flip in enumerate(flips):
                if should_flip:
                    for node_id in groups[group_idx]:
                        graph.nodes[node_id].flip()

        return best_flips

    def _calc_inter_group_weight(self, groups: List[List[int]], graph: OrientationGraph) -> float:
        """Calculate total weight between groups."""
        total_weight = 0.0

        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                if i == j:
                    continue
                for node_i in group_i:
                    for node_j in group_j:
                        if (node_i, node_j) in graph.adj_matrix:
                            edge = graph.adj_matrix[(node_i, node_j)]
                            total_weight += edge.get_current_weight()

        return total_weight


# === VoteFlipSolver ===

class VoteFlipSolver:
    """Vote-based flip solver using multiple ForestFlip runs."""

    def __init__(self, n_iterations: int = 10):
        """
        Initialize solver.

        Args:
            n_iterations: Number of ForestFlip runs for voting
        """
        self.n_iterations = n_iterations
        self.base_solver = ForestFlipSolver()

    def solve(
        self,
        consistency_matrix: np.ndarray,
        inconsistency_matrix: np.ndarray = None,
        method: str = 'forest'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve orientation problem via voting.

        Args:
            consistency_matrix: (N, N) agreement counts
            inconsistency_matrix: (N, N) disagreement counts (if None, use consistency)
            method: 'forest' for ForestFlip (only option for now)

        Returns:
            labels: (N,) boolean array, True = flip
            metrics: Dict with 'weight_sum' and 'n_iterations'
        """
        # Default inconsistency to consistency
        if inconsistency_matrix is None:
            inconsistency_matrix = consistency_matrix

        # Build graph
        n_nodes = len(consistency_matrix)
        graph = OrientationGraph(n_nodes, consistency_matrix, inconsistency_matrix)

        # Run multiple times
        all_results = []
        for i in range(self.n_iterations):
            graph.reset_flips()
            flip_labels = self.base_solver.flip(graph)
            all_results.append(flip_labels)

        # Align all to first (handle global ambiguity)
        for i in range(1, len(all_results)):
            all_results[i] = self._align_to_first(all_results[0], all_results[i])

        # Vote
        votes = np.array(all_results).sum(axis=0)
        final_labels = votes > self.n_iterations / 2

        # Apply and get metrics
        graph.reset_flips()
        graph.apply_flips(final_labels)
        metrics = {
            'weight_sum': graph.calc_total_weight(),
            'n_iterations': self.n_iterations
        }

        return final_labels, metrics

    def _align_to_first(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Align target to reference (handle global flip ambiguity)."""
        agree = np.sum(reference == target)
        disagree = np.sum(reference != target)
        return target if agree >= disagree else ~target
