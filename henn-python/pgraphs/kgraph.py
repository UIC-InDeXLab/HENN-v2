from pgraphs.base_pgraph import BaseProximityGraph
import numpy as np
import random
from typing import List, Tuple, Dict, Set


class KGraph(BaseProximityGraph):
    """
    KGraph implementation using NN-Descent algorithm.
    
    KGraph builds high-quality k-nearest neighbor graphs using the NN-Descent
    (Nearest Neighbor Descent) algorithm. This iterative algorithm starts with
    a random or approximate k-NN graph and progressively improves it by having
    nodes share information about their neighbors.
    
    The algorithm is particularly effective for large-scale datasets where
    brute-force k-NN construction would be too expensive.
    """

    def __init__(self):
        """Initialize KGraph."""
        self.init_node = None

    def build_graph(
        self, henn_points: np.ndarray, layer_indices: list, params: dict = None
    ) -> Dict[int, List[int]]:
        """
        Build KGraph using NN-Descent algorithm.
        
        Args:
            henn_points: Array of points
            layer_indices: Indices of points in this layer
            params: Parameters for KGraph construction:
                - k: Number of nearest neighbors (default: 10)
                - rho: Sample rate for NN-Descent (default: 1.0)
                - delta: Early termination threshold (default: 0.001)
                - max_iterations: Maximum iterations (default: 30)
                - recall_target: Target recall for quality (default: 0.9)
        
        Returns:
            Dictionary mapping node indices to neighbor lists
        """
        if params is None:
            params = {}

        k = params.get("k", 10)
        rho = params.get("rho", 1.0)  # Sample rate
        delta = params.get("delta", 0.001)  # Early termination threshold
        max_iterations = params.get("max_iterations", 30)
        recall_target = params.get("recall_target", 0.9)

        n = len(layer_indices)

        if n == 0:
            return {}

        if n == 1:
            return {layer_indices[0]: []}

        # Limit k to maximum possible neighbors
        k = min(k, n - 1)

        # Step 1: Initialize with random k-NN graph
        knn_graph = self._initialize_random_knn(henn_points, layer_indices, k)

        # Step 2: Run NN-Descent iterations
        final_graph = self._nn_descent(
            henn_points, layer_indices, knn_graph, k, rho, delta, max_iterations
        )

        # Step 3: Find best entry point
        self._find_entry_point(final_graph, layer_indices)

        return final_graph

    def _initialize_random_knn(
        self, henn_points: np.ndarray, layer_indices: list, k: int
    ) -> Dict[int, List[Tuple[float, int]]]:
        """
        Initialize with random k-NN graph.
        
        Returns graph with (distance, neighbor_idx) tuples for easy sorting.
        """
        n = len(layer_indices)
        knn_graph = {}

        for idx in layer_indices:
            # Randomly sample k neighbors (or n-1 if k > n-1)
            available_neighbors = [other_idx for other_idx in layer_indices if other_idx != idx]
            actual_k = min(k, len(available_neighbors))
            
            if actual_k > 0:
                neighbors = random.sample(available_neighbors, actual_k)
                
                # Calculate distances and store as (distance, neighbor_idx) tuples
                neighbor_list = []
                for neighbor_idx in neighbors:
                    dist = np.linalg.norm(henn_points[idx] - henn_points[neighbor_idx])
                    neighbor_list.append((dist, neighbor_idx))
                
                # Sort by distance
                neighbor_list.sort()
                knn_graph[idx] = neighbor_list
            else:
                knn_graph[idx] = []

        return knn_graph

    def _nn_descent(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        knn_graph: Dict[int, List[Tuple[float, int]]],
        k: int,
        rho: float,
        delta: float,
        max_iterations: int,
    ) -> Dict[int, List[int]]:
        """
        Run NN-Descent algorithm to iteratively improve the k-NN graph.
        """
        updates_count = 0
        
        for iteration in range(max_iterations):
            iteration_updates = 0
            
            # Create new and old neighbor sets for this iteration
            new_neighbors = {idx: [] for idx in layer_indices}
            old_neighbors = {idx: [] for idx in layer_indices}
            
            # Sample neighbors based on recency and randomness
            for node_idx in layer_indices:
                current_neighbors = knn_graph[node_idx]
                
                # Split into new and old neighbors (simplified: treat all as potentially new)
                # In practice, you'd track which neighbors were added recently
                for dist, neighbor_idx in current_neighbors:
                    if random.random() < rho:  # Sample with probability rho
                        new_neighbors[node_idx].append(neighbor_idx)
                    else:
                        old_neighbors[node_idx].append(neighbor_idx)
            
            # Main NN-Descent update step
            for node_idx in layer_indices:
                candidates = set()
                
                # Generate candidates from new neighbors
                for new_neighbor in new_neighbors[node_idx]:
                    # Add neighbors of new neighbors
                    for _, candidate_idx in knn_graph[new_neighbor]:
                        if candidate_idx != node_idx:
                            candidates.add(candidate_idx)
                
                # Generate candidates from old neighbors (with lower probability)
                for old_neighbor in old_neighbors[node_idx]:
                    if random.random() < rho:  # Sample old neighbors less frequently
                        for _, candidate_idx in knn_graph[old_neighbor]:
                            if candidate_idx != node_idx:
                                candidates.add(candidate_idx)
                
                # Add reverse candidates (nodes that might want this node as neighbor)
                for candidate_idx in layer_indices:
                    if candidate_idx != node_idx:
                        # Check if node_idx could be a good neighbor for candidate_idx
                        candidate_neighbors = [neighbor_idx for _, neighbor_idx in knn_graph[candidate_idx]]
                        if len(candidate_neighbors) < k:
                            candidates.add(candidate_idx)
                
                # Evaluate candidates and update if better
                current_neighbors = knn_graph[node_idx].copy()
                node_point = henn_points[node_idx]
                
                for candidate_idx in candidates:
                    candidate_dist = np.linalg.norm(henn_points[candidate_idx] - node_point)
                    
                    # Check if this candidate is better than current worst neighbor
                    if len(current_neighbors) < k:
                        current_neighbors.append((candidate_dist, candidate_idx))
                        current_neighbors.sort()
                        iteration_updates += 1
                    elif candidate_dist < current_neighbors[-1][0]:
                        # Replace worst neighbor
                        current_neighbors[-1] = (candidate_dist, candidate_idx)
                        current_neighbors.sort()
                        iteration_updates += 1
                
                knn_graph[node_idx] = current_neighbors[:k]
            
            # Check for convergence
            if iteration_updates == 0:
                print(f"NN-Descent converged after {iteration + 1} iterations")
                break
            
            updates_count += iteration_updates
            
            # Early termination based on update rate
            if iteration > 0 and iteration_updates < delta * len(layer_indices) * k:
                print(f"NN-Descent early termination after {iteration + 1} iterations")
                break
        
        # Convert back to simple adjacency list format
        final_graph = {}
        for node_idx, neighbors in knn_graph.items():
            final_graph[node_idx] = [neighbor_idx for _, neighbor_idx in neighbors]
        
        print(f"NN-Descent completed with {updates_count} total updates")
        return final_graph

    def _find_entry_point(self, graph: Dict[int, List[int]], layer_indices: list):
        """
        Find the best entry point for searches.
        
        For KGraph, we choose the node with highest degree (most connections).
        """
        max_degree = -1
        best_node = None

        for node_idx in layer_indices:
            degree = len(graph.get(node_idx, []))
            if degree > max_degree:
                max_degree = degree
                best_node = node_idx

        self.init_node = best_node if best_node is not None else layer_indices[0]

    def get_initial_search_node(
        self, henn_points: np.ndarray, layer_indices: list, edges: dict = None
    ) -> int:
        """
        Get the initial search node for this layer.
        
        Returns the entry point found during graph construction,
        or the highest degree node if edges are provided.
        """
        # Use stored entry point
        if self.init_node is not None and self.init_node in layer_indices:
            return self.init_node
        
        # Fallback to first node
        return layer_indices[0] if layer_indices else None

    def _calculate_recall(
        self,
        henn_points: np.ndarray,
        layer_indices: list,
        approximate_graph: Dict[int, List[int]],
        k: int,
        sample_size: int = 100,
    ) -> float:
        """
        Calculate recall of the approximate k-NN graph compared to true k-NN.
        
        This is used for quality assessment but is expensive, so we sample.
        """
        if len(layer_indices) <= sample_size:
            sample_indices = layer_indices
        else:
            sample_indices = random.sample(layer_indices, sample_size)
        
        total_recall = 0.0
        
        for node_idx in sample_indices:
            # Get approximate neighbors
            approx_neighbors = set(approximate_graph.get(node_idx, []))
            
            # Calculate true k-NN for this node
            node_point = henn_points[node_idx]
            distances = []
            
            for other_idx in layer_indices:
                if other_idx != node_idx:
                    dist = np.linalg.norm(henn_points[other_idx] - node_point)
                    distances.append((dist, other_idx))
            
            distances.sort()
            true_neighbors = set([neighbor_idx for _, neighbor_idx in distances[:k]])
            
            # Calculate recall for this node
            if len(true_neighbors) > 0:
                intersection = len(approx_neighbors.intersection(true_neighbors))
                recall = intersection / len(true_neighbors)
                total_recall += recall
        
        return total_recall / len(sample_indices)

    def evaluate_quality(
        self, henn_points: np.ndarray, layer_indices: list, graph: Dict[int, List[int]], k: int
    ) -> Dict[str, float]:
        """
        Evaluate the quality of the constructed graph.
        
        Returns metrics like recall, average degree, etc.
        """
        # Calculate basic statistics
        total_edges = sum(len(neighbors) for neighbors in graph.values())
        avg_degree = total_edges / len(graph) if graph else 0
        max_degree = max(len(neighbors) for neighbors in graph.values()) if graph else 0
        
        # Calculate recall (expensive, so use small sample)
        recall = self._calculate_recall(henn_points, layer_indices, graph, k, sample_size=50)
        
        return {
            "average_degree": avg_degree,
            "max_degree": max_degree,
            "total_edges": total_edges,
            "recall": recall,
            "target_degree": k,
        }
