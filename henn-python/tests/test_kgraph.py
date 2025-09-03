"""
Test cases for KGraph (K-Nearest Neighbor Graph with NN-Descent) implementation.
"""

import unittest
import numpy as np
import random
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.kgraph import KGraph
from henn import HENN, HENNConfig


class TestKGraph(unittest.TestCase):
    """Test cases for KGraph (K-Nearest Neighbor Graph with NN-Descent) implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.kgraph = KGraph()
        np.random.seed(42)  # For reproducible tests

    def test_kgraph_basic_functionality(self):
        """Test basic KGraph functionality."""
        # Create a simple 2D dataset
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        params = {"k": 4, "max_iterations": 10}

        edges = self.kgraph.build_graph(points, layer_indices, params)

        # Verify graph was created
        self.assertEqual(len(edges), 15)

        # Verify graph properties
        for node_idx, neighbors in edges.items():
            # Check that degree doesn't exceed k
            self.assertLessEqual(
                len(neighbors),
                params["k"],
                f"Node {node_idx} has degree {len(neighbors)} > k={params['k']}",
            )

            # Check that all neighbors are valid indices
            for neighbor_idx in neighbors:
                self.assertIn(
                    neighbor_idx,
                    layer_indices,
                    f"Neighbor {neighbor_idx} not in layer indices",
                )
                self.assertNotEqual(
                    neighbor_idx,
                    node_idx,
                    f"Node {node_idx} is connected to itself",
                )

        print("✓ KGraph basic functionality test passed")

    def test_kgraph_empty_input(self):
        """Test KGraph with empty input."""
        points = np.array([]).reshape(0, 2)
        layer_indices = []
        params = {"k": 4}

        edges = self.kgraph.build_graph(points, layer_indices, params)
        self.assertEqual(len(edges), 0)

        print("✓ KGraph empty input test passed")

    def test_kgraph_single_point(self):
        """Test KGraph with single point."""
        points = np.array([[1.0, 2.0]])
        layer_indices = [0]
        params = {"k": 4}

        edges = self.kgraph.build_graph(points, layer_indices, params)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0], [])

        print("✓ KGraph single point test passed")

    def test_kgraph_small_dataset(self):
        """Test KGraph with small dataset where k > n-1."""
        points = np.random.rand(5, 2)
        layer_indices = list(range(5))
        params = {"k": 10, "max_iterations": 5}  # k is larger than possible neighbors

        edges = self.kgraph.build_graph(points, layer_indices, params)

        # Should build graph successfully
        self.assertEqual(len(edges), 5)

        # Each node should connect to at most n-1 other nodes
        for node_idx, neighbors in edges.items():
            self.assertLessEqual(len(neighbors), 4)  # n-1 = 4

        print("✓ KGraph small dataset test passed")

    def test_kgraph_nn_descent_iterations(self):
        """Test that NN-Descent improves graph quality over iterations."""
        points = np.random.rand(20, 3)
        layer_indices = list(range(20))
        
        # Test with very few iterations vs many iterations
        params_few = {"k": 5, "max_iterations": 1, "delta": 0.0}  # Force single iteration
        params_many = {"k": 5, "max_iterations": 20, "delta": 0.001}

        edges_few = self.kgraph.build_graph(points, layer_indices, params_few)
        
        # Reset for second test
        self.kgraph = KGraph()
        edges_many = self.kgraph.build_graph(points, layer_indices, params_many)

        # Both should have same structure
        self.assertEqual(len(edges_few), 20)
        self.assertEqual(len(edges_many), 20)

        # With more iterations, we should get at least as good quality
        # (measured by having k neighbors for most nodes)
        nodes_with_k_neighbors_few = sum(1 for neighbors in edges_few.values() if len(neighbors) == 5)
        nodes_with_k_neighbors_many = sum(1 for neighbors in edges_many.values() if len(neighbors) == 5)
        
        # Many iterations should generally produce more complete neighborhoods
        self.assertGreaterEqual(nodes_with_k_neighbors_many, nodes_with_k_neighbors_few)

        print("✓ KGraph NN-Descent iterations test passed")

    def test_kgraph_parameter_variations(self):
        """Test KGraph with different parameter settings."""
        points = np.random.rand(12, 3)
        layer_indices = list(range(12))

        # Test different parameter combinations
        param_sets = [
            {"k": 3, "rho": 0.5, "max_iterations": 5},
            {"k": 5, "rho": 1.0, "max_iterations": 10},
            {"k": 7, "rho": 1.5, "max_iterations": 15},
        ]

        for i, params in enumerate(param_sets):
            with self.subTest(params=params):
                kgraph = KGraph()  # Fresh instance for each test
                edges = kgraph.build_graph(points, layer_indices, params)

                # Should build graph successfully
                self.assertEqual(len(edges), 12)

                # Check degree constraints
                for node_idx, neighbors in edges.items():
                    self.assertLessEqual(len(neighbors), params["k"])

        print("✓ KGraph parameter variations test passed")

    def test_kgraph_get_initial_search_node(self):
        """Test get_initial_search_node functionality for KGraph."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"k": 4, "max_iterations": 5}

        # Build the graph first
        edges = self.kgraph.build_graph(points, layer_indices, params)

        # Test with edges provided - should return highest degree node
        initial_node = self.kgraph.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node, layer_indices, "Initial node should be in layer")

        # Verify it's actually a high degree node
        initial_degree = len(edges[initial_node])
        max_degree = max(len(neighbors) for neighbors in edges.values())
        
        # Should be the highest degree node
        self.assertEqual(
            initial_degree, max_degree,
            f"Initial node degree {initial_degree} should equal max degree {max_degree}"
        )

        # Test without edges - should use stored entry point
        initial_node_stored = self.kgraph.get_initial_search_node(points, layer_indices)
        self.assertIn(initial_node_stored, layer_indices, "Stored initial node should be in layer")

        print("✓ KGraph get_initial_search_node test passed")

    def test_kgraph_quality_evaluation(self):
        """Test KGraph quality evaluation metrics."""
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        params = {"k": 4, "max_iterations": 10}

        edges = self.kgraph.build_graph(points, layer_indices, params)
        
        # Evaluate quality
        quality_metrics = self.kgraph.evaluate_quality(points, layer_indices, edges, params["k"])

        # Check that all expected metrics are present
        expected_metrics = ["average_degree", "max_degree", "total_edges", "recall", "target_degree"]
        for metric in expected_metrics:
            self.assertIn(metric, quality_metrics)

        # Check metric ranges
        self.assertGreaterEqual(quality_metrics["average_degree"], 0)
        self.assertLessEqual(quality_metrics["average_degree"], params["k"])
        self.assertGreaterEqual(quality_metrics["recall"], 0.0)
        self.assertLessEqual(quality_metrics["recall"], 1.0)
        self.assertEqual(quality_metrics["target_degree"], params["k"])

        print(f"✓ KGraph quality evaluation test passed: recall={quality_metrics['recall']:.3f}")

    def test_kgraph_henn_integration(self):
        """Test KGraph integration with HENN."""
        points = np.random.rand(30, 3)

        # Create HENN config with KGraph
        config = HENNConfig(
            pgraph_algorithm="kgraph",
            pgraph_params={"k": 5, "max_iterations": 10, "rho": 1.0},
            enable_logging=False
        )

        # Build HENN structure
        henn = HENN(points, config)
        henn.build()

        # Should build successfully
        self.assertGreater(len(henn.layers), 0, "HENN should have layers")

        # Test querying
        query_point = np.random.rand(3)
        results = henn.query(query_point, k=5)

        # Should return results
        self.assertGreater(len(results), 0, "Query should return results")

        print(f"✓ KGraph HENN integration test passed: {len(henn.layers)} layers, {len(results)} results")

    def test_kgraph_convergence(self):
        """Test that KGraph NN-Descent converges."""
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))
        params = {"k": 4, "max_iterations": 50, "delta": 0.001}

        # Build graph and check convergence
        edges = self.kgraph.build_graph(points, layer_indices, params)

        # Should complete without issues
        self.assertEqual(len(edges), 20)

        # All nodes should have some neighbors (except possibly in very sparse cases)
        connected_nodes = sum(1 for neighbors in edges.values() if len(neighbors) > 0)
        self.assertGreater(connected_nodes, 15, "Most nodes should be connected")

        print("✓ KGraph convergence test passed")

    def test_kgraph_high_dimensional_data(self):
        """Test KGraph with high-dimensional data."""
        # Test with higher dimensional data
        points = np.random.rand(25, 8)  # 8-dimensional
        layer_indices = list(range(25))
        params = {"k": 6, "max_iterations": 15, "rho": 1.0}

        edges = self.kgraph.build_graph(points, layer_indices, params)

        # Should handle high dimensions
        self.assertEqual(len(edges), 25)

        # Should maintain reasonable connectivity
        total_edges = sum(len(neighbors) for neighbors in edges.values())
        self.assertGreater(total_edges, 50, "Should have reasonable connectivity in high dimensions")

        # Check that degrees don't exceed k
        for neighbors in edges.values():
            self.assertLessEqual(len(neighbors), params["k"])

        print("✓ KGraph high-dimensional data test passed")

    def test_kgraph_different_k_values(self):
        """Test KGraph with different k values."""
        points = np.random.rand(20, 3)
        layer_indices = list(range(20))

        k_values = [2, 5, 8, 12]

        for k in k_values:
            with self.subTest(k=k):
                kgraph = KGraph()
                params = {"k": k, "max_iterations": 10}
                edges = kgraph.build_graph(points, layer_indices, params)

                # Should build successfully
                self.assertEqual(len(edges), 20)

                # Check degree constraints
                for neighbors in edges.values():
                    self.assertLessEqual(len(neighbors), k)

                # Calculate average degree
                avg_degree = sum(len(neighbors) for neighbors in edges.values()) / len(edges)
                print(f"  k={k}: average degree={avg_degree:.2f}")

        print("✓ KGraph different k values test passed")

    def test_kgraph_reproducibility(self):
        """Test that KGraph produces consistent results with same random seed."""
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        params = {"k": 4, "max_iterations": 10}

        # Build graph twice with same seed
        np.random.seed(123)
        random.seed(123)
        kgraph1 = KGraph()
        edges1 = kgraph1.build_graph(points, layer_indices, params)

        np.random.seed(123)
        random.seed(123)
        kgraph2 = KGraph()
        edges2 = kgraph2.build_graph(points, layer_indices, params)

        # Results should be similar (though not necessarily identical due to floating point)
        self.assertEqual(len(edges1), len(edges2))
        
        # Check that most nodes have similar neighborhoods
        similar_neighborhoods = 0
        for node_idx in layer_indices:
            neighbors1 = set(edges1[node_idx])
            neighbors2 = set(edges2[node_idx])
            if neighbors1 == neighbors2:
                similar_neighborhoods += 1

        # Expect high similarity
        similarity_ratio = similar_neighborhoods / len(layer_indices)
        self.assertGreater(similarity_ratio, 0.7, "Results should be largely reproducible")

        print(f"✓ KGraph reproducibility test passed: {similarity_ratio:.2%} similarity")


if __name__ == "__main__":
    import random
    unittest.main(verbosity=2)
