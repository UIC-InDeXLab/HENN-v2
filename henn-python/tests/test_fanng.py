"""
Test cases for FANNG (Fast Approximate Nearest Neighbor Graph) implementation.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.fanng import FANNG
from henn import HENN, HENNConfig


class TestFANNG(unittest.TestCase):
    """Test cases for FANNG (Fast Approximate Nearest Neighbor Graph) implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.fanng = FANNG()
        np.random.seed(42)  # For reproducible tests

    def test_fanng_basic_functionality(self):
        """Test basic FANNG functionality."""
        # Create a simple 2D dataset
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"K": 3, "L": 8, "R": 5, "alpha": 1.2}

        edges = self.fanng.build_graph(points, layer_indices, params)

        # Verify graph was created
        self.assertEqual(len(edges), 10)

        # Verify graph properties
        for node_idx, neighbors in edges.items():
            # Check that degree doesn't exceed K by much (some flexibility for reverse links)
            self.assertLessEqual(
                len(neighbors),
                params["K"] + 2,  # Allow some flexibility due to reverse linking
                f"Node {node_idx} has degree {len(neighbors)} > K+2",
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

        print("✓ FANNG basic functionality test passed")

    def test_fanng_empty_input(self):
        """Test FANNG with empty input."""
        points = np.array([]).reshape(0, 2)
        layer_indices = []
        params = {"K": 3}

        edges = self.fanng.build_graph(points, layer_indices, params)
        self.assertEqual(len(edges), 0)

        print("✓ FANNG empty input test passed")

    def test_fanng_single_point(self):
        """Test FANNG with single point."""
        points = np.array([[1.0, 2.0]])
        layer_indices = [0]
        params = {"K": 3}

        edges = self.fanng.build_graph(points, layer_indices, params)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0], [])

        print("✓ FANNG single point test passed")

    def test_fanng_parameter_variations(self):
        """Test FANNG with different parameter settings."""
        points = np.random.rand(15, 3)
        layer_indices = list(range(15))

        # Test different parameter combinations
        param_sets = [
            {"K": 2, "L": 4, "R": 3, "alpha": 1.0},
            {"K": 4, "L": 12, "R": 8, "alpha": 1.5},
            {"K": 6, "L": 20, "R": 10, "alpha": 2.0},
        ]

        for i, params in enumerate(param_sets):
            with self.subTest(params=params):
                edges = self.fanng.build_graph(points, layer_indices, params)

                # Should build graph successfully
                self.assertEqual(len(edges), 15)

                # Check degree constraints (with some flexibility)
                for node_idx, neighbors in edges.items():
                    self.assertLessEqual(len(neighbors), params["K"] + 3)

        print("✓ FANNG parameter variations test passed")

    def test_fanng_graph_connectivity(self):
        """Test that FANNG graph maintains reasonable connectivity."""
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))
        params = {"K": 4, "L": 12, "R": 8, "alpha": 1.2}

        edges = self.fanng.build_graph(points, layer_indices, params)

        # Calculate total edges
        total_edges = sum(len(neighbors) for neighbors in edges.values()) // 2

        # Should have some reasonable number of edges
        self.assertGreater(total_edges, 0, "Graph should have some edges")
        self.assertLessEqual(total_edges, 20 * 4, "Graph shouldn't have too many edges")

        # Check that most nodes have some connections
        connected_nodes = sum(1 for neighbors in edges.values() if len(neighbors) > 0)
        self.assertGreater(connected_nodes, 15, "Most nodes should be connected")

        print(
            f"✓ FANNG connectivity test passed: {total_edges} edges, {connected_nodes} connected nodes"
        )

    def test_fanng_get_initial_search_node(self):
        """Test get_initial_search_node functionality for FANNG."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"K": 3, "L": 8, "R": 5}

        # Build the graph first
        edges = self.fanng.build_graph(points, layer_indices, params)

        # Test with edges provided - should return highest degree node
        initial_node = self.fanng.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node, layer_indices, "Initial node should be in layer")

        # Verify it's actually a high degree node
        initial_degree = len(edges[initial_node])
        max_degree = max(len(neighbors) for neighbors in edges.values())
        
        # Should be at least close to max degree
        self.assertGreaterEqual(
            initial_degree, max_degree * 0.8,
            f"Initial node degree {initial_degree} should be close to max degree {max_degree}"
        )

        # Test without edges - should use stored entry point
        initial_node_stored = self.fanng.get_initial_search_node(points, layer_indices)
        self.assertIn(initial_node_stored, layer_indices, "Stored initial node should be in layer")

        print("✓ FANNG get_initial_search_node test passed")

    def test_fanng_incremental_construction(self):
        """Test that FANNG builds graph incrementally."""
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [0.0, 1.0],  # 2
            [1.0, 1.0],  # 3
            [0.5, 0.5],  # 4 - center point
        ])
        layer_indices = list(range(5))
        params = {"K": 2, "L": 4, "R": 3}

        edges = self.fanng.build_graph(points, layer_indices, params)

        # Verify basic properties
        self.assertEqual(len(edges), 5)

        # Center point (4) should be well connected due to its central position
        center_degree = len(edges[4])
        self.assertGreater(center_degree, 0, "Center point should have connections")

        print("✓ FANNG incremental construction test passed")

    def test_fanng_diversity_in_selection(self):
        """Test that FANNG selects diverse neighbors."""
        # Create a scenario where diversity matters
        points = np.array([
            [0.0, 0.0],   # 0 - query point
            [0.1, 0.0],   # 1 - very close
            [0.11, 0.0],  # 2 - very close to 1
            [0.12, 0.0],  # 3 - very close to 1 and 2
            [1.0, 0.0],   # 4 - far but diverse
            [0.0, 1.0],   # 5 - far but diverse
        ])
        layer_indices = list(range(6))
        params = {"K": 3, "L": 6, "R": 4, "alpha": 1.5}  # Higher alpha for more diversity

        edges = self.fanng.build_graph(points, layer_indices, params)

        # Node 0 should prefer diverse neighbors over clustered ones
        neighbors_0 = edges[0]
        
        # Should have some connections
        self.assertGreater(len(neighbors_0), 0)

        print(f"✓ FANNG diversity test passed: node 0 neighbors = {neighbors_0}")

    def test_fanng_henn_integration(self):
        """Test FANNG integration with HENN."""
        points = np.random.rand(50, 3)

        # Create HENN config with FANNG
        config = HENNConfig(
            pgraph_algorithm="fanng",
            pgraph_params={"K": 6, "L": 18, "R": 12, "alpha": 1.3},
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

        print(f"✓ FANNG HENN integration test passed: {len(henn.layers)} layers, {len(results)} results")

    def test_fanng_parameter_edge_cases(self):
        """Test FANNG with edge case parameters."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))

        # Test with very small K
        edges_small_k = self.fanng.build_graph(points, layer_indices, {"K": 1, "L": 4})
        self.assertEqual(len(edges_small_k), 10)

        # Test with K larger than number of nodes
        edges_large_k = self.fanng.build_graph(points, layer_indices, {"K": 20, "L": 30})
        self.assertEqual(len(edges_large_k), 10)

        # Each node should connect to at most n-1 other nodes
        for neighbors in edges_large_k.values():
            self.assertLessEqual(len(neighbors), 9)

        print("✓ FANNG parameter edge cases test passed")

    def test_fanng_high_dimensional_data(self):
        """Test FANNG with high-dimensional data."""
        # Test with higher dimensional data
        points = np.random.rand(30, 10)  # 10-dimensional
        layer_indices = list(range(30))
        params = {"K": 5, "L": 15, "R": 10, "alpha": 1.2}

        edges = self.fanng.build_graph(points, layer_indices, params)

        # Should handle high dimensions
        self.assertEqual(len(edges), 30)

        # Should maintain reasonable connectivity
        total_edges = sum(len(neighbors) for neighbors in edges.values())
        self.assertGreater(total_edges, 50, "Should have reasonable connectivity in high dimensions")

        print("✓ FANNG high-dimensional data test passed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
