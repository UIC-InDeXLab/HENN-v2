"""
Unit tests for k-NN graph implementation.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.knn import Knn


class TestKnn(unittest.TestCase):
    """Test cases for k-NN (k-Nearest Neighbors) graph implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.knn = Knn()
        np.random.seed(42)  # For reproducible tests

    def test_knn_basic_functionality(self):
        """Test basic k-NN functionality."""
        # Create a simple 2D dataset
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"k": 3}

        edges = self.knn.build_graph(points, layer_indices, params)

        # Verify graph was created
        self.assertEqual(len(edges), 10)

        # Verify graph properties
        for node_idx, neighbors in edges.items():
            # Each node should have exactly k neighbors (or fewer if not enough nodes)
            expected_k = min(params["k"], len(layer_indices) - 1)
            self.assertEqual(
                len(neighbors), expected_k,
                f"Node {node_idx} has {len(neighbors)} neighbors, expected {expected_k}"
            )

            # All neighbors should be valid indices
            for neighbor_idx in neighbors:
                self.assertIn(neighbor_idx, layer_indices, f"Invalid neighbor {neighbor_idx}")
                self.assertNotEqual(neighbor_idx, node_idx, "Node should not be neighbor of itself")

        print(f"✓ k-NN basic test passed: Built graph with {len(edges)} nodes")

    def test_knn_edge_cases(self):
        """Test k-NN edge cases."""
        points = np.random.rand(5, 2)

        # Test empty layer
        edges = self.knn.build_graph(points, [], {"k": 3})
        self.assertEqual(edges, {}, "Empty layer should return empty graph")

        # Test single point
        edges = self.knn.build_graph(points, [0], {"k": 3})
        self.assertEqual(edges, {0: []}, "Single point should have no neighbors")

        # Test two points
        edges = self.knn.build_graph(points, [0, 1], {"k": 3})
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0], [1], "Point 0 should have point 1 as neighbor")
        self.assertEqual(edges[1], [0], "Point 1 should have point 0 as neighbor")

        print("✓ k-NN edge cases test passed")

    def test_knn_missing_parameters(self):
        """Test k-NN with missing parameters."""
        points = np.random.rand(5, 2)

        # Should raise ValueError if k not provided
        with self.assertRaises(ValueError):
            self.knn.build_graph(points, [0, 1, 2], {})

        with self.assertRaises(ValueError):
            self.knn.build_graph(points, [0, 1, 2], None)

    def test_knn_different_k_values(self):
        """Test k-NN with different k values."""
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))

        for k in [1, 3, 5, 10, 19]:
            with self.subTest(k=k):
                params = {"k": k}
                edges = self.knn.build_graph(points, layer_indices, params)

                # Verify all nodes are present
                self.assertEqual(len(edges), 20)

                # Check that each node has exactly k neighbors
                for node_idx, neighbors in edges.items():
                    self.assertEqual(len(neighbors), k)

        print("✓ k-NN different k values test passed")

    def test_knn_distance_correctness(self):
        """Test that k-NN correctly finds nearest neighbors."""
        # Create points where nearest neighbors are obvious
        points = np.array([
            [0.0, 0.0],  # 0
            [0.1, 0.0],  # 1 - closest to 0
            [0.0, 0.1],  # 2 - second closest to 0
            [1.0, 1.0],  # 3 - far from 0
            [1.1, 1.0],  # 4 - closest to 3
        ])
        layer_indices = list(range(5))
        params = {"k": 2}

        edges = self.knn.build_graph(points, layer_indices, params)

        # Check that point 0's nearest neighbors are points 1 and 2
        neighbors_0 = set(edges[0])
        self.assertEqual(neighbors_0, {1, 2}, f"Point 0 neighbors should be {{1, 2}}, got {neighbors_0}")

        # Check that point 3's nearest neighbors include point 4
        neighbors_3 = set(edges[3])
        self.assertIn(4, neighbors_3, "Point 4 should be among point 3's nearest neighbors")

        print("✓ k-NN distance correctness test passed")

    def test_knn_get_initial_search_node(self):
        """Test get_initial_search_node functionality for k-NN."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"k": 3}

        # Build the graph first
        edges = self.knn.build_graph(points, layer_indices, params)

        # Test that initial node is randomly selected from layer
        np.random.seed(123)
        initial_node1 = self.knn.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node1, layer_indices, "Initial node should be in layer")

        # Test with different seed gives different result (high probability)
        np.random.seed(456)
        initial_node2 = self.knn.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node2, layer_indices, "Initial node should be in layer")

        # Test with same seed gives same result
        np.random.seed(123)
        initial_node3 = self.knn.get_initial_search_node(points, layer_indices, edges)
        self.assertEqual(initial_node1, initial_node3, "Same seed should give same result")

        # Test with empty layer
        empty_initial = self.knn.get_initial_search_node(points, [], edges)
        self.assertIsNone(empty_initial, "Empty layer should return None")

        # Test with single node
        single_node_initial = self.knn.get_initial_search_node(points, [7], edges)
        self.assertEqual(single_node_initial, 7, "Single node layer should return that node")

        print("✓ k-NN get_initial_search_node test passed")

    def test_knn_initial_node_distribution(self):
        """Test that k-NN initial node selection is reasonably distributed."""
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))
        params = {"k": 4}

        # Build the graph
        edges = self.knn.build_graph(points, layer_indices, params)

        # Sample initial nodes many times
        selected_nodes = []
        for i in range(100):
            np.random.seed(i)
            initial_node = self.knn.get_initial_search_node(points, layer_indices, edges)
            selected_nodes.append(initial_node)

        # Check that multiple different nodes were selected
        unique_nodes = set(selected_nodes)
        self.assertGreater(
            len(unique_nodes), 5, 
            f"Should select various nodes, got {len(unique_nodes)} unique from {unique_nodes}"
        )

        # All selected nodes should be valid
        for node in unique_nodes:
            self.assertIn(node, layer_indices, f"Selected node {node} should be in layer")

        print(f"✓ k-NN distribution test passed: {len(unique_nodes)} unique nodes selected")

    def test_knn_higher_dimensional_data(self):
        """Test k-NN with higher dimensional data."""
        # Test with 5D data
        points = np.random.rand(15, 5)
        layer_indices = list(range(15))
        params = {"k": 4}

        edges = self.knn.build_graph(points, layer_indices, params)

        # Basic validation
        self.assertEqual(len(edges), 15)
        for node_idx, neighbors in edges.items():
            self.assertEqual(len(neighbors), 4)

        # Test initial node selection
        initial_node = self.knn.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node, layer_indices)

        print("✓ k-NN higher dimensional test passed")

    def test_knn_large_k_value(self):
        """Test k-NN with k close to total number of points."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"k": 9}  # All possible neighbors

        edges = self.knn.build_graph(points, layer_indices, params)

        # Each node should have 9 neighbors (all other nodes)
        for node_idx, neighbors in edges.items():
            self.assertEqual(len(neighbors), 9)
            # Should include all other nodes
            expected_neighbors = set(layer_indices) - {node_idx}
            self.assertEqual(set(neighbors), expected_neighbors)

        # Test initial node selection
        initial_node = self.knn.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node, layer_indices)

        print("✓ k-NN large k value test passed")


class TestKnnIntegration(unittest.TestCase):
    """Integration tests for k-NN with other components."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_knn_graph_properties(self):
        """Test mathematical properties of k-NN graphs."""
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        params = {"k": 5}

        knn = Knn()
        edges = knn.build_graph(points, layer_indices, params)

        # Calculate average degree
        total_edges = sum(len(neighbors) for neighbors in edges.values())
        avg_degree = total_edges / len(layer_indices)
        self.assertEqual(avg_degree, 5.0, "Average degree should equal k")

        # Calculate total unique edges (each edge counted once)
        unique_edges = set()
        for node, neighbors in edges.items():
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))
                unique_edges.add(edge)

        # For directed k-NN, total edges = n*k
        self.assertEqual(total_edges, 15 * 5, "Total directed edges should be n*k")

        print(f"✓ k-NN graph properties test passed: {len(unique_edges)} unique edges")

    def test_knn_vs_other_graphs_initial_node(self):
        """Compare initial node selection across different graph types."""
        from pgraphs.nsw import NSW
        from pgraphs.nsg import NSG

        points = np.random.rand(20, 2)
        layer_indices = list(range(20))

        # Build all three graph types
        knn = Knn()
        knn_edges = knn.build_graph(points, layer_indices, {"k": 5})

        nsw = NSW()
        nsw_edges = nsw.build_graph(points, layer_indices, {"M": 5})

        nsg = NSG()
        nsg_edges = nsg.build_graph(points, layer_indices, {"R": 5, "L": 10, "C": 15})

        # Get initial nodes
        np.random.seed(999)  # Fix seed for k-NN random selection
        knn_initial = knn.get_initial_search_node(points, layer_indices, knn_edges)
        nsw_initial = nsw.get_initial_search_node(points, layer_indices, nsw_edges)
        nsg_initial = nsg.get_initial_search_node(points, layer_indices, nsg_edges)

        # All should be valid
        self.assertIn(knn_initial, layer_indices)
        self.assertIn(nsw_initial, layer_indices)
        self.assertIn(nsg_initial, layer_indices)

        # k-NN is random, so can be any node
        # NSW should prefer high-degree nodes
        # NSG should use medoid

        print(f"✓ Initial node comparison: k-NN={knn_initial}, NSW={nsw_initial}, NSG={nsg_initial}")


if __name__ == "__main__":
    unittest.main()
