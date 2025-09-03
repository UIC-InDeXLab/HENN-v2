"""
Unit tests for NSW graph implementation.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.nsw import NSW


class TestNSW(unittest.TestCase):
    """Test cases for NSW (Navigable Small World) graph implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.nsw = NSW()
        np.random.seed(42)  # For reproducible tests

    def test_nsw_basic_functionality(self):
        """Test basic NSW functionality."""
        # Create a simple 2D dataset
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"M": 3}

        edges = self.nsw.build_graph(points, layer_indices, params)

        # Verify graph was created
        self.assertEqual(len(edges), 10)

        # Verify graph properties
        for node_idx, neighbors in edges.items():
            # Check that degree doesn't exceed M (except possibly during construction)
            self.assertLessEqual(
                len(neighbors),
                params["M"] + 1,
                f"Node {node_idx} has degree {len(neighbors)} > M+1",
            )

            # Check bidirectional connections
            for neighbor_idx in neighbors:
                self.assertIn(
                    node_idx,
                    edges[neighbor_idx],
                    f"Connection {node_idx}->{neighbor_idx} is not bidirectional",
                )

        print(f"✓ NSW basic test passed: Built graph with {len(edges)} nodes")

    def test_nsw_edge_cases(self):
        """Test NSW edge cases."""
        points = np.random.rand(5, 2)

        # Test empty layer
        edges = self.nsw.build_graph(points, [], {"M": 3})
        self.assertEqual(edges, {}, "Empty layer should return empty graph")

        # Test single point
        edges = self.nsw.build_graph(points, [0], {"M": 3})
        self.assertEqual(edges, {0: []}, "Single point should have no connections")

        # Test two points
        edges = self.nsw.build_graph(points, [0, 1], {"M": 3})
        self.assertIn(0, edges, "Point 0 should be in graph")
        self.assertIn(1, edges, "Point 1 should be in graph")
        self.assertIn(1, edges[0], "Points should be connected")
        self.assertIn(0, edges[1], "Points should be connected bidirectionally")

        print("✓ NSW edge cases test passed")

    def test_nsw_missing_parameters(self):
        """Test NSW with missing parameters (should use defaults)."""
        points = np.random.rand(5, 2)

        # Should work with no parameters (use defaults)
        edges = self.nsw.build_graph(points, [0, 1, 2], {})
        self.assertEqual(len(edges), 3)

    def test_nsw_backward_compatibility(self):
        """Test NSW backward compatibility with 'k' parameter."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))

        # Should work with old 'k' parameter
        edges = self.nsw.build_graph(points, layer_indices, {"k": 4})
        self.assertEqual(len(edges), 10)

        # Verify degree constraint
        for node_idx, neighbors in edges.items():
            self.assertLessEqual(len(neighbors), 4 + 1)  # Allow some flexibility

    def test_nsw_different_parameters(self):
        """Test NSW with different parameter values."""
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))

        for M in [2, 4, 6, 8]:
            with self.subTest(M=M):
                params = {"M": M, "efConstruction": max(M * 2, 50)}
                edges = self.nsw.build_graph(points, layer_indices, params)

                # Verify all nodes are present
                self.assertEqual(len(edges), 20)

                # Check degree constraints
                for node_idx, neighbors in edges.items():
                    self.assertLessEqual(len(neighbors), M + 1)

                    # Check bidirectional connections
                    for neighbor_idx in neighbors:
                        self.assertIn(node_idx, edges[neighbor_idx])

        print("✓ NSW different parameter values test passed")

    def test_nsw_graph_connectivity(self):
        """Test that NSW graph maintains reasonable connectivity."""
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        params = {"M": 4}

        edges = self.nsw.build_graph(points, layer_indices, params)

        # Calculate total edges
        total_edges = sum(len(neighbors) for neighbors in edges.values()) // 2

        # Should have some reasonable number of edges
        self.assertGreater(total_edges, 0, "Graph should have some edges")
        self.assertLessEqual(total_edges, 15 * 4, "Graph shouldn't have too many edges")

        # Check that most nodes have some connections (except possibly isolated ones)
        connected_nodes = sum(1 for neighbors in edges.values() if len(neighbors) > 0)
        self.assertGreater(connected_nodes, 10, "Most nodes should be connected")

        print(
            f"✓ NSW connectivity test passed: {total_edges} edges, {connected_nodes} connected nodes"
        )

    def test_nsw_incremental_construction(self):
        """Test that NSW builds graph incrementally."""
        points = np.array(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [0.0, 1.0],  # 2
                [1.0, 1.0],  # 3
                [0.5, 0.5],  # 4 - center point
            ]
        )
        layer_indices = list(range(5))
        params = {"M": 2}

        edges = self.nsw.build_graph(points, layer_indices, params)

        # Verify basic properties
        self.assertEqual(len(edges), 5)

        # Center point (4) should be well connected due to its central position
        center_degree = len(edges[4])
        self.assertGreater(center_degree, 0, "Center point should have connections")

        # All connections should be bidirectional
        for node_idx, neighbors in edges.items():
            for neighbor_idx in neighbors:
                self.assertIn(node_idx, edges[neighbor_idx])

        print("✓ NSW incremental construction test passed")

    def test_nsw_efconstruction_parameter(self):
        """Test NSW with different efConstruction values."""
        points = np.random.rand(15, 3)
        layer_indices = list(range(15))

        # Test with different efConstruction values
        for ef in [10, 50, 100]:
            with self.subTest(efConstruction=ef):
                params = {"M": 4, "efConstruction": ef}
                edges = self.nsw.build_graph(points, layer_indices, params)

                # Should build graph successfully
                self.assertEqual(len(edges), 15)

                # Check degree constraints
                for node_idx, neighbors in edges.items():
                    self.assertLessEqual(len(neighbors), 4 + 1)

        print("✓ NSW efConstruction parameter test passed")

    def test_nsw_get_initial_search_node(self):
        """Test get_initial_search_node functionality for NSW."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        params = {"M": 3}

        # Build the graph first
        edges = self.nsw.build_graph(points, layer_indices, params)

        # Test with edges provided - should return highest degree node
        initial_node = self.nsw.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node, layer_indices, "Initial node should be in layer")

        # Verify it's actually the highest degree node (or one of them if tie)
        max_degree = max(len(neighbors) for neighbors in edges.values())
        initial_degree = len(edges[initial_node])
        self.assertEqual(
            initial_degree, max_degree, 
            f"Initial node degree {initial_degree} should equal max degree {max_degree}"
        )

        # Test with no edges provided - should return random node
        np.random.seed(123)
        initial_node_no_edges = self.nsw.get_initial_search_node(points, layer_indices, None)
        self.assertIn(initial_node_no_edges, layer_indices, "Random initial node should be in layer")

        # Test with empty layer
        empty_initial = self.nsw.get_initial_search_node(points, [], edges)
        self.assertIsNone(empty_initial, "Empty layer should return None")

        # Test reproducibility with same edges
        initial_node2 = self.nsw.get_initial_search_node(points, layer_indices, edges)
        self.assertEqual(initial_node, initial_node2, "Should return same node with same graph")

        print("✓ NSW get_initial_search_node test passed")

    def test_nsw_initial_node_degree_analysis(self):
        """Test that NSW initial node selection prefers high-degree nodes."""
        # Create a star topology where one node is connected to many others
        points = np.array([
            [0.5, 0.5],  # Center node (should have highest degree)
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],  # Corner nodes
            [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3]   # Other nodes
        ])
        layer_indices = list(range(9))
        params = {"M": 6}  # High connectivity to create clear degree differences

        edges = self.nsw.build_graph(points, layer_indices, params)

        # Get initial node multiple times to see if it consistently picks high-degree nodes
        initial_nodes = []
        for _ in range(10):
            initial_node = self.nsw.get_initial_search_node(points, layer_indices, edges)
            initial_nodes.append(initial_node)

        # All selected nodes should have high degrees
        degrees = {node: len(edges[node]) for node in layer_indices}
        max_degree = max(degrees.values())
        
        for node in set(initial_nodes):
            node_degree = degrees[node]
            # Should be at least 70% of max degree
            self.assertGreaterEqual(
                node_degree, max_degree * 0.7,
                f"Selected node {node} has degree {node_degree}, max is {max_degree}"
            )

        print(f"✓ NSW degree analysis test passed: degrees={degrees}, selected nodes={set(initial_nodes)}")


if __name__ == "__main__":
    unittest.main()
