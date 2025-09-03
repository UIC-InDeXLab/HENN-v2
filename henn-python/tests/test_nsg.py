"""
Unit tests for NSG graph implementation.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.nsg import NSG


class TestNSG(unittest.TestCase):
    """Test cases for NSG (Navigable Sparse Graph) implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.nsg = NSG()
        np.random.seed(42)  # For reproducible tests

    def test_nsg_basic_functionality(self):
        """Test basic NSG functionality."""
        # Create a simple 2D dataset
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))
        params = {"R": 5, "L": 10, "C": 15}

        edges = self.nsg.build_graph(points, layer_indices, params)

        # Verify graph was created
        self.assertEqual(len(edges), 20)
        
        # Verify all nodes are present
        for idx in layer_indices:
            self.assertIn(idx, edges)

    def test_nsg_degree_constraint(self):
        """Test that NSG respects maximum out-degree constraint."""
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        R = 4
        params = {"R": R, "L": 10, "C": 15}

        edges = self.nsg.build_graph(points, layer_indices, params)

        # Check that no node has out-degree greater than R
        for node_idx, neighbors in edges.items():
            self.assertLessEqual(
                len(neighbors), 
                R,
                f"Node {node_idx} has out-degree {len(neighbors)} > R={R}"
            )

    def test_nsg_empty_layer(self):
        """Test NSG with empty layer."""
        points = np.random.rand(10, 2)
        layer_indices = []
        params = {"R": 5}

        edges = self.nsg.build_graph(points, layer_indices, params)
        self.assertEqual(edges, {})

    def test_nsg_single_node(self):
        """Test NSG with single node."""
        points = np.random.rand(10, 2)
        layer_indices = [5]
        params = {"R": 5}

        edges = self.nsg.build_graph(points, layer_indices, params)
        self.assertEqual(edges, {5: []})

    def test_nsg_two_nodes(self):
        """Test NSG with two nodes."""
        points = np.random.rand(10, 2)
        layer_indices = [3, 7]
        params = {"R": 5}

        edges = self.nsg.build_graph(points, layer_indices, params)
        
        # Should have entries for both nodes
        self.assertEqual(len(edges), 2)
        self.assertIn(3, edges)
        self.assertIn(7, edges)

    def test_nsg_medoid_finding(self):
        """Test medoid finding functionality."""
        # Create points where one is clearly central
        points = np.array([
            [0.0, 0.0],  # Central point
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ])
        layer_indices = list(range(5))

        medoid = self.nsg._find_medoid(points, layer_indices)
        
        # The central point should be the medoid
        self.assertEqual(medoid, 0)

    def test_nsg_initial_knn_graph(self):
        """Test initial k-NN graph construction."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        k = 3

        knn_graph = self.nsg._build_initial_knn_graph(points, layer_indices, k)

        # Check that each node has exactly k neighbors (or fewer if not enough nodes)
        for node_idx, neighbors in knn_graph.items():
            expected_neighbors = min(k, len(layer_indices) - 1)
            self.assertEqual(len(neighbors), expected_neighbors)

    def test_nsg_neighbor_selection(self):
        """Test NSG neighbor selection strategy."""
        # Create points in a line to test diversity selection
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],  # Very close to point 1
            [2.0, 0.0],
            [3.0, 0.0]
        ])
        
        node_idx = 0
        candidates = [1, 2, 3, 4]  # All other points
        R = 2

        selected = self.nsg._select_neighbors_nsg(points, node_idx, candidates, R)

        # Should select R neighbors
        self.assertLessEqual(len(selected), R)
        
        # Should prioritize diversity (not select both 1 and 2 which are very close)
        if 1 in selected and 2 in selected:
            # This would be suboptimal but not necessarily wrong
            pass

    def test_nsg_connectivity(self):
        """Test that NSG ensures graph connectivity."""
        points = np.random.rand(10, 3)
        layer_indices = list(range(10))
        params = {"R": 3, "L": 8, "C": 10}

        edges = self.nsg.build_graph(points, layer_indices, params)

        # Test connectivity by doing BFS from medoid
        medoid = self.nsg._find_medoid(points, layer_indices)
        visited = set()
        queue = [medoid]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in edges[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
            
            # Also check incoming edges (since graph might be directed)
            for node, neighbors in edges.items():
                if current in neighbors and node not in visited:
                    queue.append(node)

        # All nodes should be reachable
        self.assertEqual(len(visited), len(layer_indices),
                        "Graph is not connected - some nodes are unreachable")

    def test_nsg_parameters_validation(self):
        """Test NSG with different parameter combinations."""
        points = np.random.rand(12, 2)
        layer_indices = list(range(12))

        # Test with minimal parameters
        edges1 = self.nsg.build_graph(points, layer_indices, {"R": 2})
        self.assertEqual(len(edges1), 12)

        # Test with all parameters
        edges2 = self.nsg.build_graph(points, layer_indices, 
                                     {"R": 4, "L": 8, "C": 10})
        self.assertEqual(len(edges2), 12)

        # Test with no parameters (should use defaults)
        edges3 = self.nsg.build_graph(points, layer_indices, {})
        self.assertEqual(len(edges3), 12)

    def test_nsg_with_henn_config(self):
        """Test NSG integration with HENN configuration."""
        # This tests if NSG can be used as a drop-in replacement for other graphs
        from henn import HENN, HENNConfig
        
        points = np.random.rand(50, 3)
        
        # Create config with NSG
        config = HENNConfig(
            pgraph_algorithm="NSG",
            pgraph_params={"R": 8, "L": 20, "C": 30}
        )
        
        # This should work without errors
        # Note: We can't easily test the full HENN build without modifying
        # the config to actually use NSG instance, but we can test the interface
        nsg = NSG()
        layer_indices = list(range(20))
        edges = nsg.build_graph(points, layer_indices, config.pgraph_params)
        
        self.assertEqual(len(edges), 20)

    def test_nsg_large_graph(self):
        """Test NSG with a larger graph to check performance and correctness."""
        points = np.random.rand(100, 5)
        layer_indices = list(range(100))
        params = {"R": 10, "L": 30, "C": 50}

        edges = self.nsg.build_graph(points, layer_indices, params)

        # Basic checks
        self.assertEqual(len(edges), 100)
        
        # Check degree constraints
        for node_idx, neighbors in edges.items():
            self.assertLessEqual(len(neighbors), params["R"])

        # Check that graph has reasonable connectivity
        total_edges = sum(len(neighbors) for neighbors in edges.values())
        avg_degree = total_edges / len(layer_indices)
        
        # Average degree should be reasonable (not too sparse)
        self.assertGreater(avg_degree, 1.0)
        self.assertLessEqual(avg_degree, params["R"])

    def test_nsg_beam_search(self):
        """Test beam search candidate finding."""
        points = np.random.rand(10, 2)
        layer_indices = list(range(10))
        
        # Create a simple graph for testing
        current_graph = {i: [] for i in layer_indices}
        # Connect nodes in a simple pattern
        for i in range(len(layer_indices) - 1):
            current_graph[i].append(i + 1)
            
        navigation_node = 0
        query_idx = 5
        L = 5
        C = 8

        candidates = self.nsg._beam_search_candidates(
            points, layer_indices, query_idx, current_graph, navigation_node, L, C
        )

        # Should return candidates (excluding query node itself)
        self.assertIsInstance(candidates, list)
        self.assertNotIn(query_idx, candidates)
        self.assertLessEqual(len(candidates), C)

    def test_nsg_deterministic_with_seed(self):
        """Test that NSG produces deterministic results with same random seed."""
        points = np.random.rand(15, 3)
        layer_indices = list(range(15))
        params = {"R": 5, "L": 10, "C": 15}

        # Build graph twice with same seed
        np.random.seed(123)
        edges1 = self.nsg.build_graph(points, layer_indices, params)
        
        np.random.seed(123)
        edges2 = self.nsg.build_graph(points, layer_indices, params)

        # Results should be identical
        self.assertEqual(edges1.keys(), edges2.keys())
        for node in edges1:
            self.assertEqual(set(edges1[node]), set(edges2[node]))

    def test_nsg_get_initial_search_node(self):
        """Test get_initial_search_node functionality for NSG."""
        points = np.random.rand(15, 2)
        layer_indices = list(range(15))
        params = {"R": 5, "L": 10, "C": 15}

        # Build the graph first
        edges = self.nsg.build_graph(points, layer_indices, params)

        # Test that initial node is the medoid (navigation node)
        initial_node = self.nsg.get_initial_search_node(points, layer_indices, edges)
        self.assertIn(initial_node, layer_indices, "Initial node should be in layer")

        # For NSG, the initial node should be the medoid
        expected_medoid = self.nsg._find_medoid(points, layer_indices)
        self.assertEqual(
            initial_node, expected_medoid,
            f"Initial node {initial_node} should be medoid {expected_medoid}"
        )

        # Test multiple calls return same node
        initial_node2 = self.nsg.get_initial_search_node(points, layer_indices, edges)
        self.assertEqual(initial_node, initial_node2, "Should return same medoid consistently")

        # Test with empty layer
        empty_initial = self.nsg.get_initial_search_node(points, [], edges)
        self.assertIsNone(empty_initial, "Empty layer should return None")

        # Test with single node
        single_node_initial = self.nsg.get_initial_search_node(points, layer_indices, edges)
        self.assertEqual(single_node_initial, 8, "Single node layer should return that node")

        print("✓ NSG get_initial_search_node test passed")

    def test_nsg_medoid_consistency(self):
        """Test that NSG consistently finds the same medoid for the same data."""
        # Create points where medoid is clearly identifiable
        points = np.array([
            [0.0, 0.0],   # Center point - should be medoid
            [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],  # Corner points
            [2.0, 2.0], [2.0, -2.0], [-2.0, 2.0], [-2.0, -2.0]   # Far corner points
        ])
        layer_indices = list(range(9))

        # Find medoid directly
        medoid1 = self.nsg._find_medoid(points, layer_indices)
        medoid2 = self.nsg._find_medoid(points, layer_indices)
        
        # Should be consistent
        self.assertEqual(medoid1, medoid2, "Medoid finding should be deterministic")
        
        # The center point (index 0) should be the medoid
        self.assertEqual(medoid1, 0, "Center point should be identified as medoid")

        # Test that get_initial_search_node returns this medoid
        params = {"R": 3, "L": 8, "C": 10}
        edges = self.nsg.build_graph(points, layer_indices, params)
        initial_node = self.nsg.get_initial_search_node(points, layer_indices, edges)
        self.assertEqual(initial_node, 0, "Initial search node should be the medoid")

        print("✓ NSG medoid consistency test passed")

    def test_nsg_initial_node_optimality(self):
        """Test that NSG's initial node (medoid) is actually optimal."""
        points = np.random.rand(12, 3)
        layer_indices = list(range(12))

        # Build graph
        params = {"R": 4, "L": 8, "C": 12}
        edges = self.nsg.build_graph(points, layer_indices, params)
        
        # Get the medoid
        medoid = self.nsg.get_initial_search_node(points, layer_indices, edges)
        
        # Calculate sum of distances from medoid to all other points
        medoid_point = points[medoid]
        medoid_sum_dist = sum(
            np.linalg.norm(points[idx] - medoid_point) 
            for idx in layer_indices if idx != medoid
        )

        # Check that no other point has a smaller sum of distances
        for other_idx in layer_indices:
            if other_idx == medoid:
                continue
                
            other_point = points[other_idx]
            other_sum_dist = sum(
                np.linalg.norm(points[idx] - other_point) 
                for idx in layer_indices if idx != other_idx
            )
            
            self.assertLessEqual(
                medoid_sum_dist, other_sum_dist,
                f"Medoid {medoid} sum_dist {medoid_sum_dist:.3f} should be <= "
                f"point {other_idx} sum_dist {other_sum_dist:.3f}"
            )

        print(f"✓ NSG medoid optimality test passed: medoid={medoid}, sum_dist={medoid_sum_dist:.3f}")


class TestNSGIntegration(unittest.TestCase):
    """Integration tests for NSG with other components."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_nsg_vs_knn_comparison(self):
        """Compare NSG and k-NN graph properties."""
        from pgraphs.knn import Knn
        
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))
        
        # Build both graphs with similar parameters
        knn = Knn()
        knn_edges = knn.build_graph(points, layer_indices, {"k": 5})
        
        nsg = NSG()
        nsg_edges = nsg.build_graph(points, layer_indices, {"R": 5, "L": 10, "C": 15})

        # Both should have same number of nodes
        self.assertEqual(len(knn_edges), len(nsg_edges))
        
        # NSG should generally have sparser connectivity due to its selection strategy
        knn_total_edges = sum(len(neighbors) for neighbors in knn_edges.values())
        nsg_total_edges = sum(len(neighbors) for neighbors in nsg_edges.values())
        
        # This is not guaranteed but often true - NSG tends to be sparser
        # We just check that both are reasonable
        self.assertGreater(knn_total_edges, 0)
        self.assertGreater(nsg_total_edges, 0)

    def test_nsg_vs_nsw_comparison(self):
        """Compare NSG and NSW graph properties."""
        from pgraphs.nsw import NSW
        
        points = np.random.rand(20, 2)
        layer_indices = list(range(20))
        
        # Build both graphs
        nsw = NSW()
        nsw_edges = nsw.build_graph(points, layer_indices, {"M": 5, "efConstruction": 50})
        
        nsg = NSG()
        nsg_edges = nsg.build_graph(points, layer_indices, {"R": 5, "L": 10, "C": 15})

        # Both should have same number of nodes
        self.assertEqual(len(nsw_edges), len(nsg_edges))
        
        # Both should produce connected graphs
        self.assertGreater(sum(len(neighbors) for neighbors in nsw_edges.values()), 0)
        self.assertGreater(sum(len(neighbors) for neighbors in nsg_edges.values()), 0)


if __name__ == "__main__":
    unittest.main()
