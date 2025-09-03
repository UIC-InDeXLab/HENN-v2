"""
Integration tests for get_initial_search_node across all graph implementations.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.knn import Knn
from pgraphs.nsw import NSW
from pgraphs.nsg import NSG


class TestInitialSearchNodeIntegration(unittest.TestCase):
    """Integration tests for get_initial_search_node across all graph types."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.points = np.random.rand(20, 3)
        self.layer_indices = list(range(20))

    def test_all_graphs_initial_node_validity(self):
        """Test that all graph types return valid initial nodes."""
        # Initialize all graph types
        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        # Build graphs
        knn_edges = knn.build_graph(self.points, self.layer_indices, {"k": 5})
        nsw_edges = nsw.build_graph(self.points, self.layer_indices, {"M": 5, "efConstruction": 50})
        nsg_edges = nsg.build_graph(self.points, self.layer_indices, {"R": 5, "L": 10, "C": 15})

        # Test initial node selection
        knn_initial = knn.get_initial_search_node(self.points, self.layer_indices, knn_edges)
        nsw_initial = nsw.get_initial_search_node(self.points, self.layer_indices, nsw_edges)
        nsg_initial = nsg.get_initial_search_node(self.points, self.layer_indices, nsg_edges)

        # All should return valid nodes
        self.assertIn(knn_initial, self.layer_indices, "k-NN initial node should be valid")
        self.assertIn(nsw_initial, self.layer_indices, "NSW initial node should be valid")
        self.assertIn(nsg_initial, self.layer_indices, "NSG initial node should be valid")

        print(f"✓ All graphs return valid initial nodes: k-NN={knn_initial}, NSW={nsw_initial}, NSG={nsg_initial}")

    def test_empty_layer_handling(self):
        """Test that all graph types handle empty layers correctly."""
        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        # All should return None for empty layer
        self.assertIsNone(knn.get_initial_search_node(self.points, [], {}))
        self.assertIsNone(nsw.get_initial_search_node(self.points, [], {}))
        self.assertIsNone(nsg.get_initial_search_node(self.points, [], {}))

        print("✓ All graphs handle empty layers correctly")

    def test_single_node_layer(self):
        """Test that all graph types handle single-node layers correctly."""
        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        single_layer = [10]

        # All should return the single node
        self.assertEqual(knn.get_initial_search_node(self.points, single_layer, {}), 10)
        self.assertEqual(nsw.get_initial_search_node(self.points, single_layer, {}), 10)
        self.assertEqual(nsg.get_initial_search_node(self.points, single_layer, {}), 10)

        print("✓ All graphs handle single-node layers correctly")

    def test_initial_node_strategies(self):
        """Test the different strategies used by each graph type."""
        # Create a scenario where different strategies should yield different results
        # Points arranged in a line with one central point
        points = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],  # Line
            [2.0, 0.1],  # Near center - should be medoid for NSG
            [0.5, 1.0], [1.5, 1.0], [2.5, 1.0], [3.5, 1.0]  # Parallel line
        ])
        layer_indices = list(range(10))

        # Build graphs
        knn = Knn()
        nsw = NSW() 
        nsg = NSG()

        knn_edges = knn.build_graph(points, layer_indices, {"k": 3})
        nsw_edges = nsw.build_graph(points, layer_indices, {"M": 3, "efConstruction": 20})
        nsg_edges = nsg.build_graph(points, layer_indices, {"R": 3, "L": 8, "C": 10})

        # Get initial nodes
        np.random.seed(123)  # For reproducible k-NN selection
        knn_initial = knn.get_initial_search_node(points, layer_indices, knn_edges)
        nsw_initial = nsw.get_initial_search_node(points, layer_indices, nsw_edges)
        nsg_initial = nsg.get_initial_search_node(points, layer_indices, nsg_edges)

        # Analyze strategies
        print(f"Graph strategies test:")
        print(f"  k-NN (random): {knn_initial}")
        print(f"  NSW (high degree): {nsw_initial} (degree: {len(nsw_edges.get(nsw_initial, []))})")
        print(f"  NSG (medoid): {nsg_initial}")

        # NSG should consistently pick the medoid
        expected_medoid = nsg._find_medoid(points, layer_indices)
        self.assertEqual(nsg_initial, expected_medoid, "NSG should return medoid")

        # NSW should pick a well-connected node
        nsw_degree = len(nsw_edges.get(nsw_initial, []))
        max_degree = max(len(neighbors) for neighbors in nsw_edges.values())
        self.assertEqual(nsw_degree, max_degree, "NSW should pick highest degree node")

        print("✓ Graph strategies work as expected")

    def test_consistency_across_runs(self):
        """Test consistency of initial node selection across multiple runs."""
        # Build graphs once
        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        knn_edges = knn.build_graph(self.points, self.layer_indices, {"k": 4})
        nsw_edges = nsw.build_graph(self.points, self.layer_indices, {"M": 4, "efConstruction": 40})
        nsg_edges = nsg.build_graph(self.points, self.layer_indices, {"R": 4, "L": 10, "C": 15})

        # Test NSG consistency (should always return same medoid)
        nsg_initials = []
        for _ in range(5):
            initial = nsg.get_initial_search_node(self.points, self.layer_indices, nsg_edges)
            nsg_initials.append(initial)
        
        self.assertEqual(len(set(nsg_initials)), 1, "NSG should always return same medoid")

        # Test NSW consistency (should always return same highest-degree node)
        nsw_initials = []
        for _ in range(5):
            initial = nsw.get_initial_search_node(self.points, self.layer_indices, nsw_edges)
            nsw_initials.append(initial)
        
        self.assertEqual(len(set(nsw_initials)), 1, "NSW should always return same highest-degree node")

        # Test k-NN with fixed seed (should be consistent)
        knn_initials = []
        for i in range(5):
            np.random.seed(i + 100)  # Different seeds
            initial = knn.get_initial_search_node(self.points, self.layer_indices, knn_edges)
            knn_initials.append(initial)

        # k-NN with different seeds might give different results, but with same seed should be same
        np.random.seed(999)
        knn_initial1 = knn.get_initial_search_node(self.points, self.layer_indices, knn_edges)
        np.random.seed(999)
        knn_initial2 = knn.get_initial_search_node(self.points, self.layer_indices, knn_edges)
        self.assertEqual(knn_initial1, knn_initial2, "k-NN should be consistent with same seed")

        print("✓ Consistency tests passed")

    def test_performance_characteristics(self):
        """Test performance characteristics of initial node selection."""
        import time

        # Test with larger dataset
        large_points = np.random.rand(100, 5)
        large_indices = list(range(100))

        # Build graphs and measure initial node selection time
        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        # Build graphs (this is the expensive part)
        print("Building graphs for performance test...")
        knn_edges = knn.build_graph(large_points, large_indices, {"k": 10})
        nsw_edges = nsw.build_graph(large_points, large_indices, {"M": 10, "efConstruction": 100})
        nsg_edges = nsg.build_graph(large_points, large_indices, {"R": 10, "L": 20, "C": 30})

        # Measure initial node selection time
        times = {}

        # k-NN timing
        start = time.time()
        for _ in range(100):
            knn.get_initial_search_node(large_points, large_indices, knn_edges)
        times['knn'] = (time.time() - start) / 100

        # NSW timing  
        start = time.time()
        for _ in range(100):
            nsw.get_initial_search_node(large_points, large_indices, nsw_edges)
        times['nsw'] = (time.time() - start) / 100

        # NSG timing
        start = time.time()
        for _ in range(100):
            nsg.get_initial_search_node(large_points, large_indices, nsg_edges)
        times['nsg'] = (time.time() - start) / 100

        print(f"Initial node selection times (avg over 100 calls):")
        print(f"  k-NN: {times['knn']*1000:.3f}ms")
        print(f"  NSW:  {times['nsw']*1000:.3f}ms") 
        print(f"  NSG:  {times['nsg']*1000:.3f}ms")

        # All should be reasonably fast
        # k-NN should be very fast (just random selection)
        # NSW should be fast (just looking up stored highest degree node)
        # NSG will be slower as it needs to compute medoid
        self.assertLess(times['knn'], 0.001, f"k-NN too slow: {times['knn']:.6f}s")
        self.assertLess(times['nsw'], 0.001, f"NSW too slow: {times['nsw']:.6f}s")
        self.assertLess(times['nsg'], 0.1, f"NSG too slow: {times['nsg']:.6f}s")  # More lenient for NSG

        print("✓ Performance test passed")

    def test_edge_cases_all_graphs(self):
        """Test edge cases for all graph implementations."""
        graphs = [
            ("k-NN", Knn(), {"k": 3}),
            ("NSW", NSW(), {"M": 3, "efConstruction": 20}),
            ("NSG", NSG(), {"R": 3, "L": 8, "C": 10})
        ]

        # Test with very small datasets
        small_points = np.random.rand(3, 2)
        small_indices = [0, 1, 2]

        for name, graph, params in graphs:
            with self.subTest(graph=name):
                try:
                    edges = graph.build_graph(small_points, small_indices, params)
                    initial = graph.get_initial_search_node(small_points, small_indices, edges)
                    self.assertIn(initial, small_indices, f"{name} should handle small datasets")
                except Exception as e:
                    self.fail(f"{name} failed on small dataset: {e}")

        # Test with identical points
        identical_points = np.array([[1.0, 1.0]] * 5)
        identical_indices = list(range(5))

        for name, graph, params in graphs:
            with self.subTest(graph=name, case="identical"):
                try:
                    edges = graph.build_graph(identical_points, identical_indices, params)
                    initial = graph.get_initial_search_node(identical_points, identical_indices, edges)
                    self.assertIn(initial, identical_indices, f"{name} should handle identical points")
                except Exception as e:
                    self.fail(f"{name} failed on identical points: {e}")

        print("✓ Edge cases test passed for all graphs")


if __name__ == "__main__":
    unittest.main()
