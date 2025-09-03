"""
Comprehensive tests comparing all graph implementations.
"""

import unittest
import numpy as np
import sys
import os
import time

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgraphs.knn import Knn
from pgraphs.nsw import NSW
from pgraphs.nsg import NSG
from pgraphs.fanng import FANNG
from pgraphs.kgraph import KGraph
from henn import HENN, HENNConfig


class TestGraphComparison(unittest.TestCase):
    """Test cases comparing different graph algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.points = np.random.rand(50, 3)
        self.layer_indices = list(range(50))

    def test_all_graphs_basic_functionality(self):
        """Test that all graph implementations work with same interface."""
        params_knn = {"k": 5}
        params_nsw = {"M": 5, "efConstruction": 50}
        params_nsg = {"R": 5, "L": 10, "C": 15}
        params_fanng = {"K": 5, "L": 15, "R": 10, "alpha": 1.2}
        params_kgraph = {"k": 5, "max_iterations": 10}

        knn = Knn()
        nsw = NSW()
        nsg = NSG()
        fanng = FANNG()
        kgraph = KGraph()

        # All should build graphs without errors
        edges_knn = knn.build_graph(self.points, self.layer_indices, params_knn)
        edges_nsw = nsw.build_graph(self.points, self.layer_indices, params_nsw)
        edges_nsg = nsg.build_graph(self.points, self.layer_indices, params_nsg)
        edges_fanng = fanng.build_graph(self.points, self.layer_indices, params_fanng)
        edges_kgraph = kgraph.build_graph(self.points, self.layer_indices, params_kgraph)

        # All should have same number of nodes
        self.assertEqual(len(edges_knn), 50)
        self.assertEqual(len(edges_nsw), 50)
        self.assertEqual(len(edges_nsg), 50)
        self.assertEqual(len(edges_fanng), 50)
        self.assertEqual(len(edges_kgraph), 50)

        # All should have reasonable connectivity
        knn_edges = sum(len(neighbors) for neighbors in edges_knn.values())
        nsw_edges = sum(len(neighbors) for neighbors in edges_nsw.values())
        nsg_edges = sum(len(neighbors) for neighbors in edges_nsg.values())
        fanng_edges = sum(len(neighbors) for neighbors in edges_fanng.values())
        kgraph_edges = sum(len(neighbors) for neighbors in edges_kgraph.values())

        self.assertGreater(knn_edges, 0)
        self.assertGreater(nsw_edges, 0)
        self.assertGreater(nsg_edges, 0)
        self.assertGreater(fanng_edges, 0)
        self.assertGreater(kgraph_edges, 0)
        nsw_edges = sum(len(neighbors) for neighbors in edges_nsw.values())
        nsg_edges = sum(len(neighbors) for neighbors in edges_nsg.values())

        self.assertGreater(knn_edges, 0)
        self.assertGreater(nsw_edges, 0)
        self.assertGreater(nsg_edges, 0)

    def test_graph_degree_properties(self):
        """Test degree properties of different graphs."""
        k = 4
        params_knn = {"k": k}
        params_nsw = {"M": k, "efConstruction": 50}
        params_nsg = {"R": k, "L": 10, "C": 15}

        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        edges_knn = knn.build_graph(self.points, self.layer_indices, params_knn)
        edges_nsw = nsw.build_graph(self.points, self.layer_indices, params_nsw)
        edges_nsg = nsg.build_graph(self.points, self.layer_indices, params_nsg)

        # KNN: each node should have exactly k neighbors (directed)
        for node_idx, neighbors in edges_knn.items():
            self.assertEqual(len(neighbors), k)

        # NSW: each node should have at most M neighbors due to pruning
        for node_idx, neighbors in edges_nsw.items():
            self.assertLessEqual(len(neighbors), k + 1)  # Allow some flexibility

        # NSG: each node should have at most R neighbors
        for node_idx, neighbors in edges_nsg.items():
            self.assertLessEqual(len(neighbors), k)

    def test_graph_connectivity_properties(self):
        """Test connectivity properties of different graphs."""
        params_knn = {"k": 3}
        params_nsw = {"M": 3, "efConstruction": 30}
        params_nsg = {"R": 3, "L": 8, "C": 12}

        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        edges_knn = knn.build_graph(self.points, self.layer_indices, params_knn)
        edges_nsw = nsw.build_graph(self.points, self.layer_indices, params_nsw)
        edges_nsg = nsg.build_graph(self.points, self.layer_indices, params_nsg)

        def is_connected(edges):
            """Check if graph is weakly connected."""
            if not edges:
                return True
            
            visited = set()
            start_node = next(iter(edges.keys()))
            stack = [start_node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                # Add outgoing neighbors
                for neighbor in edges[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                
                # Add incoming neighbors (treat as undirected for connectivity)
                for node, neighbors in edges.items():
                    if current in neighbors and node not in visited:
                        stack.append(node)

            return len(visited) == len(edges)

        # NSG should always be connected due to its connectivity enforcement
        self.assertTrue(is_connected(edges_nsg), "NSG graph should be connected")
        
        # NSW should typically be connected due to its incremental construction
        # (though this is not guaranteed for all parameter settings)
        
        # KNN might not be connected as it's just nearest neighbors

    def test_henn_integration_with_different_graphs(self):
        """Test HENN integration with different graph algorithms."""
        points = np.random.rand(30, 2)

        # Test with KNN
        config_knn = HENNConfig(
            pgraph_algorithm="knn",
            pgraph_params={"k": 4}
        )
        henn_knn = HENN(points, config_knn)
        henn_knn.build()
        
        # Test with NSW
        config_nsw = HENNConfig(
            pgraph_algorithm="nsw",
            pgraph_params={"M": 4, "efConstruction": 50}
        )
        henn_nsw = HENN(points, config_nsw)
        henn_nsw.build()
        
        # Test with NSG
        config_nsg = HENNConfig(
            pgraph_algorithm="nsg",
            pgraph_params={"R": 4, "L": 10, "C": 15}
        )
        henn_nsg = HENN(points, config_nsg)
        henn_nsg.build()

        # All should build successfully
        self.assertGreater(len(henn_knn.layers), 0)
        self.assertGreater(len(henn_nsw.layers), 0)
        self.assertGreater(len(henn_nsg.layers), 0)

        # Test querying
        query_point = np.random.rand(2)
        
        results_knn = henn_knn.query(query_point, k=3)
        results_nsw = henn_nsw.query(query_point, k=3)
        results_nsg = henn_nsg.query(query_point, k=3)

        # All should return results
        self.assertGreater(len(results_knn), 0)
        self.assertGreater(len(results_nsw), 0)
        self.assertGreater(len(results_nsg), 0)

    def test_performance_comparison(self):
        """Basic performance comparison between graph algorithms."""
        # Use a larger dataset for performance testing
        large_points = np.random.rand(100, 5)
        large_indices = list(range(100))
        
        params_knn = {"k": 8}
        params_nsw = {"M": 8, "efConstruction": 100}
        params_nsg = {"R": 8, "L": 20, "C": 30}

        algorithms = [
            ("KNN", Knn(), params_knn),
            ("NSW", NSW(), params_nsw),
            ("NSG", NSG(), params_nsg)
        ]

        build_times = {}
        
        for name, algorithm, params in algorithms:
            start_time = time.time()
            edges = algorithm.build_graph(large_points, large_indices, params)
            end_time = time.time()
            
            build_times[name] = end_time - start_time
            
            # Verify the graph was built correctly
            self.assertEqual(len(edges), 100)

        # Print timing results (for manual inspection)
        print(f"\nGraph construction times for 100 points:")
        for name, time_taken in build_times.items():
            print(f"{name}: {time_taken:.4f} seconds")

        # All algorithms should complete in reasonable time (< 10 seconds)
        for name, time_taken in build_times.items():
            self.assertLess(time_taken, 10.0, f"{name} took too long: {time_taken:.4f}s")

    def test_graph_quality_metrics(self):
        """Test various quality metrics for different graphs."""
        params_knn = {"k": 5}
        params_nsw = {"M": 5, "efConstruction": 50}
        params_nsg = {"R": 5, "L": 12, "C": 18}

        knn = Knn()
        nsw = NSW()
        nsg = NSG()

        edges_knn = knn.build_graph(self.points, self.layer_indices, params_knn)
        edges_nsw = nsw.build_graph(self.points, self.layer_indices, params_nsw)
        edges_nsg = nsg.build_graph(self.points, self.layer_indices, params_nsg)

        def calculate_avg_path_length(edges, sample_pairs=10):
            """Calculate average shortest path length between random pairs."""
            path_lengths = []
            
            for _ in range(sample_pairs):
                # Pick two random nodes
                nodes = list(edges.keys())
                start = np.random.choice(nodes)
                end = np.random.choice(nodes)
                
                if start == end:
                    continue
                
                # BFS to find shortest path
                visited = set()
                queue = [(start, 0)]
                
                while queue:
                    current, dist = queue.pop(0)
                    if current == end:
                        path_lengths.append(dist)
                        break
                    
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Add neighbors
                    for neighbor in edges.get(current, []):
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))
                    
                    # Also consider reverse edges for connectivity
                    for node, neighbors in edges.items():
                        if current in neighbors and node not in visited:
                            queue.append((node, dist + 1))
            
            return np.mean(path_lengths) if path_lengths else float('inf')

        # Calculate metrics
        graphs = [("KNN", edges_knn), ("NSW", edges_nsw), ("NSG", edges_nsg)]
        
        for name, edges in graphs:
            total_edges = sum(len(neighbors) for neighbors in edges.values())
            avg_degree = total_edges / len(edges) if edges else 0
            avg_path_length = calculate_avg_path_length(edges)
            
            print(f"\n{name} metrics:")
            print(f"  Average degree: {avg_degree:.2f}")
            print(f"  Average path length: {avg_path_length:.2f}")
            
            # Basic sanity checks
            self.assertGreater(avg_degree, 0)
            self.assertLess(avg_path_length, float('inf'))

    def test_edge_case_handling(self):
        """Test how different algorithms handle edge cases."""
        algorithms = [
            ("KNN", Knn(), {"k": 3}),
            ("NSW", NSW(), {"M": 3, "efConstruction": 30}),
            ("NSG", NSG(), {"R": 3, "L": 8, "C": 12})
        ]

        # Test with very small datasets
        small_points = np.random.rand(3, 2)
        small_indices = [0, 1, 2]

        for name, algorithm, params in algorithms:
            edges = algorithm.build_graph(small_points, small_indices, params)
            self.assertEqual(len(edges), 3, f"{name} failed with 3 points")

        # Test with identical points
        identical_points = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        identical_indices = [0, 1, 2, 3]

        for name, algorithm, params in algorithms:
            # Should handle identical points without crashing
            edges = algorithm.build_graph(identical_points, identical_indices, params)
            self.assertEqual(len(edges), 4, f"{name} failed with identical points")


if __name__ == "__main__":
    unittest.main(verbosity=2)
