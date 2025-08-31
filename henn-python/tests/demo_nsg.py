"""
Demonstration of NSG graph usage with HENN.

This script shows how to use the NSG (Navigable Sparse Graph) algorithm
as a proximity graph in the HENN structure.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from henn import HENN, HENNConfig
from pgraphs.nsg import NSG
import time


def demonstrate_nsg_usage():
    """Demonstrate NSG usage with HENN."""
    
    print("NSG (Navigable Sparse Graph) Demonstration")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    dimensions = 5
    
    print(f"Generating {n_points} random points in {dimensions}D space...")
    points = np.random.rand(n_points, dimensions)
    
    # Create HENN configurations for different graph algorithms
    configs = {
        "KNN": HENNConfig(
            pgraph_algorithm="knn",
            pgraph_params={"k": 8},
            enable_logging=False
        ),
        "NSW": HENNConfig(
            pgraph_algorithm="nsw", 
            pgraph_params={"M": 8, "efConstruction": 100},
            enable_logging=False
        ),
        "NSG": HENNConfig(
            pgraph_algorithm="nsg",
            pgraph_params={"R": 8, "L": 20, "C": 30},
            enable_logging=False
        )
    }
    
    # Build HENN structures with different graph algorithms
    henn_structures = {}
    build_times = {}
    
    for name, config in configs.items():
        print(f"\nBuilding HENN with {name} graph...")
        start_time = time.time()
        
        henn = HENN(points, config)
        henn.build()
        
        end_time = time.time()
        
        henn_structures[name] = henn
        build_times[name] = end_time - start_time
        
        print(f"  Built {len(henn.layers)} layers in {build_times[name]:.4f} seconds")
        for i, layer in enumerate(henn.layers):
            total_edges = sum(len(neighbors) for neighbors in layer.edges.values())
            avg_degree = total_edges / layer.n if layer.n > 0 else 0
            print(f"    Layer {i+1}: {layer.n} points, avg degree: {avg_degree:.2f}")
    
    # Test query performance
    print(f"\nQuery Performance Comparison")
    print("-" * 30)
    
    # Generate query points
    n_queries = 20
    k = 5
    query_points = np.random.rand(n_queries, dimensions)
    
    query_times = {}
    
    for name, henn in henn_structures.items():
        print(f"\nTesting {name} queries...")
        start_time = time.time()
        
        all_results = []
        for query_point in query_points:
            results = henn.query(query_point, k=k)
            all_results.append(results)
        
        end_time = time.time()
        query_times[name] = end_time - start_time
        avg_query_time = query_times[name] / n_queries
        
        print(f"  {n_queries} queries completed in {query_times[name]:.4f} seconds")
        print(f"  Average query time: {avg_query_time:.6f} seconds")
        
        # Check result quality (all should return k results for this dataset size)
        successful_queries = sum(1 for results in all_results if len(results) == k)
        print(f"  Successful queries: {successful_queries}/{n_queries}")
    
    # Summary
    print(f"\nSummary")
    print("=" * 30)
    print("Build Times:")
    for name, time_taken in build_times.items():
        print(f"  {name}: {time_taken:.4f}s")
    
    print("\nQuery Times (total for 20 queries):")
    for name, time_taken in query_times.items():
        print(f"  {name}: {time_taken:.4f}s")
    
    print(f"\nNSG Features:")
    print("- Controlled out-degree (R parameter)")
    print("- Navigable structure with medoid entry point")
    print("- Diversity-aware neighbor selection")
    print("- Guaranteed connectivity")
    print("- Generally provides good balance between accuracy and efficiency")


def demonstrate_nsg_parameters():
    """Demonstrate the effect of different NSG parameters."""
    
    print(f"\nNSG Parameter Effects")
    print("=" * 30)
    
    # Generate sample data
    np.random.seed(123)
    points = np.random.rand(100, 3)
    
    # Test different parameter settings
    parameter_sets = [
        {"R": 4, "L": 10, "C": 15, "name": "Small (R=4, L=10, C=15)"},
        {"R": 8, "L": 20, "C": 30, "name": "Medium (R=8, L=20, C=30)"},
        {"R": 16, "L": 40, "C": 60, "name": "Large (R=16, L=40, C=60)"},
    ]
    
    nsg = NSG()
    layer_indices = list(range(100))
    
    for params in parameter_sets:
        print(f"\nTesting {params['name']}:")
        
        start_time = time.time()
        edges = nsg.build_graph(points, layer_indices, params)
        end_time = time.time()
        
        # Calculate statistics
        total_edges = sum(len(neighbors) for neighbors in edges.values())
        avg_degree = total_edges / len(edges)
        max_degree = max(len(neighbors) for neighbors in edges.values())
        
        print(f"  Build time: {end_time - start_time:.4f}s")
        print(f"  Total edges: {total_edges}")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Maximum degree: {max_degree}")
        print(f"  Degree constraint (R): {params['R']}")


if __name__ == "__main__":
    demonstrate_nsg_usage()
    demonstrate_nsg_parameters()
