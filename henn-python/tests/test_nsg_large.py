#!/usr/bin/env python3
"""
Test script to verify NSG connectivity optimization with larger datasets.
"""

import numpy as np
import time
from pgraphs.nsg import NSG


def test_large_dataset():
    """Test with a larger dataset to showcase optimization benefits."""
    # Create larger synthetic dataset
    np.random.seed(42)
    n_points = 5000
    dimensions = 64
    
    print(f"Testing NSG connectivity optimization with LARGE dataset: {n_points} points, {dimensions} dimensions")
    
    # Generate random points
    points = np.random.randn(n_points, dimensions).astype(np.float32)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)  # Normalize for cosine distance
    
    layer_indices = list(range(n_points))
    
    # Test only the fast methods for large dataset
    optimization_methods = ["vectorized", "approximate"]
    
    results = {}
    
    for method in optimization_methods:
        print(f"\n--- Testing {method} optimization on large dataset ---")
        
        # Create NSG instance
        nsg = NSG(
            distance="cosine",
            medoid_method="vectorized",  # Use fast medoid for all tests
            connectivity_optimization=method
        )
        
        # Build graph and measure time
        start_time = time.time()
        
        try:
            graph = nsg.build_graph(points, layer_indices, {"R": 16, "L": 40, "C": 80})
            build_time = time.time() - start_time
            
            # Verify graph properties
            total_edges = sum(len(neighbors) for neighbors in graph.values())
            avg_degree = total_edges / len(graph) if graph else 0
            
            results[method] = {
                "time": build_time,
                "edges": total_edges,
                "avg_degree": avg_degree,
                "success": True
            }
            
            print(f"  Build time: {build_time:.2f}s")
            print(f"  Total edges: {total_edges}")
            print(f"  Average degree: {avg_degree:.2f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method] = {
                "time": float('inf'),
                "edges": 0,
                "avg_degree": 0,
                "success": False,
                "error": str(e)
            }
    
    # Print comparison
    print(f"\n--- Large Dataset Performance Comparison ---")
    for method, result in results.items():
        if result["success"]:
            print(f"{method:12}: {result['time']:6.2f}s, {result['edges']:5d} edges, {result['avg_degree']:5.2f} avg degree")
        else:
            print(f"{method:12}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    test_large_dataset()
