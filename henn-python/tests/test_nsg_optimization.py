#!/usr/bin/env python3
"""
Test script to verify NSG connectivity optimization improvements.
"""

import numpy as np
import time
from pgraphs.nsg import NSG


def test_connectivity_optimization():
    """Test the different connectivity optimization methods."""
    # Create synthetic dataset
    np.random.seed(42)
    n_points = 1000
    dimensions = 128
    
    # Generate random points
    points = np.random.randn(n_points, dimensions).astype(np.float32)
    if n_points <= 1000:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)  # Normalize for cosine distance
    
    layer_indices = list(range(n_points))
    
    print(f"Testing NSG connectivity optimization with {n_points} points, {dimensions} dimensions")
    
    # Test different optimization levels
    optimization_methods = ["original", "vectorized", "approximate"]
    
    results = {}
    
    for method in optimization_methods:
        print(f"\n--- Testing {method} optimization ---")
        
        # Create NSG instance
        nsg = NSG(
            distance="cosine",
            medoid_method="vectorized",  # Use fast medoid for all tests
            connectivity_optimization=method
        )
        
        # Build graph and measure time
        start_time = time.time()
        
        try:
            graph = nsg.build_graph(points, layer_indices, {"R": 16, "L": 50, "C": 100})
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
    print(f"\n--- Performance Comparison ---")
    for method, result in results.items():
        if result["success"]:
            print(f"{method:12}: {result['time']:6.2f}s, {result['edges']:5d} edges, {result['avg_degree']:5.2f} avg degree")
        else:
            print(f"{method:12}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Check if optimization helped
    if results["vectorized"]["success"] and results["original"]["success"]:
        speedup = results["original"]["time"] / results["vectorized"]["time"]
        print(f"\nVectorized speedup: {speedup:.2f}x")
    
    if results["approximate"]["success"] and results["vectorized"]["success"]:
        speedup = results["vectorized"]["time"] / results["approximate"]["time"]
        print(f"Approximate speedup: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    test_connectivity_optimization()
