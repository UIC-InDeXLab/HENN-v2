#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/mohsen/henn/henn-python')

import numpy as np
from pgraphs.nsg import NSG
import time

def test_medoid_methods():
    print("Starting medoid optimization test...")
    
    # Create test data
    np.random.seed(42)
    n_points = 200
    n_dims = 10
    points = np.random.randn(n_points, n_dims)
    indices = list(range(n_points))
    
    print(f"Test data: {n_points} points, {n_dims} dimensions")
    
    methods = [
        ("original", {}),
        ("vectorized", {}),
        ("sampling", {"medoid_sample_size": 100})
    ]
    
    results = {}
    
    for method_name, extra_params in methods:
        print(f"\nTesting {method_name} method...")
        
        # Create NSG instance
        params = {"medoid_method": method_name}
        params.update(extra_params)
        nsg = NSG(**params)
        
        # Time the medoid finding
        start_time = time.time()
        medoid_idx = nsg._find_medoid(points, indices)
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[method_name] = {"time": elapsed, "medoid": medoid_idx}
        
        print(f"  Medoid index: {medoid_idx}")
        print(f"  Time: {elapsed:.4f} seconds")
    
    # Calculate speedups
    print("\nPerformance comparison:")
    original_time = results["original"]["time"]
    
    for method_name in ["vectorized", "sampling"]:
        method_time = results[method_name]["time"]
        speedup = original_time / method_time if method_time > 0 else float('inf')
        print(f"  {method_name}: {speedup:.2f}x speedup")
    
    print("\nTest completed successfully!")
    return results

if __name__ == "__main__":
    test_medoid_methods()
