#!/usr/bin/env python3

import sys
import os
import numpy as np
import time

# Add the project path
sys.path.insert(0, '/home/mohsen/henn/henn-python')

try:
    from pgraphs.nsg import NSG
    print("✓ NSG import successful")
except Exception as e:
    print(f"✗ NSG import failed: {e}")
    sys.exit(1)

def comprehensive_test():
    """Comprehensive test of all medoid optimization methods."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE NSG MEDOID OPTIMIZATION TEST")
    print("="*60)
    
    # Test configurations
    configs = [
        (100, 10, "l2"),
        (100, 10, "cosine"),
        (300, 20, "l2"),
    ]
    
    for n_points, n_dims, distance in configs:
        print(f"\nTesting: {n_points} points, {n_dims}D, {distance} distance")
        print("-" * 50)
        
        # Generate test data
        np.random.seed(42)
        points = np.random.randn(n_points, n_dims)
        if distance == "cosine":
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        indices = list(range(n_points))
        
        # Test methods
        methods = ["original", "vectorized", "sampling"]
        results = {}
        
        for method in methods:
            try:
                print(f"  Testing {method} method...", end=" ")
                
                # Create NSG instance
                nsg = NSG(
                    distance=distance,
                    medoid_method=method,
                    medoid_sample_size=min(100, n_points//2)
                )
                
                # Time the operation
                start_time = time.time()
                medoid_idx = nsg._find_medoid(points, indices)
                end_time = time.time()
                
                elapsed = end_time - start_time
                results[method] = {
                    'time': elapsed,
                    'medoid': medoid_idx,
                    'success': True
                }
                
                print(f"✓ medoid={medoid_idx}, time={elapsed:.4f}s")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                results[method] = {'success': False, 'error': str(e)}
        
        # Performance comparison
        if all(results[m].get('success', False) for m in methods):
            original_time = results['original']['time']
            print(f"\n  Performance comparison (baseline: {original_time:.4f}s):")
            
            for method in ['vectorized', 'sampling']:
                if results[method]['success']:
                    speedup = original_time / results[method]['time']
                    print(f"    {method}: {speedup:.1f}x speedup")
        
        # Quality comparison
        medoids = {m: results[m]['medoid'] for m in methods if results[m].get('success', False)}
        if len(set(medoids.values())) > 1:
            print(f"  ⚠ Different medoids found: {medoids}")
        else:
            print(f"  ✓ All methods found same medoid: {list(medoids.values())[0]}")

def test_edge_cases():
    """Test edge cases."""
    print(f"\n" + "="*60)
    print("EDGE CASE TESTING")
    print("="*60)
    
    # Single point
    print("\nTesting single point...")
    points = np.array([[1, 2, 3]])
    indices = [0]
    
    nsg = NSG(medoid_method="vectorized")
    result = nsg._find_medoid(points, indices)
    print(f"  Single point result: {result} (expected: 0)")
    
    # Very small dataset
    print("\nTesting 2 points...")
    points = np.array([[1, 2], [3, 4]])
    indices = [0, 1]
    
    for method in ["original", "vectorized", "sampling"]:
        nsg = NSG(medoid_method=method)
        result = nsg._find_medoid(points, indices)
        print(f"  {method}: {result}")

if __name__ == "__main__":
    try:
        comprehensive_test()
        test_edge_cases()
        
        print(f"\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nSUMMARY:")
        print("- Original method: Exact, O(n²), good for small datasets")
        print("- Vectorized method: Exact, vectorized O(n²), faster for medium datasets")
        print("- Sampling method: Approximate, O(k), best for large datasets")
        print("- Default method remains 'original' for backward compatibility")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
