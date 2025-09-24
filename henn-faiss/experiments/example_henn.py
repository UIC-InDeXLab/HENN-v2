#!/usr/bin/env python3
"""
Example usage of the HENN (Hierarchical EpsNet Navigation Graph) index in Python.

This demonstrates how to use the new HENN index through the Faiss Python interface.
HENN is similar to HNSW but uses EPS-Net sampling for layer construction.
"""

import numpy as np
import faiss
import time

def generate_random_data(n, d):
    """Generate random normalized vectors."""
    np.random.seed(1234)
    data = np.random.random((n, d)).astype('float32')
    # Normalize for better performance
    faiss.normalize_L2(data)
    return data

def test_henn_flat():
    """Test basic HENN flat index functionality."""
    print("=== Testing IndexHENNFlat ===")
    
    d = 64          # dimension
    n = 10000       # database size  
    nq = 100        # number of queries
    k = 10          # number of nearest neighbors
    M = 16          # max connections per node
    
    print(f"Dataset: {n} vectors of dimension {d}")
    
    # Generate data
    xb = generate_random_data(n, d)
    xq = generate_random_data(nq, d)
    
    # Create HENN index
    index = faiss.IndexHENNFlat(d, M)
    
    # Configure HENN parameters
    index.set_epsnet_strategy("random")      # or "budget_aware"
    index.set_pgraph_algorithm("nsw")        # or "knn" 
    index.set_exp_decay(2)                   # layer size reduction factor
    index.henn.ef_construction = 200         # construction parameter
    index.henn.efSearch = 32                 # search parameter
    
    print(f"Index configuration:")
    print(f"  EPS-Net strategy: {index.epsnet_strategy}")  
    print(f"  Graph algorithm: {index.pgraph_algorithm}")
    print(f"  Exponential decay: {index.exp_decay}")
    print(f"  ef_construction: {index.henn.ef_construction}")
    print(f"  efSearch: {index.henn.efSearch}")
    
    # Add vectors to index
    print(f"Adding {n} vectors...")
    start_time = time.time()
    index.add(xb)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.3f} seconds")
    
    # Print index statistics
    print(f"Index built with {len(index.henn.layers)} layers")
    print(f"Entry point: {index.henn.entry_point}")
    print(f"Max level: {index.henn.max_level}")
    
    # Search
    print(f"\nSearching for {k} nearest neighbors...")
    start_time = time.time()
    D, I = index.search(xq, k)
    search_time = time.time() - start_time
    
    print(f"Search time: {search_time:.3f} seconds")
    print(f"Average time per query: {search_time/nq*1000:.2f} ms")
    
    # Print some results
    print(f"\nFirst query results:")
    for i in range(min(5, k)):
        print(f"  {i+1}: ID={I[0,i]}, distance={D[0,i]:.6f}")
    
    return index, xb, xq

def compare_with_hnsw():
    """Compare HENN with HNSW performance."""
    print("\n=== Comparing HENN vs HNSW ===")
    
    d = 128
    n = 20000
    nq = 100 
    k = 10
    M = 16
    
    # Generate data
    xb = generate_random_data(n, d)
    xq = generate_random_data(nq, d)
    
    # Test HNSW
    print("Testing HNSW...")
    index_hnsw = faiss.IndexHNSWFlat(d, M)
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.hnsw.efSearch = 32
    
    start_time = time.time()
    index_hnsw.add(xb)
    hnsw_build_time = time.time() - start_time
    
    start_time = time.time() 
    D_hnsw, I_hnsw = index_hnsw.search(xq, k)
    hnsw_search_time = time.time() - start_time
    
    # Test HENN
    print("Testing HENN...")
    index_henn = faiss.IndexHENNFlat(d, M)
    index_henn.set_epsnet_strategy("budget_aware")
    index_henn.set_pgraph_algorithm("nsw") 
    index_henn.henn.ef_construction = 200
    index_henn.henn.efSearch = 32
    
    start_time = time.time()
    index_henn.add(xb)
    henn_build_time = time.time() - start_time
    
    start_time = time.time()
    D_henn, I_henn = index_henn.search(xq, k)
    henn_search_time = time.time() - start_time
    
    # Compare results
    print(f"\nPerformance comparison:")
    print(f"HNSW - Build: {hnsw_build_time:.3f}s, Search: {hnsw_search_time:.3f}s")
    print(f"HENN - Build: {henn_build_time:.3f}s, Search: {henn_search_time:.3f}s")
    
    # Calculate recall (approximate)
    recall = 0
    for i in range(nq):
        intersect = len(set(I_henn[i]) & set(I_hnsw[i]))
        recall += intersect / k
    recall /= nq
    
    print(f"Approximate recall between HENN and HNSW: {recall:.3f}")

def test_different_configurations():
    """Test different HENN configurations."""
    print("\n=== Testing Different HENN Configurations ===")
    
    d = 64
    n = 5000
    nq = 50
    k = 10 
    M = 16
    
    xb = generate_random_data(n, d)
    xq = generate_random_data(nq, d)
    
    configs = [
        ("random", "nsw"),
        ("random", "knn"), 
        ("budget_aware", "nsw"),
        ("budget_aware", "knn")
    ]
    
    results = []
    
    for epsnet_strategy, pgraph_algorithm in configs:
        print(f"\nTesting: EPS-Net={epsnet_strategy}, Graph={pgraph_algorithm}")
        
        index = faiss.IndexHENNFlat(d, M)
        index.set_epsnet_strategy(epsnet_strategy)
        index.set_pgraph_algorithm(pgraph_algorithm)
        index.henn.ef_construction = 100
        index.henn.efSearch = 20
        
        # Build
        start_time = time.time()
        index.add(xb)
        build_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        D, I = index.search(xq, k)
        search_time = time.time() - start_time
        
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Layers: {len(index.henn.layers)}")
        
        results.append({
            'config': (epsnet_strategy, pgraph_algorithm),
            'build_time': build_time,
            'search_time': search_time, 
            'layers': len(index.henn.layers)
        })
    
    # Summary
    print(f"\nConfiguration Summary:")
    for result in results:
        config_str = f"{result['config'][0]}-{result['config'][1]}"
        print(f"  {config_str:20} | Build: {result['build_time']:.3f}s | "
              f"Search: {result['search_time']:.3f}s | Layers: {result['layers']}")

def test_henn_pq():
    """Test HENN with Product Quantization."""
    print("\n=== Testing IndexHENNPQ ===")
    
    d = 128
    n = 10000
    nq = 100
    k = 10
    M = 16
    pq_m = 8       # number of PQ segments  
    pq_nbits = 8   # bits per PQ code
    
    xb = generate_random_data(n, d)
    xq = generate_random_data(nq, d)
    
    # Create HENN-PQ index
    index = faiss.IndexHENNPQ(d, pq_m, M, pq_nbits)
    index.set_epsnet_strategy("random")
    index.set_pgraph_algorithm("nsw")
    index.henn.ef_construction = 200
    
    # Train the PQ quantizer
    print("Training PQ quantizer...")
    xt = generate_random_data(10000, d)  # training data
    start_time = time.time()
    index.train(xt)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.3f}s")
    
    # Add data
    print("Adding data...")
    start_time = time.time()
    index.add(xb) 
    build_time = time.time() - start_time
    
    # Search
    start_time = time.time()
    D, I = index.search(xq, k)
    search_time = time.time() - start_time
    
    print(f"HENN-PQ Results:")
    print(f"  Build time: {build_time:.3f}s")
    print(f"  Search time: {search_time:.3f}s")
    print(f"  Index size: ~{(n * pq_m * pq_nbits) / 8 / 1024:.1f} KB (compressed)")

def main():
    """Main function to run all tests."""
    print("HENN Index Python Examples")
    print("=" * 50)
    print("HENN: Hierarchical EpsNet Navigation Graph")
    print("Similar to HNSW but with EPS-Net sampling for layer construction")
    print()
    
    try:
        # Basic functionality test
        test_henn_flat()
        
        # Compare with HNSW
        compare_with_hnsw()
        
        # Different configurations
        test_different_configurations()
        
        # PQ variant
        test_henn_pq()
        
        print("\n" + "=" * 50)
        print("All HENN tests completed successfully!")
        print("\nHENN Index Features:")
        print("- Hierarchical structure with EPS-Net sampling")
        print("- Choice of 'random' or 'budget_aware' sampling")
        print("- Choice of 'nsw' or 'knn' proximity graphs")  
        print("- Configurable layer size reduction (exp_decay)")
        print("- Compatible with PQ and SQ compression")
        print("- Similar search performance to HNSW")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
