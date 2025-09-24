#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for ANN Indices
Benchmarks: HENN+NSW/KNN (random & budget-aware), IndexHNSW, IndexLSH, IndexIVF+PQ, IndexNSG, Flat

Requirements:
- Uses FAISS HENN implementation (IndexHENNFlat) instead of Python HENN
- Uses ann_datasets.py to load euclidean datasets with specific sample sizes:
  - SIFT: 20k samples
  - GIST: 20k samples
  - Fashion-MNIST: 10k samples
- Uses synthetic dataset (mixture gaussians, 10k samples)
- Saves results to CSV for plotting
- Progress bars throughout
"""

import sys
import os
import time
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Add paths for both FAISS and HENN Python modules
sys.path.insert(0, "../../henn-cpp/build/faiss/python")
sys.path.insert(0, "../../henn-python")

# Import FAISS
try:
    import swigfaiss as faiss

    print("✓ Faiss imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Faiss: {e}")
    sys.exit(1)

# Note: HENN is now available through FAISS (IndexHENNFlat)
# No need to import separate Python HENN module
print("✓ Using FAISS HENN implementation")

# Import dataset loaders
try:
    from datasets.ann_datasets import ANNDatasetLoader
    from datasets.synthetic import generate_synthetic_dataset

    print("✓ Dataset modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dataset modules: {e}")
    sys.exit(1)


class BenchmarkResults:
    """Class to store and manage benchmark results"""

    def __init__(self):
        self.results = []

    def add_result(
        self,
        dataset_name: str,
        index_name: str,
        build_time: float,
        search_time: float,
        recall: float,
        qps: float,
        memory_bytes: int,
        index_params: Dict = None,
    ):
        """Add a benchmark result"""
        result = {
            "dataset": dataset_name,
            "index_name": index_name,
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory_bytes": memory_bytes,
            "index_params": str(index_params) if index_params else "",
        }
        self.results.append(result)

    def save_to_csv(self, filename: str):
        """Save results to CSV file"""
        if not self.results:
            print("❌ No results to save")
            return

        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "dataset",
                "index_name",
                "build_time",
                "search_time",
                "recall",
                "qps",
                "memory_bytes",
                "index_params",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"✓ Results saved to {filename}")


def get_memory_usage(index) -> int:
    """Estimate memory usage of an index (simplified)"""
    try:
        if hasattr(index, "ntotal"):
            # Rough estimate: assume float32 vectors + overhead
            d = getattr(index, "d", 128)
            return int(index.ntotal * d * 4 * 2)  # 2x for overhead
        else:
            return 0
    except:
        return 0


def compute_ground_truth(
    vectors: np.ndarray, queries: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ground truth using brute force search"""
    print("Computing ground truth...")
    flat_index = faiss.IndexFlatL2(vectors.shape[1])
    flat_index.add(len(vectors), faiss.swig_ptr(vectors))

    gt_distances = np.zeros((len(queries), k), dtype=np.float32)
    gt_indices = np.zeros((len(queries), k), dtype=np.int64)

    with tqdm(total=1, desc="Ground truth computation", unit="batch") as pbar:
        start_time = time.time()
        flat_index.search(
            len(queries),
            faiss.swig_ptr(queries),
            k,
            faiss.swig_ptr(gt_distances),
            faiss.swig_ptr(gt_indices),
        )
        gt_time = time.time() - start_time
        pbar.update(1)

    print(f"✓ Ground truth computed in {gt_time:.4f}s")
    return gt_distances, gt_indices


def calculate_recall(gt_indices: np.ndarray, pred_indices: np.ndarray, k: int) -> float:
    """Calculate recall@k"""
    if gt_indices is None or pred_indices is None:
        return 0.0

    recall_scores = []
    for i in range(len(gt_indices)):
        gt_set = set(gt_indices[i][:k])
        pred_set = set(pred_indices[i][:k] if i < len(pred_indices) else [])
        if len(gt_set) > 0:
            recall = len(gt_set.intersection(pred_set)) / len(gt_set)
        else:
            recall = 0.0
        recall_scores.append(recall)
    return np.mean(recall_scores)


def benchmark_henn(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    gt_indices: np.ndarray,
    epsnet_strategy: str,
    pgraph_algorithm: str,
    M: int = 32,
    ef_construction: int = 200,
) -> Dict:
    """Benchmark HENN with different configurations using FAISS IndexHENNFlat"""
    index_name = f"HENN+{pgraph_algorithm.upper()}-{epsnet_strategy.title()}"
    print(
        f"\n=== Benchmarking {index_name} (M={M}, efConstruction={ef_construction}) ==="
    )

    try:
        # Create FAISS HENN index
        with tqdm(total=1, desc=f"Creating {index_name}", unit="step") as pbar:
            henn_index = faiss.IndexHENNFlat(vectors.shape[1], M)
            pbar.update(1)

        # Configure HENN parameters
        with tqdm(total=3, desc="Configuring HENN parameters", unit="param") as pbar:
            henn_index.henn.ef_construction = ef_construction
            pbar.update(1)
            henn_index.set_epsnet_strategy(epsnet_strategy)
            pbar.update(1)
            henn_index.set_pgraph_algorithm(pgraph_algorithm)
            pbar.update(1)

        print(f"✓ {index_name} created and configured")

        # Build index with progress bar
        with tqdm(total=1, desc=f"Building {index_name}", unit="batch") as pbar:
            start_time = time.time()
            henn_index.add(len(vectors), faiss.swig_ptr(vectors))
            build_time = time.time() - start_time
            pbar.update(1)

        print(
            f"✓ {index_name} built in {build_time:.4f}s (ntotal: {henn_index.ntotal})"
        )

        # Search with progress bar
        distances = np.zeros((len(queries), k), dtype=np.float32)
        indices = np.zeros((len(queries), k), dtype=np.int64)

        with tqdm(total=1, desc=f"Searching {index_name}", unit="batch") as pbar:
            start_time = time.time()
            henn_index.search(
                len(queries),
                faiss.swig_ptr(queries),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(indices),
            )
            search_time = time.time() - start_time
            pbar.update(1)

        # Calculate metrics
        recall = calculate_recall(gt_indices, indices, k)
        qps = len(queries) / search_time if search_time > 0 else 0
        memory = get_memory_usage(henn_index)

        print(f"✓ {index_name} Recall@{k}: {recall:.4f}, QPS: {qps:.1f}")

        return {
            "name": index_name,
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory": memory,
            "indices": indices,
        }

    except Exception as e:
        print(f"❌ {index_name} failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "name": index_name,
            "build_time": 0,
            "search_time": 0,
            "recall": 0,
            "qps": 0,
            "memory": 0,
            "indices": None,
        }


def benchmark_hnsw(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    gt_indices: np.ndarray,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> Dict:
    """Benchmark HNSW performance"""
    print(f"\n=== Benchmarking IndexHNSW (M={M}, efConstruction={ef_construction}) ===")

    try:
        # Create index
        hnsw_index = faiss.IndexHNSWFlat(vectors.shape[1], M)
        hnsw_index.hnsw.efConstruction = ef_construction

        # Build index with progress bar
        with tqdm(total=1, desc="Building HNSW index", unit="batch") as pbar:
            start_time = time.time()
            hnsw_index.add(len(vectors), faiss.swig_ptr(vectors))
            build_time = time.time() - start_time
            pbar.update(1)

        print(f"✓ IndexHNSW built in {build_time:.4f}s (ntotal: {hnsw_index.ntotal})")

        # Search
        hnsw_index.hnsw.efSearch = ef_search

        distances = np.zeros((len(queries), k), dtype=np.float32)
        indices = np.zeros((len(queries), k), dtype=np.int64)

        with tqdm(total=1, desc="Searching HNSW", unit="batch") as pbar:
            start_time = time.time()
            hnsw_index.search(
                len(queries),
                faiss.swig_ptr(queries),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(indices),
            )
            search_time = time.time() - start_time
            pbar.update(1)

        # Calculate metrics
        recall = calculate_recall(gt_indices, indices, k)
        qps = len(queries) / search_time if search_time > 0 else 0
        memory = get_memory_usage(hnsw_index)

        print(f"✓ IndexHNSW Recall@{k}: {recall:.4f}, QPS: {qps:.1f}")

        return {
            "name": "IndexHNSW",
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory": memory,
            "indices": indices,
        }

    except Exception as e:
        print(f"❌ IndexHNSW failed: {e}")
        return {
            "name": "IndexHNSW",
            "build_time": 0,
            "search_time": 0,
            "recall": 0,
            "qps": 0,
            "memory": 0,
            "indices": None,
        }


def benchmark_lsh(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    gt_indices: np.ndarray,
    nbits: int = 1024,
) -> Dict:
    """Benchmark LSH performance"""
    print(f"\n=== Benchmarking IndexLSH (nbits={nbits}) ===")

    try:
        # Create LSH index
        lsh_index = faiss.IndexLSH(vectors.shape[1], nbits)

        # Build index with progress bar
        with tqdm(total=1, desc="Building LSH index", unit="batch") as pbar:
            start_time = time.time()
            lsh_index.add(len(vectors), faiss.swig_ptr(vectors))
            build_time = time.time() - start_time
            pbar.update(1)

        print(f"✓ IndexLSH built in {build_time:.4f}s (ntotal: {lsh_index.ntotal})")

        # Search
        distances = np.zeros((len(queries), k), dtype=np.float32)
        indices = np.zeros((len(queries), k), dtype=np.int64)

        with tqdm(total=1, desc="Searching LSH", unit="batch") as pbar:
            start_time = time.time()
            lsh_index.search(
                len(queries),
                faiss.swig_ptr(queries),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(indices),
            )
            search_time = time.time() - start_time
            pbar.update(1)

        # Calculate metrics
        recall = calculate_recall(gt_indices, indices, k)
        qps = len(queries) / search_time if search_time > 0 else 0
        memory = get_memory_usage(lsh_index)

        print(f"✓ IndexLSH Recall@{k}: {recall:.4f}, QPS: {qps:.1f}")

        return {
            "name": "IndexLSH",
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory": memory,
            "indices": indices,
        }

    except Exception as e:
        print(f"❌ IndexLSH failed: {e}")
        return {
            "name": "IndexLSH",
            "build_time": 0,
            "search_time": 0,
            "recall": 0,
            "qps": 0,
            "memory": 0,
            "indices": None,
        }


def benchmark_ivf_pq(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    gt_indices: np.ndarray,
    nlist: int = 100,
    m: int = 8,
    nbits: int = 8,
    nprobe: int = 10,
) -> Dict:
    """Benchmark IVF+PQ performance"""
    print(f"\n=== Benchmarking IndexIVFPQ (nlist={nlist}, m={m}, nbits={nbits}) ===")

    try:
        # Create quantizer and IVF+PQ index
        quantizer = faiss.IndexFlatL2(vectors.shape[1])
        ivfpq_index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], nlist, m, nbits)

        # Train the index
        with tqdm(total=1, desc="Training IVF+PQ index", unit="batch") as pbar:
            start_time = time.time()
            ivfpq_index.train(len(vectors), faiss.swig_ptr(vectors))
            train_time = time.time() - start_time
            pbar.update(1)

        # Build index
        with tqdm(total=1, desc="Building IVF+PQ index", unit="batch") as pbar:
            start_time = time.time()
            ivfpq_index.add(len(vectors), faiss.swig_ptr(vectors))
            build_time = time.time() - start_time + train_time
            pbar.update(1)

        print(f"✓ IndexIVFPQ built in {build_time:.4f}s (ntotal: {ivfpq_index.ntotal})")

        # Search
        ivfpq_index.nprobe = nprobe

        distances = np.zeros((len(queries), k), dtype=np.float32)
        indices = np.zeros((len(queries), k), dtype=np.int64)

        with tqdm(total=1, desc="Searching IVF+PQ", unit="batch") as pbar:
            start_time = time.time()
            ivfpq_index.search(
                len(queries),
                faiss.swig_ptr(queries),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(indices),
            )
            search_time = time.time() - start_time
            pbar.update(1)

        # Calculate metrics
        recall = calculate_recall(gt_indices, indices, k)
        qps = len(queries) / search_time if search_time > 0 else 0
        memory = get_memory_usage(ivfpq_index)

        print(f"✓ IndexIVFPQ Recall@{k}: {recall:.4f}, QPS: {qps:.1f}")

        return {
            "name": "IndexIVF+PQ",
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory": memory,
            "indices": indices,
        }

    except Exception as e:
        print(f"❌ IndexIVFPQ failed: {e}")
        return {
            "name": "IndexIVF+PQ",
            "build_time": 0,
            "search_time": 0,
            "recall": 0,
            "qps": 0,
            "memory": 0,
            "indices": None,
        }


def benchmark_nsg(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    gt_indices: np.ndarray,
    R: int = 32,
) -> Dict:
    """Benchmark NSG performance"""
    print(f"\n=== Benchmarking IndexNSG (R={R}) ===")

    try:
        # Create NSG index using Flat storage
        storage = faiss.IndexFlatL2(vectors.shape[1])
        nsg_index = faiss.IndexNSGFlat(vectors.shape[1], R)

        # Build index with progress bar
        with tqdm(total=1, desc="Building NSG index", unit="batch") as pbar:
            start_time = time.time()
            nsg_index.add(len(vectors), faiss.swig_ptr(vectors))
            build_time = time.time() - start_time
            pbar.update(1)

        print(f"✓ IndexNSG built in {build_time:.4f}s (ntotal: {nsg_index.ntotal})")

        # Search
        distances = np.zeros((len(queries), k), dtype=np.float32)
        indices = np.zeros((len(queries), k), dtype=np.int64)

        with tqdm(total=1, desc="Searching NSG", unit="batch") as pbar:
            start_time = time.time()
            nsg_index.search(
                len(queries),
                faiss.swig_ptr(queries),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(indices),
            )
            search_time = time.time() - start_time
            pbar.update(1)

        # Calculate metrics
        recall = calculate_recall(gt_indices, indices, k)
        qps = len(queries) / search_time if search_time > 0 else 0
        memory = get_memory_usage(nsg_index)

        print(f"✓ IndexNSG Recall@{k}: {recall:.4f}, QPS: {qps:.1f}")

        return {
            "name": "IndexNSG",
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory": memory,
            "indices": indices,
        }

    except Exception as e:
        print(f"❌ IndexNSG failed: {e}")
        return {
            "name": "IndexNSG",
            "build_time": 0,
            "search_time": 0,
            "recall": 0,
            "qps": 0,
            "memory": 0,
            "indices": None,
        }


def benchmark_flat(
    vectors: np.ndarray, queries: np.ndarray, k: int, gt_indices: np.ndarray
) -> Dict:
    """Benchmark Flat (brute force) performance"""
    print(f"\n=== Benchmarking IndexFlat (Brute Force) ===")

    try:
        # Create flat index
        flat_index = faiss.IndexFlatL2(vectors.shape[1])

        # Build index with progress bar
        with tqdm(total=1, desc="Building Flat index", unit="batch") as pbar:
            start_time = time.time()
            flat_index.add(len(vectors), faiss.swig_ptr(vectors))
            build_time = time.time() - start_time
            pbar.update(1)

        print(f"✓ IndexFlat built in {build_time:.4f}s (ntotal: {flat_index.ntotal})")

        # Search
        distances = np.zeros((len(queries), k), dtype=np.float32)
        indices = np.zeros((len(queries), k), dtype=np.int64)

        with tqdm(total=1, desc="Searching Flat", unit="batch") as pbar:
            start_time = time.time()
            flat_index.search(
                len(queries),
                faiss.swig_ptr(queries),
                k,
                faiss.swig_ptr(distances),
                faiss.swig_ptr(indices),
            )
            search_time = time.time() - start_time
            pbar.update(1)

        # Calculate metrics
        recall = 1.0  # Flat is exact search, so recall is perfect
        qps = len(queries) / search_time if search_time > 0 else 0
        memory = get_memory_usage(flat_index)

        print(f"✓ IndexFlat Recall@{k}: {recall:.4f}, QPS: {qps:.1f}")

        return {
            "name": "IndexFlat",
            "build_time": build_time,
            "search_time": search_time,
            "recall": recall,
            "qps": qps,
            "memory": memory,
            "indices": indices,
        }

    except Exception as e:
        print(f"❌ IndexFlat failed: {e}")
        return {
            "name": "IndexFlat",
            "build_time": 0,
            "search_time": 0,
            "recall": 0,
            "qps": 0,
            "memory": 0,
            "indices": None,
        }


def load_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load all datasets with specified sample sizes"""
    datasets = {}
    loader = ANNDatasetLoader(data_dir="../../henn-python/datasets/ann_datasets")

    print("Loading datasets...")

    # Load euclidean datasets from ann_datasets with specified sample sizes
    dataset_configs = [
        ("sift-128-euclidean", 20000),
        ("gist-960-euclidean", 20000),
        ("fashion-mnist-784-euclidean", 10000),
    ]

    for dataset_name, sample_size in tqdm(dataset_configs, desc="Loading ANN datasets"):
        try:
            if dataset_name == "fashion-mnist-784-euclidean":
                # train_data, test_data, _ = loader.load_fashion_mnist_784_euclidean(
                #     return_test=True, subset_size=sample_size
                # )
                train_data, test_data, _ = loader.load_fashion_mnist_784_euclidean(
                    return_test=True
                )
            elif dataset_name == "gist-960-euclidean":
                train_data, test_data, _ = loader.load_gist_960_euclidean(
                    return_test=True
                )
            elif dataset_name == "sift-128-euclidean":
                train_data, test_data, _ = loader.load_sift_128_euclidean(
                    return_test=True
                )

            # Take first 1000 queries or all queries if less than 1000
            query_size = min(len(test_data), 1000)
            queries = test_data[:query_size]

            datasets[dataset_name] = (
                train_data.astype(np.float32),
                queries.astype(np.float32),
            )
            print(
                f"✓ Loaded {dataset_name}: {train_data.shape} train, {queries.shape} queries"
            )

        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")

    # Load synthetic dataset with mixture of gaussians (10k samples)
    try:
        synthetic_data = generate_synthetic_dataset(
            n_samples=10000,
            dimension=128,
            distribution="mixture_gaussians",
            distance_metric="l2",  # We want euclidean (L2)
            n_clusters=10,
        )

        # Generate separate queries
        synthetic_queries = generate_synthetic_dataset(
            n_samples=1000,
            dimension=128,
            distribution="mixture_gaussians",
            distance_metric="l2",
            n_clusters=10,
        )

        datasets["synthetic-128-euclidean"] = (
            synthetic_data.astype(np.float32),
            synthetic_queries.astype(np.float32),
        )
        print(
            f"✓ Loaded synthetic-128-euclidean: {synthetic_data.shape} train, {synthetic_queries.shape} queries"
        )

    except Exception as e:
        print(f"❌ Failed to load synthetic dataset: {e}")

    return datasets


def run_full_benchmark(output_file: str = "benchmark_results.csv"):
    """Run comprehensive benchmark on all datasets and indices"""
    print("🚀 Starting Comprehensive ANN Benchmark")
    print("=" * 60)

    # Parameters
    k = 10  # Number of nearest neighbors

    # Load all datasets with specified sample sizes
    datasets = load_datasets()

    if not datasets:
        print("❌ No datasets loaded, exiting")
        return

    # Initialize results storage
    results = BenchmarkResults()

    # Index configurations to test
    henn_configs = [
        ("random", "nsw", "HENN+NSW-Random"),
        ("random", "knn", "HENN+KNN-Random"),
        # ("budget_aware", "nsw", "HENN+NSW-BudgetAware"),
        # ("budget_aware", "knn", "HENN+KNN-BudgetAware"),
    ]

    # Run benchmarks on each dataset
    for dataset_name, (vectors, queries) in datasets.items():
        print(f"\n{'='*60}")
        print(f"BENCHMARKING DATASET: {dataset_name.upper()}")
        print(f"Vectors: {vectors.shape}, Queries: {queries.shape}")
        print(f"{'='*60}")

        # Compute ground truth
        gt_distances, gt_indices = compute_ground_truth(vectors, queries, k)

        # All indices to benchmark
        benchmark_functions = []

        # HENN variants
        for epsnet_strategy, pgraph_alg, config_name in henn_configs:
            benchmark_functions.append(
                (
                    lambda v, q, k, gt, epsnet=epsnet_strategy, pgraph=pgraph_alg: benchmark_henn(
                        v, q, k, gt, epsnet, pgraph
                    ),
                    config_name,
                )
            )

        # Faiss indices
        benchmark_functions.extend(
            [
                (benchmark_hnsw, "IndexHNSW"),
                (benchmark_lsh, "IndexLSH"),
                (benchmark_ivf_pq, "IndexIVF+PQ"),
                (benchmark_nsg, "IndexNSG"),
                (benchmark_flat, "IndexFlat"),
            ]
        )

        # Run each benchmark
        with tqdm(
            total=len(benchmark_functions),
            desc=f"Benchmarking {dataset_name}",
            unit="index",
        ) as pbar:
            for benchmark_func, index_name in benchmark_functions:
                pbar.set_postfix_str(index_name)

                try:
                    result = benchmark_func(vectors, queries, k, gt_indices)

                    # Store result
                    results.add_result(
                        dataset_name=dataset_name,
                        index_name=result["name"],
                        build_time=result["build_time"],
                        search_time=result["search_time"],
                        recall=result["recall"],
                        qps=result["qps"],
                        memory_bytes=result["memory"],
                    )

                except Exception as e:
                    print(f"❌ Failed to benchmark {index_name} on {dataset_name}: {e}")
                    # Store failed result
                    results.add_result(
                        dataset_name=dataset_name,
                        index_name=index_name,
                        build_time=0,
                        search_time=0,
                        recall=0,
                        qps=0,
                        memory_bytes=0,
                    )

                pbar.update(1)

    # Save results
    results.save_to_csv(output_file)

    print("\n🎉 Benchmark completed!")
    print(f"Results saved to: {output_file}")
    print("\nYou can now analyze the results and create plots using the CSV data.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive ANN Benchmark")
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_results_full.csv",
        help="Output CSV file (default: benchmark_results.csv)",
    )

    args = parser.parse_args()

    run_full_benchmark(args.output)
