/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <queue>
#include <unordered_set>
#include <vector>
#include <string>
#include <random>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/impl/ResultHandler.h>

namespace faiss {

/** Implementation of the Hierarchical EpsNet Navigation Graph (HENN)
 * datastructure.
 *
 * HENN uses EPS-Net sampling to build hierarchical layers and proximity graphs
 * (NSW or KNN) to connect points within each layer. The search process is
 * identical to HNSW for compatibility.
 */

struct VisitedTable;
struct DistanceComputer;
struct HENNStats;

template <class C>
struct ResultHandler;

struct SearchParametersHENN : SearchParameters {
    int ef = 16;  // Similar to efSearch in HNSW
    bool check_relative_distance = true;
};

/** Statistics about HENN search */
struct HENNStats {
    size_t n1, n2, n3;     // number of search operations at different levels
    size_t ndis;           // number of distance computations
    size_t nhops;          // number of hops (edge traversals)

    HENNStats() {
        reset();
    }
    void reset() {
        n1 = n2 = n3 = ndis = nhops = 0;
    }

    void combine(const HENNStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        n3 += other.n3;
        ndis += other.ndis;
        nhops += other.nhops;
    }
};

extern HENNStats henn_stats;

/** HENN Layer structure */
struct HENNLayer {
    std::vector<int> indices;           // Global indices of points in this layer
    std::vector<std::vector<int>> edges; // Adjacency list (using local indices within layer)
    
    // Convert global index to local index within this layer
    int global_to_local(int global_idx) const;
    
    // Convert local index to global index
    int local_to_global(int local_idx) const;
    
    // Check if global index exists in this layer
    bool contains(int global_idx) const;
    
    // Get number of points in this layer
    size_t size() const { return indices.size(); }
};

/** The HENN object. It contains the navigation graph links. */
struct HENN {
    /// internal storage of vectors (32 bits: this is expensive)
    typedef int storage_idx_t;

    /// Faiss results are 64-bit
    typedef int64_t idx_t;

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    using C = CMax<float, int64_t>;

    /// same as for HNSW
    struct MinimaxHeap {
        int n, k, nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);
    };

    /// entry point in the index
    storage_idx_t entry_point = -1;

    /// maximum level
    int max_level = -1;

    /// Parameters
    int M = 32;                    // Max connections per node (similar to HNSW)
    int exp_decay = 2;             // Exponential decay factor for layer sizes
    std::string epsnet_strategy = "random";  // "random" or "budget_aware"
    std::string pgraph_algorithm = "nsw";    // "nsw" or "knn"
    
    /// Hierarchical layers
    std::vector<HENNLayer> layers;
    
    /// Construction parameters
    int ef_construction = 200;      // Size of candidate list during construction
    int efSearch = 16;              // Size of candidate list during search (like HNSW)
    bool search_bounded_queue = true; // Use bounded queue during search
    bool verbose = false;

    /// Random number generator
    RandomGenerator rng;

    explicit HENN(int M = 32);

    /// Build the hierarchical structure
    void build_hierarchy(
            const float* points,
            int n,
            int d,
            DistanceComputer& dis);

    /// EPS-Net sampling methods
    std::vector<int> random_sample(
            const std::vector<int>& indices, 
            int target_size);
    
    std::vector<int> budget_aware_sample(
            const float* points,
            int d,
            const std::vector<int>& indices,
            int target_size,
            DistanceComputer& dis);

    /// Build proximity graph for a layer
    void build_layer_graph(
            const float* points,
            int d,
            HENNLayer& layer,
            DistanceComputer& dis);

    /// NSW graph construction for a layer
    void build_nsw_graph(
            const float* points,
            int d,
            HENNLayer& layer,
            DistanceComputer& dis);

    /// KNN graph construction for a layer
    void build_knn_graph(
            const float* points,
            int d,
            HENNLayer& layer,
            DistanceComputer& dis);

    /// Search function (identical to HNSW search pattern)
    HENNStats search(
            DistanceComputer& dis,
            ResultHandler<C>& res,
            VisitedTable& vt,
            const SearchParameters* params = nullptr) const;

    /// Greedy search on upper layers (like HNSW)
    HENNStats greedy_update_nearest(
            DistanceComputer& dis,
            int level,
            storage_idx_t& nearest,
            float& d_nearest) const;

    /// Search from candidates (like HNSW search_from_candidates)
    void search_from_candidates(
            DistanceComputer& dis,
            ResultHandler<C>& res,
            MinimaxHeap& candidates,
            VisitedTable& vt,
            HENNStats& stats,
            int level,
            const SearchParameters* params = nullptr) const;

    /// Reset the index
    void reset();

    /// Get memory usage
    size_t get_memory_usage() const;

    /// Print statistics
    void print_stats() const;
};

/// Global HENN statistics
extern HENNStats henn_stats;

} // namespace faiss
