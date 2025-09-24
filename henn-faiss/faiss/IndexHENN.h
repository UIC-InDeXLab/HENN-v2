/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>
#include "faiss/Index.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HENN.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHENN;

/** The HENN (Hierarchical EpsNet Navigation Graph) index is a random-access
 * index with a hierarchical structure built using EPS-Net sampling and proximity graphs.
 * Similar to HNSW but with different layer construction strategy using EPS-Net sampling.
 */

struct IndexHENN : Index {
    typedef HENN::storage_idx_t storage_idx_t;

    // the hierarchical structure
    HENN henn;

    // the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;

    // EPS-Net sampling strategy: "random" or "budget_aware"
    std::string epsnet_strategy = "random";

    // Proximity graph algorithm: "nsw" or "knn"
    std::string pgraph_algorithm = "nsw";

    // Exponential decay factor for layer size reduction
    int exp_decay = 2;

    explicit IndexHENN(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexHENN(Index* storage, int M = 32);

    ~IndexHENN() override;

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    /// Set EPS-Net sampling strategy
    void set_epsnet_strategy(const std::string& strategy);

    /// Set proximity graph algorithm
    void set_pgraph_algorithm(const std::string& algorithm);

    /// Set exponential decay factor
    void set_exp_decay(int decay);

    DistanceComputer* get_distance_computer() const override;
};

/** Flat index with a HENN structure to access elements more efficiently. */
struct IndexHENNFlat : IndexHENN {
    IndexHENNFlat();
    IndexHENNFlat(int d, int M, MetricType metric = METRIC_L2);
};

/** PQ index with a HENN structure to access elements more efficiently. */
struct IndexHENNPQ : IndexHENN {
    IndexHENNPQ();
    IndexHENNPQ(
            int d,
            int pq_m,
            int M,
            int pq_nbits = 8,
            MetricType metric = METRIC_L2);
    void train(idx_t n, const float* x) override;
};

/** SQ index with a HENN structure to access elements more efficiently. */
struct IndexHENNSQ : IndexHENN {
    IndexHENNSQ();
    IndexHENNSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
};

} // namespace faiss
