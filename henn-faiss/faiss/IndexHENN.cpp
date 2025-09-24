/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHENN.h>

#include <omp.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/ResultHandler.h>

namespace faiss {

using storage_idx_t = HENN::storage_idx_t;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void henn_add_vertices(
        IndexHENN& index_henn,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose) {
    size_t d = index_henn.d;
    HENN& henn = index_henn.henn;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    
    if (verbose) {
        printf("henn_add_vertices: adding %zd elements on top of %zd\n", n, n0);
    }

    if (n == 0) {
        return;
    }

    std::unique_ptr<DistanceComputer> dis(
        storage_distance_computer(index_henn.storage));

    // Build the hierarchy for all points (including existing ones)
    henn.build_hierarchy(x - n0 * d, ntotal, d, *dis);

    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }
}

} // namespace

/**************************************************************
 * IndexHENN implementation
 **************************************************************/

IndexHENN::IndexHENN(int d, int M, MetricType metric)
        : Index(d, metric), henn(M) {
    // Create flat storage by default, similar to IndexHNSW
    if (metric == METRIC_L2) {
        storage = new IndexFlatL2(d);
    } else {
        storage = new IndexFlat(d, metric);
    }
    own_fields = true;
    is_trained = true;
}

IndexHENN::IndexHENN(Index* storage, int M)
        : Index(storage->d, storage->metric_type), henn(M), storage(storage) {
    metric_arg = storage->metric_arg;
}

IndexHENN::~IndexHENN() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHENN::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHENNFlat (or variants) instead of IndexHENN directly");
    // henn structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void henn_search(
        const IndexHENN* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHENNFlat (or variants) "
            "instead of IndexHENN directly");
    const HENN& henn = index->henn;

    int ef = henn.ef_construction;
    if (params) {
        if (const SearchParametersHENN* henn_params =
                    dynamic_cast<const SearchParametersHENN*>(params)) {
            ef = henn_params->ef;
        }
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            henn.layers.size() * index->d * ef);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, n3, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HENNStats stats = henn.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    HENNStats combined_stats;
    combined_stats.n1 = n1;
    combined_stats.n2 = n2; 
    combined_stats.n3 = n3;
    combined_stats.ndis = ndis;
    combined_stats.nhops = nhops;
    henn_stats.combine(combined_stats);
}

} // anonymous namespace

void IndexHENN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HENN::C>;
    RH bres(n, distances, labels, k);

    henn_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHENN::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HENN::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    henn_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHENN::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHENNFlat (or variants) instead of IndexHENN directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    henn_add_vertices(*this, n0, n, x, verbose);
}

void IndexHENN::reset() {
    henn.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHENN::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHENN::set_epsnet_strategy(const std::string& strategy) {
    epsnet_strategy = strategy;
    henn.epsnet_strategy = strategy;
}

void IndexHENN::set_pgraph_algorithm(const std::string& algorithm) {
    pgraph_algorithm = algorithm;
    henn.pgraph_algorithm = algorithm;
}

void IndexHENN::set_exp_decay(int decay) {
    exp_decay = decay;
    henn.exp_decay = decay;
}

DistanceComputer* IndexHENN::get_distance_computer() const {
    return storage->get_distance_computer();
}

/**************************************************************
 * IndexHENNFlat implementation
 **************************************************************/

IndexHENNFlat::IndexHENNFlat() {
    is_trained = true;
}

IndexHENNFlat::IndexHENNFlat(int d, int M, MetricType metric)
        : IndexHENN(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHENNPQ implementation
 **************************************************************/

IndexHENNPQ::IndexHENNPQ() = default;

IndexHENNPQ::IndexHENNPQ(
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        MetricType metric)
        : IndexHENN(new IndexPQ(d, pq_m, pq_nbits, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHENNPQ::train(idx_t n, const float* x) {
    IndexHENN::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHENNSQ implementation
 **************************************************************/

IndexHENNSQ::IndexHENNSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHENN(new IndexScalarQuantizer(d, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHENNSQ::IndexHENNSQ() = default;

} // namespace faiss
