/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HENN.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <limits>
#include <numeric>
#include <cstring>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

HENNStats henn_stats;

namespace {
using MinimaxHeap = HENN::MinimaxHeap;

// Extract k parameter from result handler (copied from HNSW)
template <typename C>
int extract_k_from_ResultHandler(ResultHandler<C>& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<typename RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

} // namespace

/**************************************************************
 * HENN::MinimaxHeap methods (copied from HNSW)
 **************************************************************/

void HENN::MinimaxHeap::push(storage_idx_t i, float v) {
    if (ids.size() == n) {
        ids.resize(n + 1);
        dis.resize(n + 1);
    }
    ids[k] = i;
    dis[k] = v;
    int j = k;
    while (j > 0 && dis[j] > dis[j - 1]) {
        std::swap(ids[j], ids[j - 1]);
        std::swap(dis[j], dis[j - 1]);
        j--;
    }
    k++;
    nvalid = k;
}

float HENN::MinimaxHeap::max() const {
    return dis[nvalid - 1];
}

int HENN::MinimaxHeap::size() const {
    return nvalid;
}

void HENN::MinimaxHeap::clear() {
    nvalid = k = 0;
}

int HENN::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    if (vmin_out) {
        *vmin_out = dis[0];
    }
    int imin = ids[0];
    k--;
    nvalid--;
    memmove(&ids[0], &ids[1], k * sizeof(ids[0]));
    memmove(&dis[0], &dis[1], k * sizeof(dis[0]));
    return imin;
}

int HENN::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < nvalid; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }
    return n_below;
}

/**************************************************************
 * HENNLayer methods  
 **************************************************************/

int HENNLayer::global_to_local(int global_idx) const {
    auto it = std::find(indices.begin(), indices.end(), global_idx);
    if (it != indices.end()) {
        return std::distance(indices.begin(), it);
    }
    return -1;
}

int HENNLayer::local_to_global(int local_idx) const {
    if (local_idx >= 0 && local_idx < indices.size()) {
        return indices[local_idx];
    }
    return -1;
}

bool HENNLayer::contains(int global_idx) const {
    return std::find(indices.begin(), indices.end(), global_idx) != indices.end();
}

/**************************************************************
 * HENN methods
 **************************************************************/

HENN::HENN(int M) : M(M), rng(1234) {
}

void HENN::build_hierarchy(
        const float* points,
        int n,
        int d,
        DistanceComputer& dis) {
    
    layers.clear();
    entry_point = -1;
    max_level = -1;
    
    // Calculate number of layers
    int num_layers = std::max(1, (int)std::floor(std::log(n) / std::log(exp_decay)));
    
    if (verbose) {
        printf("Building HENN with %d layers for %d points\n", num_layers, n);
    }
    
    // Start with all points
    std::vector<int> current_indices(n);
    std::iota(current_indices.begin(), current_indices.end(), 0);
    
    // Build layers from bottom to top
    for (int level = 0; level < num_layers; level++) {
        HENNLayer layer;
        
        // Calculate target size for this layer
        int target_size = n / std::pow(exp_decay, level);
        target_size = std::max(1, std::min(target_size, (int)current_indices.size()));
        
        if (verbose) {
            printf("Layer %d: sampling %d from %d points\n", 
                   level, target_size, (int)current_indices.size());
        }
        
        // Sample points for this layer using EPS-Net sampling
        std::vector<int> sampled_indices;
        if (epsnet_strategy == "budget_aware") {
            sampled_indices = budget_aware_sample(points, d, current_indices, target_size, dis);
        } else {
            sampled_indices = random_sample(current_indices, target_size);
        }
        
        layer.indices = sampled_indices;
        layer.edges.resize(sampled_indices.size());
        
        // Build proximity graph for this layer
        build_layer_graph(points, d, layer, dis);
        
        layers.push_back(std::move(layer));
        
        // Update current indices for next layer
        current_indices = sampled_indices;
        
        if (current_indices.size() <= 1) {
            break;
        }
    }
    
    // Set entry point and max level (like HNSW)
    max_level = layers.size() - 1;
    if (max_level >= 0 && !layers[max_level].indices.empty()) {
        // Use first point in top layer as entry point
        entry_point = layers[max_level].indices[0];
    }
    
    if (verbose) {
        printf("HENN hierarchy built with %d layers, entry_point=%d, max_level=%d\n", 
               (int)layers.size(), entry_point, max_level);
        print_stats();
    }
}

std::vector<int> HENN::random_sample(
        const std::vector<int>& indices,
        int target_size) {
    
    if (target_size >= indices.size()) {
        return indices;
    }
    
    std::vector<int> result = indices;
    
    // Shuffle and take first target_size elements
    for (int i = result.size() - 1; i > 0; i--) {
        int j = rng.rand_int(i + 1);
        std::swap(result[i], result[j]);
    }
    
    result.resize(target_size);
    return result;
}

std::vector<int> HENN::budget_aware_sample(
        const float* points,
        int d,
        const std::vector<int>& indices,
        int target_size,
        DistanceComputer& dis) {
    
    if (target_size >= indices.size()) {
        return indices;
    }
    
    // Simple budget-aware sampling: select diverse points
    std::vector<int> result;
    std::vector<bool> selected(indices.size(), false);
    
    // Start with a random point
    int first_idx = rng.rand_int(indices.size());
    result.push_back(indices[first_idx]);
    selected[first_idx] = true;
    
    // Iteratively add points that are farthest from already selected points
    while (result.size() < target_size) {
        int best_idx = -1;
        float best_min_dist = 0.0f;
        
        for (int i = 0; i < indices.size(); i++) {
            if (selected[i]) continue;
            
            // Find minimum distance to already selected points
            float min_dist = std::numeric_limits<float>::max();
            for (int selected_global : result) {
                float dist = dis.symmetric_dis(indices[i], selected_global);
                min_dist = std::min(min_dist, dist);
            }
            
            if (min_dist > best_min_dist) {
                best_min_dist = min_dist;
                best_idx = i;
            }
        }
        
        if (best_idx >= 0) {
            result.push_back(indices[best_idx]);
            selected[best_idx] = true;
        } else {
            break;
        }
    }
    
    return result;
}

void HENN::build_layer_graph(
        const float* points,
        int d,
        HENNLayer& layer,
        DistanceComputer& dis) {
    
    if (pgraph_algorithm == "knn") {
        build_knn_graph(points, d, layer, dis);
    } else {
        build_nsw_graph(points, d, layer, dis);
    }
}

void HENN::build_nsw_graph(
        const float* points,
        int d,
        HENNLayer& layer,
        DistanceComputer& dis) {
    
    int n = layer.indices.size();
    if (n <= 1) return;
    
    layer.edges.clear();
    layer.edges.resize(n);
    
    // NSW construction: add points one by one
    for (int i = 1; i < n; i++) {
        int global_idx = layer.indices[i];
        dis.set_query(points + global_idx * d);
        
        // Find closest points among already added points
        std::priority_queue<std::pair<float, int>> candidates;
        
        for (int j = 0; j < i; j++) {
            int other_global = layer.indices[j];
            float dist = dis(other_global);
            candidates.push({-dist, j});  // Use negative distance for min-heap behavior
            
            if (candidates.size() > M) {
                candidates.pop();
            }
        }
        
        // Add bidirectional edges
        while (!candidates.empty()) {
            int local_neighbor = candidates.top().second;
            candidates.pop();
            
            // Add edge i -> local_neighbor
            layer.edges[i].push_back(local_neighbor);
            // Add edge local_neighbor -> i
            layer.edges[local_neighbor].push_back(i);
            
            // Limit degree
            if (layer.edges[local_neighbor].size() > M) {
                // Remove farthest neighbor
                int farthest_idx = -1;
                float farthest_dist = -1.0f;
                int neighbor_global = layer.indices[local_neighbor];
                
                for (int k = 0; k < layer.edges[local_neighbor].size(); k++) {
                    int conn_local = layer.edges[local_neighbor][k];
                    int conn_global = layer.indices[conn_local];
                    float dist = dis.symmetric_dis(neighbor_global, conn_global);
                    if (dist > farthest_dist) {
                        farthest_dist = dist;
                        farthest_idx = k;
                    }
                }
                
                if (farthest_idx >= 0) {
                    layer.edges[local_neighbor].erase(
                        layer.edges[local_neighbor].begin() + farthest_idx);
                }
            }
        }
    }
}

void HENN::build_knn_graph(
        const float* points,
        int d,
        HENNLayer& layer,
        DistanceComputer& dis) {
    
    int n = layer.indices.size();
    if (n <= 1) return;
    
    layer.edges.clear();
    layer.edges.resize(n);
    
    // KNN construction: for each point, find k nearest neighbors
    for (int i = 0; i < n; i++) {
        int global_idx = layer.indices[i];
        dis.set_query(points + global_idx * d);
        
        std::vector<std::pair<float, int>> distances;
        
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            
            int other_global = layer.indices[j];
            float dist = dis(other_global);
            distances.push_back({dist, j});
        }
        
        // Sort by distance and take k closest
        std::sort(distances.begin(), distances.end());
        
        int k = std::min(M, (int)distances.size());
        for (int j = 0; j < k; j++) {
            layer.edges[i].push_back(distances[j].second);
        }
    }
}

HENNStats HENN::search(
        DistanceComputer& dis,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParameters* params) const {
    
    HENNStats stats;
    
    if (entry_point == -1) {
        return stats;
    }

    int k = extract_k_from_ResultHandler(res);
    bool bounded_queue = search_bounded_queue;

    int efSearch_param = efSearch;
    if (params) {
        auto* henn_params = dynamic_cast<const SearchParametersHENN*>(params);
        if (henn_params) {
            efSearch_param = henn_params->ef;
        }
    }

    // Greedy search on upper levels (like HNSW)
    storage_idx_t nearest = entry_point;
    float d_nearest = dis(nearest);

    for (int level = max_level; level >= 1; level--) {
        HENNStats local_stats = greedy_update_nearest(dis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    int ef = std::max(efSearch_param, k);
    if (bounded_queue) { // this is the most common branch
        MinimaxHeap candidates(ef);
        candidates.push(nearest, d_nearest);

        search_from_candidates(dis, res, candidates, vt, stats, 0, params);
    }

    vt.advance();
    return stats;
}

HENNStats HENN::greedy_update_nearest(
        DistanceComputer& dis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) const {
    
    HENNStats stats;
    
    if (level < 0 || level >= layers.size()) {
        return stats;
    }

    const HENNLayer& layer = layers[level];
    
    // Check if nearest point exists in this layer
    if (!layer.contains(nearest)) {
        // If nearest doesn't exist in this layer, pick a random point from this layer
        if (!layer.indices.empty()) {
            RandomGenerator temp_rng(1234);
            int random_idx = temp_rng.rand_int(layer.indices.size());
            nearest = layer.indices[random_idx];
            d_nearest = dis(nearest);
            stats.ndis++;
        }
        return stats;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        int nearest_local = layer.global_to_local(nearest);
        
        if (nearest_local >= 0 && nearest_local < layer.edges.size()) {
            for (int neighbor_local : layer.edges[nearest_local]) {
                int neighbor_global = layer.local_to_global(neighbor_local);
                if (neighbor_global >= 0) {
                    float d = dis(neighbor_global);
                    stats.ndis++;
                    stats.nhops++;
                    
                    if (d < d_nearest) {
                        nearest = neighbor_global;
                        d_nearest = d;
                        changed = true;
                    }
                }
            }
        }
    }
    
    stats.n1++;
    return stats;
}

void HENN::search_from_candidates(
        DistanceComputer& dis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HENNStats& stats,
        int level,
        const SearchParameters* params) const {
    
    if (level < 0 || level >= layers.size()) {
        return;
    }

    const HENNLayer& layer = layers[level];
    
    // can be overridden by search params
    bool do_dis_check = true; // check_relative_distance equivalent 
    int efSearch_param = efSearch;
    const IDSelector* sel = nullptr;
    
    if (params) {
        if (const SearchParametersHENN* henn_params =
                    dynamic_cast<const SearchParametersHENN*>(params)) {
            efSearch_param = henn_params->ef;
            do_dis_check = henn_params->check_relative_distance;
        }
        sel = params->sel;
    }

    typename C::T threshold = res.threshold;
    
    // Initialize visited table with candidates
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch_param) {
                break;
            }
        }

        // Get local index in current layer
        int v0_local = layer.global_to_local(v0);
        if (v0_local < 0 || v0_local >= layer.edges.size()) {
            continue;
        }

        // Explore neighbors
        for (int neighbor_local : layer.edges[v0_local]) {
            int neighbor_global = layer.local_to_global(neighbor_local);
            if (neighbor_global < 0) {
                continue;
            }
            
            if (vt.get(neighbor_global)) {
                continue;
            }
            vt.set(neighbor_global);

            float d = dis(neighbor_global);
            stats.ndis++;
            stats.nhops++;

            if (!sel || sel->is_member(neighbor_global)) {
                if (d < threshold) {
                    if (res.add_result(d, neighbor_global)) {
                        threshold = res.threshold;
                    }
                }
            }

            if (candidates.size() < efSearch_param || d < candidates.max()) {
                candidates.push(neighbor_global, d);
                
                // Remove the farthest candidate if we exceed efSearch
                if (candidates.size() > efSearch_param) {
                    candidates.nvalid = efSearch_param;
                    candidates.k = efSearch_param;
                }
            }
        }
        nstep++;
    }

    stats.n1++;
}

void HENN::reset() {
    layers.clear();
    entry_point = -1;
    max_level = -1;
}

size_t HENN::get_memory_usage() const {
    size_t total = 0;
    for (const auto& layer : layers) {
        total += layer.indices.size() * sizeof(int);
        for (const auto& edges : layer.edges) {
            total += edges.size() * sizeof(int);
        }
        total += layer.edges.size() * sizeof(std::vector<int>);
    }
    total += layers.size() * sizeof(HENNLayer);
    return total;
}

void HENN::print_stats() const {
    printf("HENN Statistics:\n");
    printf("  Layers: %d\n", (int)layers.size());
    for (int i = 0; i < layers.size(); i++) {
        printf("  Layer %d: %d points\n", i, (int)layers[i].size());
    }
    printf("  Memory usage: %.2f MB\n", get_memory_usage() / (1024.0 * 1024.0));
}

} // namespace faiss
