/*
 * Simple example program demonstrating how to use the HENN index in Faiss.
 * 
 * This example shows:
 * 1. Creating a HENN index with different configurations
 * 2. Adding vectors to the index
 * 3. Searching for nearest neighbors
 * 4. Configuring EPS-Net sampling and proximity graph algorithms
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include <faiss/IndexHENN.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

void generate_random_data(std::vector<float>& data, int n, int d) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 1.0);
    
    data.resize(n * d);
    for (int i = 0; i < n * d; i++) {
        data[i] = dis(gen);
    }
}

void test_henn_flat() {
    std::cout << "=== Testing IndexHENNFlat ===" << std::endl;
    
    int d = 64;    // dimension
    int n = 10000; // number of vectors
    int M = 16;    // max connections per node
    
    // Generate random data
    std::vector<float> database(n * d);
    generate_random_data(database, n, d);
    
    // Create HENN index
    faiss::IndexHENNFlat index(d, M, faiss::METRIC_L2);
    
    // Configure HENN parameters
    index.set_epsnet_strategy("random");      // or "budget_aware"
    index.set_pgraph_algorithm("nsw");        // or "knn"
    index.set_exp_decay(2);                   // exponential decay factor
    index.henn.ef_construction = 200;         // construction parameter
    index.henn.efSearch = 16;                 // default search parameter
    
    std::cout << "Adding " << n << " vectors to index..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    index.add(n, database.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index construction took: " << duration.count() << " ms" << std::endl;
    
    // Print index statistics
    index.henn.print_stats();
    
    // Generate query vectors
    int nq = 100;
    std::vector<float> queries(nq * d);
    generate_random_data(queries, nq, d);
    
    // Search parameters
    int k = 10;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    
    // Create search parameters
    faiss::SearchParametersHENN search_params;
    search_params.ef = 50;  // search parameter
    
    std::cout << "Searching for " << k << " nearest neighbors for " << nq << " queries..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    index.search(nq, queries.data(), k, distances.data(), labels.data(), &search_params);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Search took: " << duration.count() << " μs" << std::endl;
    std::cout << "Average time per query: " << duration.count() / nq << " μs" << std::endl;
    
    // Print some results
    std::cout << "First query results:" << std::endl;
    for (int i = 0; i < k; i++) {
        std::cout << "  " << i << ": label=" << labels[i] << ", distance=" << distances[i] << std::endl;
    }
}

void test_different_configurations() {
    std::cout << "\n=== Testing Different HENN Configurations ===" << std::endl;
    
    int d = 32;
    int n = 5000;
    int M = 16;
    
    std::vector<float> database(n * d);
    generate_random_data(database, n, d);
    
    std::vector<float> queries(10 * d);
    generate_random_data(queries, 10, d);
    
    // Test different configurations
    std::vector<std::pair<std::string, std::string>> configs = {
        {"random", "nsw"},
        {"random", "knn"},
        {"budget_aware", "nsw"},
        {"budget_aware", "knn"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\nTesting EPS-Net: " << config.first 
                  << ", Graph: " << config.second << std::endl;
        
        faiss::IndexHENNFlat index(d, M, faiss::METRIC_L2);
        index.set_epsnet_strategy(config.first);
        index.set_pgraph_algorithm(config.second);
        index.henn.ef_construction = 100;
        
        auto start = std::chrono::high_resolution_clock::now();
        index.add(n, database.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Build time: " << build_time.count() << " ms" << std::endl;
        std::cout << "  Layers: " << index.henn.layers.size() << std::endl;
        
        // Quick search test
        std::vector<float> distances(10);
        std::vector<faiss::idx_t> labels(10);
        
        start = std::chrono::high_resolution_clock::now();
        index.search(1, queries.data(), 10, distances.data(), labels.data());
        end = std::chrono::high_resolution_clock::now();
        
        auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  Search time: " << search_time.count() << " μs" << std::endl;
    }
}

void test_comparison_with_flat() {
    std::cout << "\n=== Comparing HENN with Flat Index ===" << std::endl;
    
    int d = 128;
    int n = 10000;
    int nq = 100;
    int k = 10;
    
    std::vector<float> database(n * d);
    std::vector<float> queries(nq * d);
    generate_random_data(database, n, d);
    generate_random_data(queries, nq, d);
    
    // Test flat index
    std::cout << "Testing IndexFlatL2..." << std::endl;
    faiss::IndexFlatL2 flat_index(d);
    
    auto start = std::chrono::high_resolution_clock::now();
    flat_index.add(n, database.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto flat_build = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::vector<float> flat_distances(nq * k);
    std::vector<faiss::idx_t> flat_labels(nq * k);
    
    start = std::chrono::high_resolution_clock::now();
    flat_index.search(nq, queries.data(), k, flat_distances.data(), flat_labels.data());
    end = std::chrono::high_resolution_clock::now();
    auto flat_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Build time: " << flat_build.count() << " ms" << std::endl;
    std::cout << "  Search time: " << flat_search.count() << " μs" << std::endl;
    
    // Test HENN index
    std::cout << "Testing IndexHENNFlat..." << std::endl;
    faiss::IndexHENNFlat henn_index(d, 16, faiss::METRIC_L2);
    henn_index.henn.ef_construction = 200;
    
    start = std::chrono::high_resolution_clock::now();
    henn_index.add(n, database.data());
    end = std::chrono::high_resolution_clock::now();
    auto henn_build = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::vector<float> henn_distances(nq * k);
    std::vector<faiss::idx_t> henn_labels(nq * k);
    
    faiss::SearchParametersHENN search_params;
    search_params.ef = 50;
    
    start = std::chrono::high_resolution_clock::now();
    henn_index.search(nq, queries.data(), k, henn_distances.data(), henn_labels.data(), &search_params);
    end = std::chrono::high_resolution_clock::now();
    auto henn_search = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Build time: " << henn_build.count() << " ms" << std::endl;
    std::cout << "  Search time: " << henn_search.count() << " μs" << std::endl;
    
    // Calculate recall
    int correct = 0;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (henn_labels[i * k + j] == flat_labels[i * k + l]) {
                    correct++;
                    break;
                }
            }
        }
    }
    
    double recall = (double)correct / (nq * k);
    std::cout << "  Recall@" << k << ": " << recall << std::endl;
    std::cout << "  Speedup: " << (double)flat_search.count() / henn_search.count() << "x" << std::endl;
}

int main() {
    std::cout << "HENN Index Example" << std::endl;
    std::cout << "==================" << std::endl;
    
    try {
        test_henn_flat();
        test_different_configurations();
        test_comparison_with_flat();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}
