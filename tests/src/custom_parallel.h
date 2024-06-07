#ifndef CUSTOM_PARALLEL_H
#define CUSTOM_PARALLEL_H

#include <cmath>
#include <vector>
#include <thread>

template<class Function_>
void test_parallelize(size_t n, size_t nthreads, Function_ f) {
    size_t jobs_per_worker = (n / nthreads) + (n % nthreads > 0);
    size_t start = 0;

    std::vector<std::thread> jobs;
    jobs.reserve(nthreads);
    
    for (size_t w = 0; w < nthreads && start != n; ++w) {
        size_t len = std::min(n - start, jobs_per_worker);
        jobs.emplace_back(f, w, start, len);
        start += len;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define KMEANS_CUSTOM_PARALLEL test_parallelize
#endif
