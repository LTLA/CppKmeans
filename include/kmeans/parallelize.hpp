#ifndef KMEANS_PARALLELIZE_HPP 
#define KMEANS_PARALLELIZE_HPP

/**
 * @file parallelize.hpp
 * @brief Utilities for parallelization.
 */

#ifndef KMEANS_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

namespace kmeans {

/**
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function to execute a range of tasks.
 *
 * @param num_workers Number of workers.
 * @param num_tasks Number of tasks.
 * @param run_task_range Function to iterate over a range of tasks within a worker.
 *
 * By default, this is an alias to `subpar::parallelize_range()`.
 * However, if the `KMEANS_CUSTOM_PARALLEL` function-like macro is defined, it is called instead. 
 * Any user-defined macro should accept the same arguments as `subpar::parallelize_range()`.
 */
template<typename Task_, class Run_>
void parallelize(int num_workers, Task_ num_tasks, Run_ run_task_range) {
#ifndef KMEANS_CUSTOM_PARALLEL
    // Do NOT make this no-throw as we don't know whether Matrix_::get_observation() might throw.
    subpar::parallelize(num_workers, num_tasks, std::move(run_task_range));
#else
    KMEANS_CUSTOM_PARALLEL(num_workers, num_tasks, run_task_range);
#endif
}

}

#endif
