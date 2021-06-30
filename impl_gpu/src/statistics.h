#ifndef STATISTICA_H
#define STATISTICS_H

#include "master.h"
#include "time.h"
#include <chrono>
#include <unordered_map>

struct statistics {
    /* Timestamp when the stream started without offset*/
    std::chrono::time_point<std::chrono::steady_clock> start_time;

    /* Timestamp when the processing is stopped */
    std::chrono::time_point<std::chrono::steady_clock> end_time;

    /* Timestamp when the stream started + offset as timespec*/
    struct timespec start_time_ts;

    /* Number of tuples that where matched by the join*/
    unsigned processed_output_tuples;

    /* summed processed tuples of both streams */
    unsigned processed_input_tuples;

    /* latency in ns of each tuple that was processed by the join*/
    std::chrono::nanoseconds summed_latency;

    /* overall runtime in ms */
    long runtime;

    /* runtime in idle mode in ms */
    long runtime_idle;

    /* runtime in processing mode */
    long runtime_proc;

    /* runtime in distribution */
    long runtime_build;
    long runtime_probe;
    long runtime_cleanup;

    /* Number of switches to processing mode */
    unsigned switches_to_proc;

    /* stores the number of output tuples per sec */
    std::unordered_map<int, int> output_tuple_map;

    /* stores the cpu usage once per second */
    std::unordered_map<int, double> cpu_usage;
};

void print_statistics(statistics* stats, FILE* outfile, FILE* resultfile, master_ctx_t* ctx);

void write_histogram_stats(statistics* stats, std::string filename);

double get_current_cpu_usage();
#endif
