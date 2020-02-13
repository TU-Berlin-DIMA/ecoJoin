#include "time.h"

struct statistics {
    /* Timestamp when the stream started */
    struct timespec start_time;

    /* Timestamp when the processing is stopped */
    struct timespec end_time;


    /* Number of tuples that where matched by the join*/
    unsigned processed_output_tuples;

    /* summed processed tuples of both streams */
    unsigned processed_input_tuples;

    /* latency in ms of each tuple that was processed by the join*/
    long summed_latency;


    /* overall runtime in ms */
    long runtime;

    /* runtime in idle mode in ms */
    long runtime_idle;

    /* runtime in processing mode */
    long runtime_proc;

};

void print_statistics(statistics *stats, FILE *outfile, FILE *resultfile);
