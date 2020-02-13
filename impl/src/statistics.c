#ifndef STATISTICS_H
#define STATISTICS_H


#include <stdio.h>
#include "statistics.h"
#include "master.h"

void print_statistics (statistics *stats, FILE *outfile, FILE *resultfile, master_ctx_t *ctx){
 	fprintf (outfile, "# Output Tuples       : %u\n", stats->processed_output_tuples);
        fprintf (outfile, "# Throughput (tuple/s): %f\n", 
			stats->processed_output_tuples/((float)ctx->num_tuples_R/(float)ctx->rate_R));
        fprintf (outfile, "# Average Latency (ms): %f\n", 
			(float)stats->summed_latency/(float)stats->processed_output_tuples*0.001);

        // Processed Tuples, Throughput, Latency, Inputtuple
        fprintf (resultfile, 
			"%u, %f, %f, %u \n", 
			stats->processed_output_tuples, 
			stats->processed_output_tuples/((float)ctx->num_tuples_R/(float)ctx->rate_R), 
			(float)stats->summed_latency/(float)stats->processed_output_tuples*0.001, 
			stats->processed_input_tuples,
			stats->runtime,
			stats->runtime_idle,
			stats->runtime_proc);
}
#endif /* STATISTICS_H */
