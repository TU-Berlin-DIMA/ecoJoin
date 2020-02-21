#ifndef STATISTICS_H
#define STATISTICS_H


#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include "statistics.h"
#include "master.h"

void print_statistics (statistics *stats, FILE *outfile, FILE *resultfile, master_ctx_t *ctx){
 	fprintf (outfile, "# Output Tuples       : %u\n", stats->processed_output_tuples);
        fprintf (outfile, "# Throughput (tuple/s): %f\n", 
			stats->processed_output_tuples/((float)ctx->num_tuples_R/(float)ctx->rate_R));
        fprintf (outfile, "# Average Latency (ms): %f\n", 
			(float)stats->summed_latency/(float)stats->processed_output_tuples*0.001);
        fprintf (outfile, "# Runtime      (ms): %ld\n", stats->runtime);
        fprintf (outfile, "# Runtime Idle (ms): %ld\n", stats->runtime_idle);
        fprintf (outfile, "# Runtime Proc (ms): %ld\n", stats->runtime_proc);
        fprintf (outfile, "# Switches to Proc : %ld\n", stats->switches_to_proc);
        fprintf (outfile, "# Processed Input  : %ld\n", stats->processed_input_tuples);


        // Processed Tuples, Throughput, Latency, Processed Input Tuples, Runtime, Idle time, Proc time
        fprintf (resultfile, 
			"%u, %f, %f, %u, %ld, %ld, %ld, %ld\n", 
			stats->processed_output_tuples, 
			stats->processed_output_tuples/((float)ctx->num_tuples_R/(float)ctx->rate_R), 
			(float)stats->summed_latency/(float)stats->processed_output_tuples*0.001, 
			stats->processed_input_tuples,
			stats->runtime,
			stats->runtime_idle,
			stats->runtime_proc,
			stats->switches_to_proc);
}

void write_output_tuple_stats(statistics *stats, std::string filename){
	std::fstream  file;
	file.open(filename, std::ios::out);
	for (int i = 0; 
		i < std::chrono::duration_cast<std::chrono::seconds>(stats->end_time - stats->start_time).count(); i++){
		if (stats->output_tuple_map.count(i))
			file << i << ", "<< stats->output_tuple_map[i] << "\n"; 
		else
			file << i << ", "<< 0 << "\n"; 
	}
	file.close();
}
#endif /* STATISTICS_H */
