#ifndef STATISTICS_H
#define STATISTICS_H


#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <array>
#include "statistics.h"
#include "master.h"

//const std::string cpu_usage_call = "{ cat /proc/stat; sleep 1; cat /proc/stat; } |     awk '/^cpu / {usr=$2-usr; sys=$4-sys; idle=$5-idle; iow=$6-iow}  END {total=usr+sys+idle+iow; printf \"%.2f\n\", (total-idle)*100/total}'";
const std::string cpu_usage_call = "/home/adi/efficient-gpu-joins/impl/benchmark/helper/cpu_usage.sh";

void print_statistics (statistics *stats, FILE *outfile, FILE *resultfile, master_ctx_t *ctx){
 	fprintf (outfile, "# Output Tuples       : %u\n", stats->processed_output_tuples);
        fprintf (outfile, "# Throughput (tuple/s): %f\n", 
			stats->processed_output_tuples/((float)ctx->num_tuples_R/(float)ctx->rate_R));
        fprintf (outfile, "# Average Latency (ns): %f\n", 
			(float)stats->summed_latency.count()/(float)stats->processed_output_tuples);
        fprintf (outfile, "# Runtime      (ns): %ld\n", stats->runtime);
        fprintf (outfile, "# Runtime Idle (ns): %ld\n", stats->runtime_idle);
        fprintf (outfile, "# Runtime Proc (ns): %ld\n", stats->runtime_proc);
        fprintf (outfile, "# Switches to Proc : %ld\n", stats->switches_to_proc);
        fprintf (outfile, "# Processed Input  : %ld\n", stats->processed_input_tuples);


        // Processed Tuples, Throughput, Latency, Processed Input Tuples, Runtime, Idle time, Proc time
        fprintf (resultfile, 
			"%u, %f, %f, %u, %ld, %ld, %ld, %ld\n", 
			stats->processed_output_tuples, 
			stats->processed_output_tuples/((float)ctx->num_tuples_R/(float)ctx->rate_R), 
			(float)stats->summed_latency.count()/(float)stats->processed_output_tuples*0.000001, /*ns to ms*/ 
			stats->processed_input_tuples,
			stats->runtime,
			stats->runtime_idle,
			stats->runtime_proc,
			stats->switches_to_proc);
}

void write_histogram_stats(statistics *stats, std::string filename){
	std::fstream  file;
	file.open(filename, std::ios::out);
	for (int i = 0; 
		i < std::chrono::duration_cast<std::chrono::seconds>(stats->end_time - stats->start_time).count(); i++){
		if (stats->output_tuple_map.count(i))
			file << i << ", " << stats->output_tuple_map[i] << ", " << stats->cpu_usage[i]<< "\n"; 
		else
			file << i << ", "<< 0 << ", " << stats->cpu_usage[i] << "\n"; 
	}
	file.close();
}

double get_current_cpu_usage() {
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cpu_usage_call.c_str(), "r"), pclose);
	if (!pipe) {
        	throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        	result += buffer.data();
        }
       	return atof(result.c_str());
}

#endif /* STATISTICS_H */
