/**
 * Supports Time-based windows
 */

#include "parameter.h"

#include "messages.h"
#include "ringbuffer.h"
#include "data.h"
#include "time.h"
#include "master.h"
#include "worker.h"

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <condition_variable>
#include "assert.h"
#include "string.h"
#include "dvs.h"
#include "hash_join_atomic.h"
#include "hash_join_chunk_chaining.h"

/**
 * Print usage information
 */
static void usage(){
	printf ("Usage:\n");
	printf ("  -n NUM   number of tuples to generate for stream R\n");
	printf ("  -N NUM   number of tuples to generate for stream S\n");
	printf ("  -O FILE  file name to write result stream to\n");
	printf ("  -r RATE  tuple rate for stream R (in tuples/sec)\n");
	printf ("  -R RATE  tuple rate for stream S (in tuples/sec)\n");
	printf ("  -w SIZE  window size for stream R (in seconds)\n");
	printf ("  -W SIZE  window size for stream S (in seconds)\n");
	printf ("  -p []  processing mode\n");
	printf ("  -P enable range predicate\n");
	printf ("  -s SEC  idle window time\n");
	printf ("  -S SEC  process window time\n");
	printf ("  -T enable sleep time window\n");
	printf ("  -t sleep control in worker\n");
	printf ("  -b NUM  batchsize for stream R\n");
	printf ("  -B NUM  batchsize for stream S\n");
	printf ("  -g NUM  GPU gridsize\n");
	printf ("  -G NUM  GPU blocksize\n");
	printf ("  -f enable frequency by stream join\n");
	printf ("  -e end when worker ends\n");
	printf ("  -z process data in one batch\n");
}

int main(int argc, char **argv) {

	master_ctx_t *ctx = (master_ctx_t *) malloc (sizeof (*ctx));
	int ch;

	/* Setup Master */
	ctx->result_queue = new_ringbuffer(MESSAGE_QUEUE_LENGTH,0);
	ctx->outfile = stdout;
	ctx->logfile = fopen ("/dev/null", "w");
	ctx->resultfile = fopen ("/dev/null", "w");
	ctx->rate_S = 950;
	ctx->rate_R = 950;
	ctx->window_size_S = 600;
	ctx->window_size_R = 600;
	ctx->num_tuples_S = 1800000;
	ctx->num_tuples_R = 1800000;
	ctx->int_value_range   = 10000;
	ctx->float_value_range = 10000;
	ctx->processing_mode = cpu1_mode;
	ctx->idle_window_time = 0;
	ctx->process_window_time = 10;
	ctx->r_available = 0;
	ctx->s_available = 0;
	ctx->r_processed = 0;
	ctx->s_processed = 0;
	ctx->max_cpu_freq = 14;
	ctx->min_cpu_freq = 1;
	ctx->r_batch_size = 2000;
	ctx->s_batch_size = 2000;
	ctx->time_sleep = false;
        ctx->time_sleep_control_in_worker = true;
        ctx->gpu_gridsize = 1;
        ctx->gpu_blocksize = 128;
        ctx->enable_freq_scaling = false;
	ctx->range_predicate = false;
	ctx->batch_mode = false;
	ctx->linear_data = false;
	ctx->r_iterations = 1;
	ctx->s_iterations = 1;
	
	/* parse command lines */
	while ((ch = getopt (argc, argv, "n:N:O:r:R:w:W:p:s:S:TtB:b:g:G:f:F:ePzl")) != -1)
	{
		switch (ch)
		{
			case 'n':
				ctx->num_tuples_R = strtol (optarg, NULL, 10);
				break;

			case 'N':
				ctx->num_tuples_S = strtol (optarg, NULL, 10);
				break;

			case 'O':
				if (!(ctx->resultfile = fopen (optarg, "w")))
				{
					fprintf (stderr, "unable to open %s\n", optarg);
					exit (EXIT_FAILURE);
				}
				break;

			case 'r':
				ctx->rate_R = strtol (optarg, NULL, 10);
				break;

			case 'R':
				ctx->rate_S = strtol (optarg, NULL, 10);
				break;

			case 'w':
				ctx->window_size_R = strtol (optarg, NULL, 10);
				break;

			case 'W':
				ctx->window_size_S = strtol (optarg, NULL, 10);
				break;
			case 'p':
				if (strncmp(optarg,"cpu1",4) == 0)
					ctx->processing_mode = cpu1_mode;
				else if (strncmp(optarg,"cpu2",4) == 0)
					ctx->processing_mode = cpu2_mode;
				else if (strncmp(optarg,"cpu3",4) == 0)
					ctx->processing_mode = cpu3_mode;
				else if (strncmp(optarg,"cpu4",4) == 0)
					ctx->processing_mode = cpu4_mode;
				else if (strncmp(optarg,"gpu",3) == 0)
					ctx->processing_mode = gpu_mode;
				else if (strncmp(optarg,"atomic",6) == 0)
					ctx->processing_mode = atomic_mode;
				else if (strncmp(optarg,"ht_cpu1",7) == 0)
					ctx->processing_mode = ht_cpu1_mode;
				else if (strncmp(optarg,"ht_cpu2",7) == 0)
					ctx->processing_mode = ht_cpu2_mode;
				else if (strncmp(optarg,"ht_cpu3",7) == 0)
					ctx->processing_mode = ht_cpu3_mode;
				else if (strncmp(optarg,"ht_cpu4",7) == 0)
					ctx->processing_mode = ht_cpu4_mode;
				else
					ctx->processing_mode = cpu1_mode;
				break;
			case 'P':
				ctx->range_predicate = true;
				break;
			case 's':
				ctx->idle_window_time = strtol (optarg, NULL, 10) * 1000000;
				break;
			case 'S':
				ctx->process_window_time = strtol (optarg, NULL, 10);
				break;
			case 'T':
				ctx->time_sleep = true;
				break;
			case 't':
				ctx->time_sleep_control_in_worker = true;
				break;
			case 'b':
				ctx->r_batch_size = strtol (optarg, NULL,10);
				break;
			case 'B':
				ctx->s_batch_size = strtol (optarg, NULL,10);
				break;
			case 'g':
				ctx->gpu_gridsize = (unsigned)atoi(optarg);
				break;
			case 'G':
				ctx->gpu_blocksize = (unsigned)atoi(optarg);
				break;
			case 'f':
				if (!ctx->enable_freq_scaling)
					ctx->frequency_mode = cpu;
				else 
					ctx->frequency_mode = both;

				ctx->enable_freq_scaling = true;
				// no inits in switch case allowed
				ctx->min_cpu_freq = std::stoi(std::string(optarg).substr(0, std::string(optarg).find('-')))-1;
                                ctx->max_cpu_freq = 
					std::stoi(std::string(optarg).substr(std::string(optarg).find('-')+1, std::string(optarg).length()))-1;
				if (ctx->min_cpu_freq+1 < 1 || ctx->max_cpu_freq+1 > 14){
					std::cout << "CPU Frequency selection must be between 1 and 14\n";
					exit(0);
				}
				break;
			case 'F':
				if (!ctx->enable_freq_scaling)
					ctx->frequency_mode = gpu;
				else 
					ctx->frequency_mode = both;

				ctx->enable_freq_scaling = true;
				ctx->min_gpu_freq = std::stoi(std::string(optarg).substr(0, std::string(optarg).find('-')))-1;
                                ctx->max_gpu_freq = 
					std::stoi(std::string(optarg).substr(std::string(optarg).find('-')+1, std::string(optarg).length()))-1;
				if (ctx->min_gpu_freq+1 < 1 || ctx->max_gpu_freq+1 > 12){
					std::cout << "GPU Frequency selection must be between 1 and 12\n";
					exit(0);
				}
				break;
			case 'z':
				ctx->batch_mode = true;
				break;
			case 'l':
				ctx->linear_data = true;
				break;
			case 'h':
			case '?':
			default:
				usage (); exit (EXIT_SUCCESS);
		}
	}

	if(ctx->enable_freq_scaling)
		set_freq(ctx->frequency_mode, ctx->min_cpu_freq, ctx->min_gpu_freq);

	fprintf (ctx->outfile, "# Generating input data...\n");
	fprintf (ctx->outfile, "# Using parameters:\n");
	fprintf (ctx->outfile, "#   - num_tuples_R: %u; num_tuples_S: %u\n",
			ctx->num_tuples_R, ctx->num_tuples_S);
	fprintf (ctx->outfile, "#   - rate_R: %u; rate_S: %u\n",
			ctx->rate_R, ctx->rate_S);
	fprintf (ctx->outfile, "#   - window_size_R: %u; window_size_S: %u\n",
			ctx->window_size_R, ctx->window_size_S);
	fprintf (ctx->outfile, "#   - r_batch_size: %u; s_batch_size: %u\n",
			ctx->r_batch_size, ctx->s_batch_size);
	if (ctx->window_size_S == ctx->window_size_R && ctx->num_tuples_R == ctx->num_tuples_S)
		fprintf (ctx->outfile, "# Stream will run for %f minutes\n",(float)ctx->num_tuples_R/(float)ctx->rate_R/60);
	generate_data (ctx);
	fprintf (ctx->outfile, "# Data generation done.\n");
	
	/* Setup worker */
	worker_ctx_t *w_ctx = (worker_ctx_t *) malloc (sizeof (*w_ctx));
	w_ctx->range_predicate = ctx->range_predicate;
	w_ctx->r_first = 0;
	w_ctx->s_first = 0;
	w_ctx->r_available = &(ctx->r_available);
	w_ctx->s_available = &(ctx->s_available);
	w_ctx->r_processed = &(ctx->r_processed);
	w_ctx->s_processed = &(ctx->s_processed);
	w_ctx->proc_start_time= std::chrono::steady_clock::now();
	w_ctx->idle_start_time = std::chrono::steady_clock::now();
        w_ctx->min_cpu_freq = ctx->min_cpu_freq;
        w_ctx->max_cpu_freq = ctx->max_cpu_freq;
        w_ctx->min_gpu_freq = ctx->min_gpu_freq;
        w_ctx->max_gpu_freq = ctx->max_gpu_freq;
	w_ctx->S.a = ctx->S.a;
	w_ctx->S.b = ctx->S.b;
	w_ctx->S.t_ns = ctx->S.t_ns;
	w_ctx->R.x = ctx->R.x;
	w_ctx->R.y = ctx->R.y;
	w_ctx->R.t_ns = ctx->R.t_ns;
	w_ctx->processing_mode = ctx->processing_mode;
	w_ctx->frequency_mode = ctx->frequency_mode;
	w_ctx->gpu_output_buffer = NULL;
	w_ctx->gpu_output_buffer_size= 0;
	w_ctx->window_size_S = ctx->window_size_S;
	w_ctx->window_size_R = ctx->window_size_R;
	w_ctx->num_tuples_S = ctx->num_tuples_S;
	w_ctx->num_tuples_R = ctx->num_tuples_R;
	w_ctx->idle_window_time = ctx->idle_window_time;
	w_ctx->process_window_time = ctx->process_window_time;
	w_ctx->time_sleep = ctx->time_sleep;
	w_ctx->time_sleep_control_in_worker = ctx->time_sleep_control_in_worker;
	w_ctx->r_batch_size = ctx->r_batch_size;
	w_ctx->s_batch_size = ctx->s_batch_size;
	w_ctx->gpu_gridsize = ctx->gpu_gridsize;
	w_ctx->gpu_blocksize = ctx->gpu_blocksize;
	w_ctx->enable_freq_scaling = ctx->enable_freq_scaling;
	w_ctx->stop_signal = 0;
	w_ctx->stop_signal_ack = 0;
	w_ctx->resultfile = ctx->resultfile;
		
	/* Setup statistics*/
	w_ctx->stats.processed_output_tuples = 0;
	w_ctx->stats.processed_input_tuples = 0;
	w_ctx->stats.summed_latency = std::chrono::nanoseconds(0);
	w_ctx->stats.runtime_idle = 0;
	w_ctx->stats.runtime_proc = 0;
	w_ctx->stats.start_time = std::chrono::steady_clock::now();
	w_ctx->stats.end_time = std::chrono::steady_clock::now();
	w_ctx->stats.runtime = 0;
	w_ctx->stats.switches_to_proc = 0;
	w_ctx->stats.start_time_ts = (struct timespec) { .tv_sec  = 0, .tv_nsec = 0 };
	//w_ctx->stats.output_tuple_map = std::unordered_map<int, int>();
	//w_ctx->stats.cpu_usage = std::unordered_map<int, double>();

	if (ctx->processing_mode == cpu1_mode)
		fprintf (ctx->outfile, "# Use CPU 1 core processing mode\n\n");
	else if (ctx->processing_mode == cpu2_mode)
		fprintf (ctx->outfile, "# Use CPU 2 processing mode\n\n");
	else if (ctx->processing_mode == cpu3_mode)
		fprintf (ctx->outfile, "# Use CPU 3 processing mode\n\n");
	else if (ctx->processing_mode == cpu4_mode)
		fprintf (ctx->outfile, "# Use CPU 4 processing mode\n\n");
	else if (ctx->processing_mode == gpu_mode)
		fprintf (ctx->outfile, "# Use GPU processing mode\n\n");
	else if (ctx->processing_mode == atomic_mode)
		fprintf (ctx->outfile, "# Use GPU atomic processing mode\n\n");
	else if (ctx->processing_mode == ht_cpu1_mode)
		fprintf (ctx->outfile, "# Use CPU 1 hash join processing mode\n\n");
	else if (ctx->processing_mode == ht_cpu2_mode)
		fprintf (ctx->outfile, "# Use CPU 2 hash join processing mode\n\n");
	else if (ctx->processing_mode == ht_cpu3_mode)
		fprintf (ctx->outfile, "# Use CPU 3 hash join processing mode\n\n");
	else if (ctx->processing_mode == ht_cpu4_mode)
		fprintf (ctx->outfile, "# Use CPU 4 hash join processing mode\n\n");

	if (ctx->range_predicate)
		fprintf (ctx->outfile, "# Use range predicate\n\n");
	else
		fprintf (ctx->outfile, "# Do not use range predicate\n\n");

	if (ctx->batch_mode) {
		fprintf (ctx->outfile, "# Process in one batch\n");
		start_batch(ctx, w_ctx);

	} else {
		fprintf (ctx->outfile, "# Start stream\n");
		start_stream(ctx, w_ctx);
	}

	return EXIT_SUCCESS;
}

