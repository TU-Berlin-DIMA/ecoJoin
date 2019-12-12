/**
 * Supports Time-based windows
 */


#include "parameter.h"

#include "messages.h"
#include "ringbuffer.h"
#include "data.h"
#include "time.h"
#include "master.h"
#include "string.h"

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <condition_variable>
#include "assert.h"

#include "worker.h"

/* ----- forward declarations ----- */
static void start_stream(master_ctx_t *ctx, worker_ctx_t *w_ctx);
static void *collect_results (void *ctx);

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
	printf ("  -p [cpu, gpu]  processing mode (cpu or gpu)\n");
	printf ("  -s MSEC  idle window time\n");
	printf ("  -S MSEC  process window time\n");
	printf ("  -m 	    enable data collection monitor\n");
}

int main(int argc, char **argv) {
	master_ctx_t *ctx = (master_ctx_t *) malloc (sizeof (*ctx));
	int ch;
	pthread_t collector;

	ctx->result_queue = new_ringbuffer(MESSAGE_QUEUE_LENGTH,0);
	ctx->outfile = stdout;
	ctx->logfile = stdout;
	ctx->resultfile = NULL;
	ctx->rate_S = 950;
	ctx->rate_R = 950;
	ctx->window_size_S = 600;
	ctx->window_size_R = 600;
	ctx->num_tuples_S = 1800000;
	ctx->num_tuples_R = 1800000;
	ctx->int_value_range   = 10000;
	ctx->float_value_range = 10000;
	ctx->processing_mode = cpu_mode;
	ctx->idle_window_time = 0;
	ctx->process_window_time = 10;
	ctx->data_collection_monitor = false;
	ctx->r_available = 0;
	ctx->s_available = 0;
	ctx->r_processed = 0;
	ctx->s_processed = 0;

	/* parse command lines */
	while ((ch = getopt (argc, argv, "n:N:O:r:R:w:W:p:s:S:m")) != -1)
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
				if (strncmp(optarg,"cpu",3) == 0)
					ctx->processing_mode = cpu_mode;
				else if (strncmp(optarg,"gpu",3) == 0)
					ctx->processing_mode = gpu_mode;
				else
					ctx->processing_mode = cpu_mode;
				break;
			case 's':
				ctx->idle_window_time = strtol (optarg, NULL, 10);
				break;
			case 'S':
				ctx->process_window_time = strtol (optarg, NULL, 10);
				break;
			case 'm':
				ctx->data_collection_monitor = true;
				break;

			case 'h':
			case '?':
			default:
				usage (); exit (EXIT_SUCCESS);
		}
	}

	fprintf (ctx->outfile, "# Generating input data...\n");
	fprintf (ctx->outfile, "# Using parameters:\n");
	fprintf (ctx->outfile, "#   - num_tuples_R: %u; num_tuples_S: %u\n",
			ctx->num_tuples_R, ctx->num_tuples_S);
	fprintf (ctx->outfile, "#   - rate_R: %u; rate_S: %u\n",
			ctx->rate_R, ctx->rate_S);
	fprintf (ctx->outfile, "#   - window_size_R: %u; window_size_S: %u\n",
			ctx->window_size_R, ctx->window_size_S);
	if (ctx->window_size_S == ctx->window_size_R && ctx->num_tuples_R == ctx->num_tuples_S)
		fprintf (ctx->outfile, "# Stream will run for %f minutes\n",(float)ctx->num_tuples_R/(float)ctx->rate_R/60);
	generate_data (ctx);
	fprintf (ctx->outfile, "# Data generation done.\n");

	if (ctx->processing_mode == cpu_mode)
		fprintf (ctx->outfile, "# Use CPU processing mode\n");
	else if (ctx->processing_mode == gpu_mode)
		fprintf (ctx->outfile, "# Use GPU processing mode\n");
	
	/* Setup worker */
	worker_ctx_t *w_ctx = (worker_ctx_t *) malloc (sizeof (*w_ctx));
	w_ctx->processing_mode = ctx->processing_mode;
	w_ctx->idle_window_time = ctx->idle_window_time;
	w_ctx->process_window_time = ctx->process_window_time;
	w_ctx->S.a = ctx->S.a;
	w_ctx->S.b = ctx->S.b;
	w_ctx->S.t = ctx->S.t;
	w_ctx->R.x = ctx->R.x;
	w_ctx->R.y = ctx->R.y;
	w_ctx->R.t = ctx->R.t;
	w_ctx->r_first = 0;
	w_ctx->s_first = 0;
	w_ctx->r_processed = &(ctx->r_processed);
	w_ctx->s_processed = &(ctx->s_processed);
	w_ctx->r_available = &(ctx->r_available);
	w_ctx->s_available = &(ctx->s_available);
	w_ctx->partial_result_msg;
	w_ctx->partial_result_msg.pos = 0;
	w_ctx->data_cv = &(ctx->data_cv);
	w_ctx->data_mutex = &(ctx->data_mutex);
	/* Statistics*/
	w_ctx->stats.processed_output_tuples = 0;
	w_ctx->stats.summed_latency = 0;
	w_ctx->stats.start_time = (struct timespec) { .tv_sec  = 0, .tv_nsec = 0 };

	printf ("#\n");
	fprintf (ctx->outfile, "# Start Stream\n");
	std::thread first (start_stream, ctx, w_ctx);

	fprintf (ctx->outfile, "# Start Worker\n");
	start_worker(w_ctx);
	
	first.join();

	return EXIT_SUCCESS;
}

/**
 * Handles the stream queues
 */
static void start_stream (master_ctx_t *ctx, worker_ctx_t *w_ctx)
{
	/* is the next tuple from the R stream */
	bool next_is_R;

	/* timestamp for next tuple to send, relative to start of experiment */
	struct timespec t_rel;
	/* timestamp as real time, i.e., t_rel + time measured at begin */
	struct timespec t_real;
	/* difference t_real - t_rel in seconds */
	time_t t_offset;

	struct timespec t_start;
        struct timespec t_end;

	if (hj_gettime (&t_start))
	{
		fprintf (stderr,
			 "Something went wrong with the real time interface.\n");
		 fprintf (stderr, "A call to hj_gettime() failed.\n");
		 exit (EXIT_FAILURE);
	}

	/* add a few seconds delay to play safe */
	t_start.tv_sec += 5;
	t_offset = t_start.tv_sec;
	w_ctx->stats.start_time =  (struct timespec) { .tv_sec  = t_offset, .tv_nsec = 0 };
	int t_last_sec = 0;
	int t_last_nsec = 0;

	while (ctx->r_available < ctx->num_tuples_R || ctx->s_available < ctx->num_tuples_S) {

		/* is the next tuple an R or an S tuple? */
		next_is_R = (ctx->R.t[ctx->r_available].tv_sec * 1000000000L + ctx->R.t[ctx->r_available].tv_nsec)
			< (ctx->S.t[ctx->s_available].tv_sec * 1000000000L + ctx->S.t[ctx->s_available].tv_nsec);

		/* sleep until we have to send the next tuple */
		if (next_is_R)
			t_rel = ctx->R.t[ctx->r_available];
		else
			t_rel = ctx->S.t[ctx->s_available];

		t_real = (struct timespec) { .tv_sec  = t_rel.tv_sec + t_offset,
			.tv_nsec = t_rel.tv_nsec };
	
		hj_nanosleep (&t_real);

		/*
		 * TODO:
		 * Verschiebe um throughput pro sec
		 * schlafe 1 sec
		 * Unterschied im Througput 
		 * Auslastung CPU messen ohne worker
		 */

		/*
		 * TODO: Hier Windows
		 */ 

		/* Update available tuple */
		if (next_is_R){
			ctx->r_available++;

			/*
			 * TODO.
			 * Batchsize kleiner Throughput ist nicht möglich
			 *  TUPLES_PER_CHUNK_R <= Throughput
			 */ 
			/* Notify condition */
			if (ctx->r_available >= ctx->r_processed + TUPLES_PER_CHUNK_R){
				ctx->data_cv.notify_one();
			}
		} else {
			ctx->s_available++;

			/* Notify condition */
			if (ctx->s_available >= ctx->s_processed + TUPLES_PER_CHUNK_S){
				ctx->data_cv.notify_one();
			}
		}
	}
	printf("End of stream\n");

	fprintf (ctx->outfile, "Processed Tuples    : %u\n", w_ctx->stats.processed_output_tuples);
	fprintf (ctx->outfile, "Average Latency (ms): %f\n", (float)w_ctx->stats.summed_latency/(float)w_ctx->stats.processed_output_tuples*0.001);
	exit(0);
}
