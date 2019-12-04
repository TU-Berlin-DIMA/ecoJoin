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
#include "assert.h"

#include "worker.h"

/* ----- forward declarations ----- */
static void start_stream(master_ctx_t *ctx);
static void start_worker(master_ctx_t *ctx);
static void *collect_results (void *ctx);
static void emit_result (master_ctx_t *ctx, unsigned int r, unsigned int s);
static inline void flush_result (master_ctx_t *ctx);


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
}

int main(int argc, char **argv) {
	master_ctx_t *ctx = (master_ctx_t *) malloc (sizeof (*ctx));
	int ch;
	pthread_t     collector;

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

	ctx->data_S_queue = new_ringbuffer(MESSAGE_QUEUE_LENGTH*3,0);
	ctx->data_R_queue = new_ringbuffer(MESSAGE_QUEUE_LENGTH*3,0);
	ctx->result_queue = new_ringbuffer(MESSAGE_QUEUE_LENGTH,0);


	/* parse command lines */
	while ((ch = getopt (argc, argv, "n:N:O:r:R:w:W:p:s:S:")) != -1)
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
	generate_data (ctx);
	fprintf (ctx->outfile, "# Data generation done.\n");

	if (ctx->processing_mode == cpu_mode)
		fprintf (ctx->outfile, "# Use CPU processing mode\n");
	else if (ctx->processing_mode == gpu_mode)
		fprintf (ctx->outfile, "# Use GPU processing mode\n");
	
	worker_ctx_t *w_ctx = (worker_ctx_t *) malloc (sizeof (*w_ctx));
	w_ctx->result_queue = ctx->result_queue;
	w_ctx->data_S_queue = ctx->data_S_queue;
	w_ctx->data_R_queue = ctx->data_R_queue;
	w_ctx->processing_mode = ctx->processing_mode;
	w_ctx->idle_window_time = ctx->idle_window_time;
	w_ctx->process_window_time = ctx->process_window_time;
	w_ctx->S.a = ctx->S.a;
	w_ctx->S.b = ctx->S.b;
	w_ctx->R.x = ctx->R.x;
	w_ctx->R.y = ctx->R.y;
	w_ctx->r_first = 0;
	w_ctx->s_first = 0;
	w_ctx->r_end = 0;
	w_ctx->s_end = 0;
	w_ctx->partial_result_msg;
	w_ctx->partial_result_msg.pos = 0;

	fprintf (ctx->outfile, "# Setting up result collector...\n");
	int status = 0;
	status = pthread_create (&collector, NULL, collect_results, ctx);
	assert (status == 0);
	fprintf (ctx->outfile, "# Collector setup done.\n");
	
	printf ("#\n");
	fprintf (ctx->outfile, "# Start Stream\n");
	std::thread first (start_stream, ctx);
	
	fprintf (ctx->outfile, "# Start Worker\n");
	start_worker(w_ctx);
	
	first.join();

	return EXIT_SUCCESS;
}


static inline bool
send_new_R_tuple (master_ctx_t *ctx, unsigned int start_idx, unsigned int size)
{
    chunk_R_msg_t c_msg;
    c_msg.start_idx = start_idx;
    c_msg.size = size;

    core2core_msg_t msg;
    msg.type = new_R_msg;
    msg.msg.chunk_R = c_msg ;

    bool ret = send (ctx->data_R_queue, &msg, sizeof (msg));

    if (!ret)
    {
        fprintf (stderr, "Cannot send R tuple. FIFO queue full.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}

static inline bool
send_new_S_tuple (master_ctx_t *ctx, unsigned int start_idx, unsigned int size)
{   
    chunk_S_msg_t c_msg;
    c_msg.start_idx = start_idx;
    c_msg.size = size;

    core2core_msg_t msg;
    msg.type = new_S_msg;
    msg.msg.chunk_S = c_msg ;

    bool ret = send (ctx->data_S_queue, &msg, sizeof (msg));

    if (!ret)
    {
        fprintf (stderr, "Cannot send S tuple. FIFO queue full.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}


/**
 * Handles the stream queues
 */
static void start_stream (master_ctx_t *ctx)
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
	int t_last_sec = 0;
	int t_last_nsec = 0;

	/* current tuple marker */
	unsigned r = 0;
	unsigned s = 0;

	unsigned r_end = 0;
	unsigned s_end = 0;
	
	unsigned s_last_sent = 0;
	unsigned r_last_sent = 0;

	while (r < ctx->num_tuples_R || s < ctx->num_tuples_S)
	{
		/* is the next tuple an R or an S tuple? */
		next_is_R = (ctx->R.t[r].tv_sec * 1000000000L + ctx->R.t[r].tv_nsec)
			< (ctx->S.t[s].tv_sec * 1000000000L + ctx->S.t[s].tv_nsec);

		/* sleep until we have to send the next tuple */
		if (next_is_R)
			t_rel = ctx->R.t[r];
		else
			t_rel = ctx->S.t[s];

		t_real = (struct timespec) { .tv_sec  = t_rel.tv_sec + t_offset,
			.tv_nsec = t_rel.tv_nsec };
	
		/* Print time */
		/*const uint TIME_FMT = strlen("2012-12-31 12:59:59.123456789") + 1;
		char timestr[TIME_FMT];

		struct timeval t;

		if (timespec2str(timestr, sizeof(timestr), &t_real) != 0) {
			printf("timespec2str failed!\n");
		} else {
			printf("CLOCK_REALTIME: time=%s\n", timestr);
		}*/

		hj_nanosleep (&t_real);

		if (next_is_R){
			send_new_R_tuple (ctx, r, TUPLES_PER_CHUNK_R);
			//r++;
			r += TUPLES_PER_CHUNK_R;
		} else {
			send_new_S_tuple (ctx, s, TUPLES_PER_CHUNK_S);
			//s++;
			s += TUPLES_PER_CHUNK_S;
		}
		
		/*if (!next_is_R){
			int s_ = s;
			for (unsigned int r_ = r_last_sent; r_ < r; r_++)
			{
				const a_t a = ctx->S.a[s_] - ctx->R.x[r_];
				const b_t b = ctx->S.b[s_] - ctx->R.y[r_];
				if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
					emit_result (ctx, r_, s_);
			}
			s++;
			s_last_sent = s;
		}


		if (next_is_R){
			int r_ = r;
			for (unsigned int s_ = s_last_sent; s_ < s; s_++)
			{
				const a_t a = ctx->S.a[s_] - ctx->R.x[r_];
				const b_t b = ctx->S.b[s_] - ctx->R.y[r_];
				if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
					emit_result (ctx, r_, s_);
			}
			r++;
			r_last_sent = r;
		}*/
	}
	printf("End of stream\n");
	exit(0);

}

/**
 * Collect result tuples from all the workers.  This function is run
 * as its own thread.  It will spin and try to collect as many result
 * tuples from all the workers as it can.
 *
 * The function will serialize the result tuples to the output file
 * given on the command line (see master_ctx_t#outfile).  Also, the
 * function will report throughput statistics (i.e., result output
 * rates) approximately every second.
 *
 * @param arg a master context (signature of this function must match
 *            pthread_create(3).
 *
 * @return This function will never return.
 */
static void *
collect_results (void *arg)
{
	master_ctx_t    *ctx = (master_ctx_t *) arg;
	result_msg_t     msg;
	struct timespec  t_last;
	struct timespec  t_now;
	struct timespec  t_next;
	unsigned int     n_last = 0;
	unsigned int     n_now = 0;
	unsigned int     msg_size;
	unsigned int     num_result;

	hj_gettime (&t_last);

	t_next = t_last;

	while (true)
	{
		t_next.tv_nsec += COLLECT_INTERVAL;

		t_next.tv_sec  += t_next.tv_nsec / 1000000000L;
		t_next.tv_nsec %= 1000000000L;

		hj_nanosleep (&t_next);

		/* consume results from all result queues */
		while (! empty_ (ctx->result_queue))
		{
			msg_size = receive (ctx->result_queue, &msg);

			/*
			 * The message size tells us how many result tuples there
			 * are encoded in the message.
			 * See worker_ctx_t#partial_result_msg.
			 */
			num_result = msg_size / sizeof (result_t);

			for (unsigned int j = 0; j < num_result; j++)
			{
				if (ctx->resultfile)
				{
					fprintf (ctx->resultfile,
							"%4lu.%09lu | %8u | %8.2f | %20s || "
							"%4lu.%09lu | %8u | %8.2f | %10.2f | %5s || "
							"\n",
							ctx->R.t[msg[j].r].tv_sec,
							ctx->R.t[msg[j].r].tv_nsec,
							ctx->R.x[msg[j].r], ctx->R.y[msg[j].r],
							ctx->R.z[msg[j].r],
							ctx->S.t[msg[j].s].tv_sec,
							ctx->S.t[msg[j].s].tv_nsec,
							ctx->S.a[msg[j].s], ctx->S.b[msg[j].s],
							ctx->S.c[msg[j].s],
							ctx->S.d[msg[j].s] ? "true" : "false"
							);
				}
				/*
				printf (	"%4lu.%09lu | %8u | %8.2f | %20s || "
							"%4lu.%09lu | %8u | %8.2f | %10.2f | %5s || "
							"\n",
							ctx->R.t[msg[j].r].tv_sec,
							ctx->R.t[msg[j].r].tv_nsec,
							ctx->R.x[msg[j].r], ctx->R.y[msg[j].r],
							ctx->R.z[msg[j].r],
							ctx->S.t[msg[j].s].tv_sec,
							ctx->S.t[msg[j].s].tv_nsec,
							ctx->S.a[msg[j].s], ctx->S.b[msg[j].s],
							ctx->S.c[msg[j].s],
							ctx->S.d[msg[j].s] ? "true" : "false"
							);
							*/
				n_now++;
			}
		}

		/* check time stamp to report output data rate */
		hj_gettime (&t_now);

		/* report approximately every second */
		if (t_now.tv_sec != t_last.tv_sec)
		{
			const double sec
				= ((double) ((t_now.tv_sec * 1000000000L + t_now.tv_nsec)
							- (t_last.tv_sec * 1000000000L + t_last.tv_nsec)))
				/ 1e9;

			fprintf (stdout, "%u result tuples retrieved in %5.3f sec "
					"(%6.2f tuples/sec)\n",
					n_now - n_last, sec, ((double) (n_now - n_last)) / sec);

			if (ctx->resultfile && n_last != n_now)
				fflush (ctx->resultfile);

			t_last = t_now;
			n_last = n_now;
		}

	}

	/* never reached */
	return NULL;
}

/*static void
emit_result (master_ctx_t *ctx, unsigned int r, unsigned int s)
{
    //LOG(ctx->logfile, "result: r = %u, s = %u", r, s);

    assert (ctx->partial_result_msg.pos < RESULTS_PER_MESSAGE);

    ctx->partial_result_msg.msg[ctx->partial_result_msg.pos]
        = (result_t) { .r = r, .s = s };

    ctx->partial_result_msg.pos++;

    if (ctx->partial_result_msg.pos == RESULTS_PER_MESSAGE)
        flush_result (ctx);
}*/

/**
 * Flush queue to result collector; see emit_result().
 */
/*static inline void
flush_result (master_ctx_t *ctx)
{
    if (ctx->partial_result_msg.pos != 0)
    {
        //LOG(ctx->logfile, "flushing result buffer (%u tuples)",
        //        ctx->partial_result_msg.pos);

        send (ctx->result_queue, &ctx->partial_result_msg.msg,
                ctx->partial_result_msg.pos * sizeof (result_t));

        ctx->partial_result_msg.pos = 0;
    }
    else
    {
        //LOG(ctx->logfile, "flushing requested, but nothing to flush");
    }
}*/

/*
 * Dummy woker
 * Just removing queue items
 */
static void start_worker(master_ctx_t *ctx)
{
	ringbuffer_t *r = ctx->data_R_queue;	
	ringbuffer_t *s = ctx->data_S_queue;	

	message_t *msg;

	while(true)
	{
		peek(r,(void**) &msg);
		receive(r, msg);
		peek(s, (void**)&msg);
		receive(s, msg);
	}

}
