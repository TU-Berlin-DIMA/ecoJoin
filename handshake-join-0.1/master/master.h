/**
 * @file
 *
 * Context for the master thread.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010, ETH Zurich, Systems Group
 *
 * $Id: master.h 952 2011-03-09 12:16:34Z jteubner $
 */

#ifndef MASTER_H
#define MASTER_H

#include "config.h"
#include "parameters.h"

#include <pthread.h>
#include <time.h>

#include "comm/ringbuffer.h"
#include "worker/worker.h"
#include "data/data.h"

struct master_ctx_t {

    /** number of workers to spawn */
    unsigned int num_workers;

    /** FIFOs between workers, left to right */
    ringbuffer_t **left_queues;

    /** FIFOs between workers, right to left */
    ringbuffer_t **right_queues;

    /** FIFOs to send results back to master */
    ringbuffer_t **result_queues;

    /** all the worker contexts */
    worker_ctx_t **worker_ctxs;

    /** pthread handles for all workers */
    pthread_t     *workers;

    /**
     * Queue to drive handshake join on the left end of the pipeline.
     * The driver thread will post new data for stream R there and
     * ``acknowledgement'' messages for stream S.
     */
    ringbuffer_t  *data_R_queue;

    /**
     * Back channel from left end of pipeline.  We can pick up expired
     * S tuples here, as well as acknowledgement messages for R.
     */
    ringbuffer_t  *ack_R_queue;

    /**
     * Queue to drive handshake join on the right end of the pipeline.
     * The driver thread will post new data for stream S there and
     * ``acknowledgement'' messages for stream R.
     */
    ringbuffer_t  *data_S_queue;

    /**
     * Back channel from right end of pipeline.  We can pick up expired
     * R tuples here, as well as acknowledgement messages for S.
     */
    ringbuffer_t  *ack_S_queue;


    /**
     * Since we process tuples in chunks, there may be some corner cases
     * to keep time-based window semantics correct.  We handle these in
     * the join driver and use this channel to send such matches to the
     * result collector.
     */
    ringbuffer_t  *result_queue;

    /* --- these are for join_driver(), for use with the GUI version --- */

    /** time interval in which statistics are collected (seconds) */
    unsigned int   collect_interval;

    /** the join driver will wake up in these intervals (micro-seconds) */
    unsigned int   driver_interval;

    /**
     * every @a r_divider times the join driver wakes up, it will
     * generate an R tuple/chunk.
     */
    unsigned int   r_divider;

    /**
     * every @a s_divider times the join driver wakes up, it will
     * generate an S tuple/chunk.
     */
    unsigned int   s_divider;

    /** window size (in tuples/chunks) for input stream R */
    unsigned int   r_size;

    /** window size (in tuples/chunks) for input stream S */
    unsigned int   s_size;

    /* ----- these two are always needed, however ----- */

    /**
     * We create a dummy worker context, such that even the left-
     * and right-most processing cores can see what their ``neighbors''
     * are doing.  (What they will do is see how loaded the neighbors
     * are and only dump work on neighbors if they aren't too loaded.)
     */
    worker_ctx_t  *left_dummy_ctx;

    /** @see #left_dummy_ctx */
    worker_ctx_t  *right_dummy_ctx;

    /** ----- files, mainly for debugging and program inspection ----- */

    /** files as stated on the command line; statistics output */
    FILE          *outfile;

    /** files as stated on the command line; GUI command input */
    FILE          *infile;

    /** file for debugging output; see command line option @c -l */
    FILE          *logfile;

    FILE          *resultfile;

    /**
     * Assume GUI-controlled operation.  For our actual experiments, we
     * pre-generate data and feed them into the workers strictly according
     * to generated timestamps.
     *
     * The alternative is to control the program from a GUI, where the
     * user can modify data rate, window size, etc.  In that case, data
     * is not generated in advance, but on the fly.  Data rates are less
     * predictable in this mode, and data generation may become a
     * bottleneck.
     *
     * @see command line option @c -g
     */
    bool           use_gui;

    /** print more informational messages to #outfile ? */
    bool           verbose;

    /**
     * Prefix to use for data dump files; command line option @c -d.
     */
    char          *data_prefix;

    /* ----- NUMA stuff ----- */

    /**
     * Use NUMA support at runtime (needs NUMA enabled at compile time
     * and kernel NUMA support at runtime).
     */
    bool           use_numa;

    /**
     * Number of NUMA nodes available (numa_max_node() + 1).
     */
    unsigned int   numa_nodes;

    /* ----- these are for use with an experiment on pre-generated data ----- */

    /** number of tuples to pre-generate (and to feed through algo) for R */
    unsigned int   num_tuples_R;

    /** input data rate for stream R (in tuples/sec) */
    unsigned int   rate_R;

    /** window size for R (in seconds) */
    unsigned int   window_size_R;

    /** number of tuples to pre-generate (and to feed through algo) for S */
    unsigned int   num_tuples_S;

    /** input data rate for stream S (in tuples/sec) */
    unsigned int   rate_S;

    /** window size for S (in seconds) */
    unsigned int   window_size_S;

    /** value range for the join attribute of type @c int (x and a) */
    unsigned int   int_value_range;

    /** value range for the join attribute of type @c float (y and b) */
    unsigned int  float_value_range;

    /**
     * Input data stream R, materialized in memory to run experiments;
     * in column-wise storage format (as needed by the worker threads)
     */
    struct {
        struct timespec *t;
        x_t             *x;
        y_t             *y;
        //z_t             *z;
    } R;

    /**
     * Input data stream S, materialized in memory to run experiments;
     * in column-wise storage format (as needed by the worker threads)
     */
    struct {
        struct timespec *t;
        a_t             *a;
        b_t             *b;
        //c_t             *c;
        //d_t             *d;
    } S;

    /**
     * We build up partial result messages here.  As there are
     * #RESULTS_PER_MESSAGE tuples in there, we'll send out the
     * message.
     *
     * @see worker_ctx_t#partial_result_msg
     */
    struct {
        unsigned int   pos;   /**< number of tuples already in the message */
        result_msg_t   msg;   /**< the partial message itself */
    } partial_result_msg;

};
typedef struct master_ctx_t master_ctx_t;

#endif  /* MASTER_H */
