/**
 * Declarations around a worker thread (which implements a ``processing
 * core'').
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 Systems Group, ETH Zurich, Switzerland
 *
 * $Id: worker.h 586 2010-08-02 11:15:46Z jteubner $
 */

#ifndef WORKER_H
#define WORKER_H

#include "config.h"
#include "parameters.h"

#include <stdio.h>

#include "comm/ringbuffer.h"
#include "data/data.h"
#include "messages/result.h"

/**
 * Context of a worker thread; all the state that a worker has to
 * keep individually.
 */
struct worker_ctx_t {

    /** Id of each worker (workers will be numbered from "left" to "right") */
    unsigned int id;

    /* ----- FIFO queues ----- */
    ringbuffer_t *left_recv_queue;   /**< receive queue from left neighbor */
    ringbuffer_t *left_send_queue;   /**< send queue to left neighbor */
    ringbuffer_t *right_recv_queue;  /**< receive queue from right neighbor */
    ringbuffer_t *right_send_queue;  /**< send queue to right neighbor */

    ringbuffer_t *result_queue;      /**< results sent back to master */

    /* ----- current R window ----- */

    /** First tuple in the window */
    uint32_t wnd_R_start;

    /** First tuple in window that has not yet been sent to right */
    uint32_t wnd_R_sent;

    /** ``Tuple'' behind last tuple in the window */
    uint32_t wnd_R_end;

    /* ----- current S window ----- */

    /** First tuple in the window */
    uint32_t wnd_S_start;

    /** First tuple in window that has not yet been sent to right */
    uint32_t wnd_S_sent;

    /** ``Tuple'' behind last tuple in the window */
    uint32_t wnd_S_end;

    /* ----- access to the contexts of our neighbors ----- */
    struct worker_ctx_t *left_ctx;   /**< left neighbor */
    struct worker_ctx_t *right_ctx;  /**< right neighbor */

    /**
     * How many workers are there in total?
     * (Currently this information is used to know whether we are
     * the right-most node or not.)
     */
    unsigned int num_workers;

    /* ----- NUMA support ----- */

    /** NUMA node to run this worker on */
    int    numa_node;

    /** Should we physically copy R data over to our node? */
    bool   copy_R;

    /** Should we physically copy S data over to our node? */
    bool   copy_S;

    /* ----- for debugging purposes: log what we are doing ----- */
    FILE  *log;

    /**
     * Those pieces of R that we (need to) have locally for join
     * execution.  "Locally" means local to this NUMA node; the
     * first worker on each NUMA node that receives a chunk of
     * data will physically copy it into local memory.
     *
     * Note that, thanks to column-wise storage, we only have to
     * have the two join attributes local.
     */
    struct {
        x_t             *x;
        y_t             *y;
    } R;

    /**
     * Those pieces of S that we (need to) have locally for join
     * execution.
     *
     * @see #R
     */
    struct {
        a_t             *a;
        b_t             *b;
    } S;

    /**
     * We build up partial result messages here.  As there are
     * #RESULTS_PER_MESSAGE tuples in there, we'll send out the
     * message.
     *
     * @note We're doing a bit of a hack here.  During normal operation,
     *       we'll send out "full" chunks with #RESULTS_PER_MESSAGE tuples
     *       each.  When we flush buffers, we also send out messages with
     *       less tuples.  Thus, when we send out data, we use
     *       <code>n * sizeof (result_t)</code>
     *       as the message size (knowing that #result_msg_t actually is an
     *       array of #result_t).  During collect_results() we look at the
     *       retrieved message size to know how many tuples there were in
     *       the message.
     */
    struct {
        unsigned int   pos;   /**< number of tuples already in the message */
        result_msg_t   msg;   /**< the partial message itself */
    } partial_result_msg;

#ifndef NDEBUG
    /* more debugging */
    bool did_not_send_to_left;
    bool did_not_send_to_right;
#endif
};

typedef struct worker_ctx_t worker_ctx_t;

void * handshake_join (void *);

#endif  /* WORKER_H */
