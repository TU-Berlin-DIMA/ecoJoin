#ifndef HASH_JOIN_WORKER_H
#define HASH_JOIN_WORKER_H

#include "config.h"
#include "parameter.h"

#include <stdio.h>

#include "ringbuffer.h"
#include "data.h"


struct worker_ctx_t {
    ringbuffer_t *data_S_queue;
    ringbuffer_t *data_R_queue;
    ringbuffer_t *result_queue;      /**< results sent back to master */
    
    struct {
        unsigned int   pos;   /**< number of tuples already in the message */
        result_msg_t   msg;   /**< the partial message itself */
    } partial_result_msg;

    unsigned r_first;
    unsigned r_end;
    
    unsigned s_first;
    unsigned s_end;


    /* time to sleep after processing */
    unsigned sleep_time;

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
        x_t        *x;
        y_t        *y;
    } R;

    /**
     * Those pieces of S that we (need to) have locally for join
     * execution.
     *
     * @see #R
     */
    struct {
        a_t        *a;
        b_t        *b;
    } S;

};

void *start_worker(void *ctx);
#endif  /* HASH_JOIN_WORKER_H */
