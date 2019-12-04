#ifndef HASH_JOIN_WORKER_H
#define HASH_JOIN_WORKER_H

#include "config.h"
#include "parameter.h"

#include <stdio.h>

#include "ringbuffer.h"
#include "data.h"
#include "messages.h"
#include "master.h"


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
     * execution.     
     */
    struct {
        x_t        *x;
        y_t        *y;
    } R;

    /**
     * Those pieces of S that we (need to) have locally for join
     * execution.
     */
    struct {
        a_t        *a;
        b_t        *b;
    } S;
    
    /**
     * whether the join is processed on the cpu or gpu
     * defined in master.h
     */
    processing_mode_e processing_mode;


    /** window size for R (in sec) **/
    unsigned window_size_R;

    /** window size for S (in sec) **/
    unsigned window_size_S;
};

void *start_worker(void *ctx);
#endif  /* HASH_JOIN_WORKER_H */
