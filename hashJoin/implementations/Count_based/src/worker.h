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
    ringbuffer_t *result_queue;      /**< results sent back to master */
   
    /* Message to send already */
    struct {
        unsigned int   pos;   /**< number of tuples already in the message */
        result_msg_t   msg;   /**< the partial message itself */
    } partial_result_msg;


    /* Current start of the window for r and s*/
    unsigned r_first;
    unsigned s_first;

    /* index to newest tuples available by master */
    unsigned *r_available;
    unsigned *s_available;

    /* index to last processed tuple */
    unsigned *r_processed;
    unsigned *s_processed;

    /* vars to lock x_available */
    // See: https://en.cppreference.com/w/cpp/thread/condition_variable
    std::condition_variable *data_cv;
    std::mutex *data_mutex;

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

    /* time to sleep after processing */
    unsigned idle_window_time;
    unsigned process_window_time;
};

void *start_worker(void *ctx);
#endif  /* HASH_JOIN_WORKER_H */
