#ifndef HASH_JOIN_WORKER_H
#define HASH_JOIN_WORKER_H

#include "config.h"
#include "parameter.h"

#include <stdio.h>
#include <time.h>

#include "ringbuffer.h"
#include "data.h"
#include "messages.h"
#include "master.h"

struct statistics {
    /* Timestamp when the stream started *stream started */
    struct timespec start_time;

    /* Number of tuples that where matched by the join*/
    unsigned processed_output_tuples;

    /* latency in ms of each tuple that was processed by the join*/
    long summed_latency;
};

struct worker_ctx_t {
    /* stores statistics */
    statistics  stats;
   
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
	struct timespec *t;
        x_t             *x;
        y_t             *y;
    } R;

    /**
     * Those pieces of S that we (need to) have locally for join
     * execution.
     */
    struct {
	struct timespec *t;
        a_t             *a;
        b_t             *b;
    } S;
    
    /**
     * whether the join is processed on the cpu or gpu
     * defined in master.h
     */
    processing_mode_e processing_mode;

    /* Pointer to the Buffer that the gpu uses to write to */
    int *gpu_output_buffer;

    /** window size for R (in sec) **/
    unsigned window_size_R;

    /** window size for S (in sec) **/
    unsigned window_size_S;


    /** number of tuples in S (in sec) **/
    unsigned num_tuples_S;
    
    /** number of tuples in R (in sec) **/
    unsigned num_tuples_R;

    /* time to sleep after processing */
    unsigned idle_window_time;
    unsigned process_window_time;

    /* enbable time sleep */
    bool time_sleep;

    /* manage time sleep in worker*/ 
    bool time_sleep_control_in_worker;

    /* batch processing size in tuples */
    unsigned s_batch_size;

    /* batch processing size in tuples */
    unsigned r_batch_size;

    /* GPU parameter */
    unsigned gpu_gridsize;
    unsigned gpu_blocksize;

    /* Enable frequency scaling in code*/
    bool enable_freq_scaling;
};



void *start_worker(void *ctx);
#endif  /* HASH_JOIN_WORKER_H */
