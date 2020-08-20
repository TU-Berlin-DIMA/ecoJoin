#ifndef WORKER_H
#define WORKER_H

#include <chrono>
#include "config.h"
#include "parameter.h"

#include <stdio.h>
#include <time.h>

#include "ringbuffer.h"
#include "data.h"
#include "messages.h"
#include "master.h"
#include "statistics.h"

struct worker_ctx_t {
    /* stores statistics */
    statistics  stats;

    /* use a range prediacate to join */
    bool range_predicate;

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

    /* keeps track of time spent idle or processing */ 
    std::chrono::time_point<std::chrono::steady_clock> proc_start_time;
    std::chrono::time_point<std::chrono::steady_clock> idle_start_time;
    
    /* vars to lock x_available */
    // See: https://en.cppreference.com/w/cpp/thread/condition_variable
    std::condition_variable *data_cv;
    std::mutex *data_mutex;

    /* minimum and maximum frequencies */
    unsigned min_cpu_freq;
    unsigned max_cpu_freq;

    /* minimum and maximum frequencies */
    unsigned min_gpu_freq;
    unsigned max_gpu_freq;

    /**
    * Input data stream R
    */
    struct {
	    std::chrono::nanoseconds *t_ns;
	    x_t             *x;
	    y_t             *y;
    } R;
    
    /**
    * Input data stream S
    */
    struct {
	    std::chrono::nanoseconds *t_ns;
	    a_t             *a;
	    b_t             *b;
    } S;

    /**
     * whether the join is processed on the cpu or gpu
     * defined in master.h
     */
    processing_mode_e processing_mode;


    /* scale frequency of gpu, cpu or both */
    frequency_mode_e frequency_mode;


    /* Pointer to the Buffer that the gpu uses to write to */
    int *gpu_output_buffer;

    /* Size of output buffer in bytes */
    int gpu_output_buffer_size;

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

    /* Worker can stop processing */
    bool stop_signal;

    /* Worker acks processing stop and finished statistics update */
    bool stop_signal_ack;

    /** File to write the result tuples to  **/
    FILE *resultfile;
};



void start_batch(master_ctx_t *ctx, worker_ctx_t *w_ctx);
void start_stream(master_ctx_t *ctx, worker_ctx_t *w_ctx);
void end_processing(worker_ctx_t *w_ctx);
void emit_result (worker_ctx_t *w_ctx, unsigned int r, unsigned int s);
#endif  /* WORKER_H */
