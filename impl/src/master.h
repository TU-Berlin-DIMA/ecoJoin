#ifndef MASTER_H
#define MASTER_H

#include <stdio.h>
#include <mutex>
#include <iostream>
#include <condition_variable>

#include "ringbuffer.h"
#include "data.h"

enum processing_mode_e{
	cpu1_mode, 
	cpu2_mode, 
	cpu3_mode, 
	cpu4_mode, 
	gpu_mode,
	atomic_mode,
	ht_cpu1_mode, 
	ht_cpu2_mode, 
	ht_cpu3_mode, 
	ht_cpu4_mode
};


enum frequency_mode_e {
	gpu,
	cpu,
	both
};

struct master_ctx_t {

	/** This queue handles the imput data of the R stream **/
	ringbuffer_t *data_R_queue;

	/** This queue handles the imput data of the S stream **/
	ringbuffer_t *data_S_queue;

	/** Result tuples are stored in this queue **/
	ringbuffer_t *result_queue;

	/** data rate  of the R stream **/
	unsigned rate_R;

	/** data rate of the S stream **/
	unsigned rate_S;

	/** window size for R (in sec) **/
	unsigned window_size_R;

	/** window size for S (in sec) **/
	unsigned window_size_S;
	
	/** number of tuples to pregenerate for R **/
	unsigned num_tuples_R;

	/** number of tuples to pregenerate for S **/
	unsigned num_tuples_S;

	/** value range for the join attribute of type @c int (x and a) **/
	unsigned int_value_range;

	/** value range for the join attribute of type @c float (y and b) **/
	long float_value_range;

	/** File to write the result tuples to  **/
	FILE *resultfile;

	/** File to write the output **/
	FILE *outfile;

	FILE *logfile;
	
	/** Enable the data collection thread  **/
	bool data_collection_monitor;

	/* time to sleep after processing */
    	unsigned idle_window_time;
    	unsigned process_window_time;

	/* number of generated tuples */
	long unsigned generate_tuples_R;
	long unsigned generate_tuples_S;

	/* index of the newest currently available tuple*/
	unsigned r_available;
	unsigned s_available;
	
	/* index of the newest currently available tuple*/
	unsigned r_processed;
	unsigned s_processed;
	
	/* minimum and maximum frequencies */
	unsigned min_cpu_freq;
	unsigned max_cpu_freq;

	/* minimum and maximum frequencies */
	unsigned min_gpu_freq;
	unsigned max_gpu_freq;

	/* scale frequency of gpu, cpu or both */
	frequency_mode_e frequency_mode;

	/* vars to lock x_available */
	// See: https://en.cppreference.com/w/cpp/thread/condition_variable
	std::condition_variable data_cv;
	std::mutex data_mutex;
	
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
	 * Number of iterations through dataset
	 * We emulate large datasets by going 
	 * through generated data multiple times
	 */
	unsigned s_iterations;
	unsigned r_iterations;

        /**
         * Whether the join is processed on the cpu or gpu
         */
        processing_mode_e processing_mode;
    
        /* Generate data randomly */
        bool linear_data;

        /* use a range prediacate to join */
        bool range_predicate;
    
        /* enbable time sleep */
        bool time_sleep;

        /* manage time sleep in worker*/
        bool time_sleep_control_in_worker;

        /* batch processing size in tuples */
        unsigned r_batch_size;
        unsigned s_batch_size;
      
        /* GPU parameter */
        unsigned gpu_gridsize;
        unsigned gpu_blocksize;
      
        /* Enable frequency scaling in code*/
        bool enable_freq_scaling;

        /* end when worker ends */ 
        bool end_when_worker_ends;
      
        /* process data in one batch instead of stream-wise */ 
        bool batch_mode;
};
typedef struct master_ctx_t master_ctx_t;

inline
std::chrono::nanoseconds r_get_tns(master_ctx_t *ctx, unsigned r){
	if (ctx->r_iterations > 1) {
		//std::cout << ctx->R.t_ns[r].count() << "\n";
		//std::cout << (std::chrono::nanoseconds((ctx->r_iterations-1) * ctx->generate_tuples_R * (long)(1.e9 / ctx->rate_R))).count() << "\n";
		//std::cout << (ctx->R.t_ns[r % ctx->generate_tuples_R]
	//		+ std::chrono::nanoseconds((ctx->r_iterations-1) * ctx->generate_tuples_R * (long)(1.e9 / ctx->rate_R))).count() << "\n";
		return ctx->R.t_ns[r % ctx->generate_tuples_R] 
			+ std::chrono::nanoseconds((ctx->r_iterations-1) * ctx->generate_tuples_R * (long)(1.e9 / ctx->rate_R));
	} else {
		return ctx->R.t_ns[r];
	}
}

inline
std::chrono::nanoseconds s_get_tns(master_ctx_t *ctx, unsigned s){
	if (ctx->s_iterations > 1) {
		return ctx->S.t_ns[s % ctx->generate_tuples_S] 
			+ std::chrono::nanoseconds((ctx->s_iterations-1) * ctx->generate_tuples_S * (long)(1.e9 / ctx->rate_S));
	} else {
		return ctx->S.t_ns[s];
	}
}
#endif /* MASTER_H */
