#include "config.h"
#include "parameter.h"

#include "assert.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <bitset>
#include <omp.h>
#include <sstream>
#include <atomic>

#include "data.h"
#include "master.h"
#include "ringbuffer.h"
#include "messages.h"
#include "worker.h"
#include "cuda_helper.h"
#include "dvs.h"
#include "murmur3.h"
#include "hash_join_gpu.h"
#include "time.h"

/* --- forward declarations --- */
void init_worker (worker_ctx_t *w_ctx, master_ctx_t *ctx);
void process_s (master_ctx_t *ctx, worker_ctx_t *w_ctx);
void process_r (master_ctx_t *ctx, worker_ctx_t *w_ctx);
void interprete_s (worker_ctx_t *w_ctx, int *bitmap);
void interprete_r (worker_ctx_t *w_ctx, int *bitmap);
void expire_outdated_tuples (worker_ctx_t *w_ctx, master_ctx_t *ctx);
void set_num_of_threads (worker_ctx_t *w_ctx);

/**
 * Process in batches instead of in streams (start_stream)
 */
void start_batch(master_ctx_t *ctx, worker_ctx_t *w_ctx){
	
	Timer::Timer timer = Timer::Timer();

        // Compute full batch
        w_ctx->r_available = &(ctx->num_tuples_R);
        w_ctx->s_available = &(ctx->num_tuples_S);
        w_ctx->r_batch_size = ctx->num_tuples_R;
        w_ctx->s_batch_size = ctx->num_tuples_S;

	init_worker(w_ctx, ctx);

	set_num_of_threads(w_ctx);

	if(w_ctx->enable_freq_scaling)
		set_freq(w_ctx->frequency_mode, w_ctx->max_cpu_freq, w_ctx->max_gpu_freq);
	
	process_r(ctx, w_ctx);
	process_s(ctx, w_ctx);
	
	if(w_ctx->enable_freq_scaling)
		set_freq(w_ctx->frequency_mode, w_ctx->min_cpu_freq, w_ctx->min_gpu_freq);

        end_processing(w_ctx);

        // Statistics
        std::cout << "# Write Statistics \n";
        w_ctx->stats.end_time = timer.now();
        w_ctx->stats.runtime = std::chrono::duration_cast
                        <std::chrono::nanoseconds>(w_ctx->stats.end_time - w_ctx->stats.start_time).count();

        print_statistics(&(w_ctx->stats), ctx->outfile, ctx->logfile, ctx);
        //write_histogram_stats(&(w_ctx->stats), "output_tuple_stats.csv");
        //mt_atomic_chunk::print_ht(w_ctx);

        fprintf (ctx->outfile, "# Exit\n");
        exit(0);
}

/*
 * worker main loop
 */
void start_stream(master_ctx_t *ctx, worker_ctx_t *w_ctx){

	init_worker(w_ctx, ctx);

	set_num_of_threads(w_ctx);

	Timer::Timer timer = Timer::Timer();
        w_ctx->stats.start_time = timer.now();
        auto start_time = timer.now();
        auto end_time = timer.now();

	/* is the next tuple from the R stream */
        bool next_is_R;

        /* size of tuple batch to release */
        const int emit_batch_size_r = w_ctx->r_batch_size-1;
        const int emit_batch_size_s = w_ctx->s_batch_size-1;
        const int next_r = emit_batch_size_r + 1;
        const int next_s = emit_batch_size_s + 1;


        while ((ctx->r_available + (ctx->generate_tuples_R * (ctx->r_iterations-1)) + next_r < ctx->num_tuples_R )
			&& (ctx->s_available + (ctx->generate_tuples_S * (ctx->s_iterations-1)) + next_s < ctx->num_tuples_S) ){ /* Still tuples available */
		//std::cout << "s: " << (ctx->s_available + (ctx->generate_tuples_S * (ctx->s_iterations-1)) + next_s < ctx->num_tuples_R) << " " << ctx->num_tuples_R << "\n";
		//std::cout << "r: " << (ctx->r_available + (ctx->generate_tuples_R * (ctx->r_iterations-1)) + next_r  < ctx->num_tuples_S) << " " << ctx->num_tuples_S << "\n";

		/* Update Iterations */
		if (ctx->r_available+next_r > ctx->generate_tuples_R){
			std::cout << "Iteration R: " << ctx->r_iterations << "\n";
			ctx->r_iterations++;
			ctx->r_available = 0;
			ctx->r_processed = 0;
		}
		if (ctx->s_available+next_s > ctx->generate_tuples_S){
			std::cout << "Iteration S: " << ctx->s_iterations << "\n";
			ctx->s_iterations++;
			ctx->s_available = 0;
			ctx->s_processed = 0;
		}


		/* is the next tuple an R or an S tuple? */
                if (ctx->r_available + (ctx->generate_tuples_R * (ctx->r_iterations-1)) + next_r >= ctx->num_tuples_R ){
                        next_is_R = false; // R Stream ended
		} else if (ctx->s_available + (ctx->generate_tuples_S * (ctx->s_iterations-1)) + next_s >= ctx->num_tuples_S){
                        next_is_R = true;  // S Stream ended
                } else {
                        next_is_R = (r_get_tns(ctx,ctx->r_available+next_r) < s_get_tns(ctx,ctx->s_available+next_s));
                }
 

                /* sleep until we the next tuple */
                if (next_is_R) {

			//std::cout << "Sleep time (ms): " <<  std::chrono::duration_cast<std::chrono::milliseconds>(
			//		w_ctx->stats.start_time + ctx->R.t_ns[ctx->r_available+next_r] - timer.now()).count() << "\n";
                        //while(std::chrono::duration_cast<std::chrono::nanoseconds>(w_ctx->stats.start_time 
                        //                + r_get_tns(ctx,ctx->r_available+next_r) - timer.now()).count() > 0);
		
                        std::this_thread::sleep_for(w_ctx->stats.start_time
                                        + std::chrono::nanoseconds(r_get_tns(ctx,ctx->r_available+next_r)) - timer.now());
		} else {

			//std::cout << "Sleep time (ms): " <<  std::chrono::duration_cast<std::chrono::milliseconds>(
			//		w_ctx->stats.start_time + ctx->S.t_ns[ctx->s_available+next_s] - timer.now()).count() << "\n";
                        //while(std::chrono::duration_cast<std::chrono::nanoseconds>(w_ctx->stats.start_time 
                        //                + s_get_tns(ctx,ctx->s_available+next_s) - timer.now()).count() > 0);

                        std::this_thread::sleep_for(w_ctx->stats.start_time
                                       + std::chrono::nanoseconds(s_get_tns(ctx,ctx->s_available+next_s)) - timer.now());
		}

                if (next_is_R){
                        ctx->r_available += emit_batch_size_r;

                        if (ctx->r_available >= ctx->r_processed + ctx->r_batch_size){
				if(w_ctx->enable_freq_scaling)
					set_freq(w_ctx->frequency_mode, w_ctx->max_cpu_freq, w_ctx->max_gpu_freq);
				w_ctx->stats.switches_to_proc++;
				end_time = timer.now();
				w_ctx->stats.runtime_idle += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

				process_r(ctx, w_ctx);

				start_time = timer.now();

				if(w_ctx->enable_freq_scaling)
					set_freq(w_ctx->frequency_mode, w_ctx->min_cpu_freq, w_ctx->min_gpu_freq);
                        }
                } else {
                        ctx->s_available += emit_batch_size_s;

                        if (ctx->s_available >= ctx->s_processed + ctx->s_batch_size){
				if(w_ctx->enable_freq_scaling)
					set_freq(w_ctx->frequency_mode, w_ctx->max_cpu_freq, w_ctx->max_gpu_freq);
				w_ctx->stats.switches_to_proc++;
				end_time = timer.now();
				w_ctx->stats.runtime_idle += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

				process_s(ctx, w_ctx);

				start_time = timer.now();

				if(w_ctx->enable_freq_scaling)
					set_freq(w_ctx->frequency_mode, w_ctx->min_cpu_freq, w_ctx->min_gpu_freq);
                        }

                }

	}
	
	fprintf (ctx->outfile, "# End of Stream\n");

        end_processing(w_ctx);

        /* Statistics */
        std::cout << "# Write Statistics \n";
        w_ctx->stats.end_time = timer.now();
        w_ctx->stats.runtime = std::chrono::duration_cast
                        <std::chrono::nanoseconds>(w_ctx->stats.end_time - w_ctx->stats.start_time).count();;

        print_statistics(&(w_ctx->stats), ctx->outfile, ctx->resultfile, ctx);

        exit(0);

}

void end_processing(worker_ctx_t *w_ctx){
	hj_gpu::end_processing(w_ctx);
}


void set_num_of_threads (worker_ctx_t *w_ctx){
	omp_set_num_threads(4);
}

void process_s (master_ctx_t *ctx, worker_ctx_t *w_ctx){
	hj_gpu::process_s_ht_cpu(ctx, w_ctx);
	
	w_ctx->stats.processed_input_tuples += w_ctx->s_batch_size;
}

void process_r (master_ctx_t *ctx, worker_ctx_t *w_ctx){
	hj_gpu::process_r_ht_cpu(ctx, w_ctx);
	
	w_ctx->stats.processed_input_tuples += w_ctx->r_batch_size;
}

void init_worker (worker_ctx_t *w_ctx, master_ctx_t *ctx){
	/* Allocate output buffer */

	hj_gpu::init(ctx);
}
