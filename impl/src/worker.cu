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
#include "kernels.h"
#include "cuda_helper.h"
#include "dvs.h"
#include "hash_join.h"
#include "murmur3.h"
#include "hash_join_multithreaded.h"
#include "hash_join_atomic.h"
#include "hash_join_chunk_chaining.h"
#include "time.h"

/* --- forward declarations --- */
void init_worker (worker_ctx_t *w_ctx);
void process_s (master_ctx_t *ctx, worker_ctx_t *w_ctx);
void process_r (master_ctx_t *ctx, worker_ctx_t *w_ctx);
void process_s_cpu (worker_ctx_t *w_ctx, unsigned threads);
void process_r_cpu (worker_ctx_t *w_ctx, unsigned threads);
void process_s_gpu (worker_ctx_t *w_ctx);
void process_r_gpu (worker_ctx_t *w_ctx);
void process_s_gpu_atomics (worker_ctx_t *w_ctx);
void process_r_gpu_atomics (worker_ctx_t *w_ctx);
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

	init_worker(w_ctx);

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

	init_worker(w_ctx);

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
                                        + r_get_tns(ctx,ctx->r_available+next_r) - timer.now());
		} else {
			//std::cout << "Sleep time (ms): " <<  std::chrono::duration_cast<std::chrono::milliseconds>(
			//		w_ctx->stats.start_time + ctx->S.t_ns[ctx->s_available+next_s] - timer.now()).count() << "\n";
                        //while(std::chrono::duration_cast<std::chrono::nanoseconds>(w_ctx->stats.start_time 
                        //                + s_get_tns(ctx,ctx->s_available+next_s) - timer.now()).count() > 0);

                        std::this_thread::sleep_for(w_ctx->stats.start_time
                                       + s_get_tns(ctx,ctx->s_available+next_s) - timer.now());
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
		
		expire_outdated_tuples (w_ctx,ctx);

	}
	
	fprintf (ctx->outfile, "# End of Stream\n");

        end_processing(w_ctx);

        /* Statistics */
        std::cout << "# Write Statistics \n";
        w_ctx->stats.end_time = timer.now();
        w_ctx->stats.runtime = std::chrono::duration_cast
                        <std::chrono::nanoseconds>(w_ctx->stats.end_time - w_ctx->stats.start_time).count();;

        print_statistics(&(w_ctx->stats), ctx->outfile, ctx->logfile, ctx);
        write_histogram_stats(&(w_ctx->stats), "output_tuple_stats.csv");
        //mt_atomic_chunk::print_ht(w_ctx);

        exit(0);

}

void end_processing(worker_ctx_t *w_ctx){

	// set processed tuples
	if (w_ctx->processing_mode == ht_cpu1_mode
		|| w_ctx->processing_mode == ht_cpu2_mode
		|| w_ctx->processing_mode == ht_cpu3_mode
		|| w_ctx->processing_mode == ht_cpu4_mode){
		w_ctx->stats.processed_output_tuples = mt_atomic_chunk::processed_tuples;
	}
}


void set_num_of_threads (worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == ht_cpu1_mode){
		omp_set_num_threads(1);
        } else if (w_ctx->processing_mode == ht_cpu2_mode){
		omp_set_num_threads(2);
        } else if (w_ctx->processing_mode == ht_cpu3_mode){
		omp_set_num_threads(3);
        } else if (w_ctx->processing_mode == ht_cpu4_mode){
		omp_set_num_threads(4);
       	} else if (w_ctx->processing_mode == cpu1_mode){
		omp_set_num_threads(1);
	} else if (w_ctx->processing_mode == cpu2_mode){
		omp_set_num_threads(2);
	} else if (w_ctx->processing_mode == cpu3_mode){
		omp_set_num_threads(3);
	} else if (w_ctx->processing_mode == cpu4_mode){
		omp_set_num_threads(4);
	}
}

void process_s (master_ctx_t *ctx, worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == cpu1_mode){
		process_s_cpu(w_ctx,1);
	} else if (w_ctx->processing_mode == cpu2_mode){
		process_s_cpu(w_ctx,2);
	} else if (w_ctx->processing_mode == cpu3_mode){
		process_s_cpu(w_ctx,3);
	} else if (w_ctx->processing_mode == cpu4_mode){
		process_s_cpu(w_ctx,4);
	} else if (w_ctx->processing_mode == gpu_mode){
		process_s_gpu(w_ctx);
	} else if (w_ctx->processing_mode == atomic_mode){
		process_s_gpu_atomics(w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu1_mode){
		//process_s_ht_cpu(w_ctx,1);
		//mt_tbb::process_s_ht_cpu(w_ctx,1);
		//mt_atomic::process_s_ht_cpu(w_ctx,1);
		mt_atomic_chunk::process_s_ht_cpu(ctx, w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu2_mode){
		mt_atomic_chunk::process_s_ht_cpu(ctx, w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu3_mode){
		mt_atomic_chunk::process_s_ht_cpu(ctx, w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu4_mode){
		mt_atomic_chunk::process_s_ht_cpu(ctx, w_ctx);
	}
	w_ctx->stats.processed_input_tuples += w_ctx->s_batch_size;
}

void process_r (master_ctx_t *ctx, worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == cpu1_mode){
		process_r_cpu(w_ctx,1);
	} else if (w_ctx->processing_mode == cpu2_mode){
		process_r_cpu(w_ctx,2);
	} else if (w_ctx->processing_mode == cpu3_mode){
		process_r_cpu(w_ctx,3);
	} else if (w_ctx->processing_mode == cpu4_mode){
		process_r_cpu(w_ctx,4);
	} else if (w_ctx->processing_mode == gpu_mode){
		process_r_gpu(w_ctx);
	} else if (w_ctx->processing_mode == atomic_mode){
		process_r_gpu_atomics(w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu1_mode){
		mt_atomic_chunk::process_r_ht_cpu(ctx, w_ctx);
		//mt_atomic::process_r_ht_cpu(w_ctx,1);
		//mt_tbb::process_r_ht_cpu(w_ctx,1);
		//process_r_ht_cpu(w_ctx,1);
	} else if (w_ctx->processing_mode == ht_cpu2_mode){
		mt_atomic_chunk::process_r_ht_cpu(ctx, w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu3_mode){
		mt_atomic_chunk::process_r_ht_cpu(ctx, w_ctx);
	} else if (w_ctx->processing_mode == ht_cpu4_mode){
		mt_atomic_chunk::process_r_ht_cpu(ctx, w_ctx);
	}
	w_ctx->stats.processed_input_tuples += w_ctx->r_batch_size;
}

void init_worker (worker_ctx_t *w_ctx){
	/* Allocate output buffer */
	if (w_ctx->processing_mode == gpu_mode ||
			w_ctx->processing_mode == atomic_mode){
		
		// Estimated maximal buffersize
		if (w_ctx->s_batch_size > w_ctx->r_batch_size )
			w_ctx->gpu_output_buffer_size = w_ctx->num_tuples_R * w_ctx->s_batch_size;
		else
			w_ctx->gpu_output_buffer_size = w_ctx->num_tuples_S * w_ctx->r_batch_size;
		
		CUDA_SAFE(cudaHostAlloc((void**)&(w_ctx->gpu_output_buffer), w_ctx->gpu_output_buffer_size, 0));
		std::memset(w_ctx->gpu_output_buffer, 0, w_ctx->gpu_output_buffer_size);
	}

	if (w_ctx->processing_mode == ht_cpu1_mode ||
		w_ctx->processing_mode == ht_cpu2_mode ||
		w_ctx->processing_mode == ht_cpu3_mode || 
		w_ctx->processing_mode == ht_cpu4_mode) {
		mt_atomic_chunk::init_ht();
	}
		
}

/* Process TUPLES_PER_CHUNK_S Tuples on the gpu with nested loop join
 * Similar to HELLS JOIN
 */
void process_s_gpu (worker_ctx_t *w_ctx){
    	const unsigned s_processed = *(w_ctx->s_processed);
    	const unsigned r_first = w_ctx->r_first;
    	const unsigned r_processed = *(w_ctx->r_processed);
    	
	if (r_processed - r_first > 0){
		
		/* Start kernel */
		if (w_ctx->range_predicate){
			compare_kernel_new_s_range<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				w_ctx->s_batch_size,
				r_processed - r_first);
		} else {
			compare_kernel_new_s<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				w_ctx->s_batch_size,
				r_processed - r_first);
		}

		CUDA_SAFE(cudaDeviceSynchronize());
		
		interprete_s(w_ctx, w_ctx->gpu_output_buffer);
	}

	/* Update processed tuples index */
	*(w_ctx->s_processed) += w_ctx->s_batch_size;

}

	
/* Process TUPLES_PER_CHUNK_R Tuples on the gpu with nested loop join
 * Similar to HELLS JOIN
 */
void process_r_gpu (worker_ctx_t *w_ctx){
    	const unsigned r_processed = *(w_ctx->r_processed);
    	const unsigned s_first = w_ctx->s_first;
    	const unsigned s_processed = *(w_ctx->s_processed);
    	
	if (s_processed - s_first > 0){
		
		/* Start kernel */
		if (w_ctx->range_predicate){
			compare_kernel_new_r_range<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				w_ctx->r_batch_size);
		} else {
			compare_kernel_new_r<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				w_ctx->r_batch_size);
		}

		CUDA_SAFE(cudaDeviceSynchronize());

		interprete_r(w_ctx, w_ctx->gpu_output_buffer);
	}

	/* Update processed tuples index */
	*(w_ctx->r_processed) += w_ctx->r_batch_size;

}

/* Process Tuples on the gpu with nested loop join
 * atomics version without cpu interprete step
 */
void process_s_gpu_atomics (worker_ctx_t *w_ctx){
    	const unsigned s_processed = *(w_ctx->s_processed);
    	const unsigned r_first = w_ctx->r_first;
    	const unsigned r_processed = *(w_ctx->r_processed);
    	
	if (r_processed - r_first > 0){
		if (w_ctx->range_predicate){
			compare_kernel_new_s_atomics_range<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer,
			       	w_ctx->gpu_output_buffer_size/8,
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				w_ctx->s_batch_size,
				r_processed - r_first);
		} else {
			compare_kernel_new_s_atomics<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer,
			       	w_ctx->gpu_output_buffer_size/8,
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				w_ctx->s_batch_size,
				r_processed - r_first);
		}

		CUDA_SAFE(cudaDeviceSynchronize());
	}

	for (int i = 0; i < w_ctx->gpu_output_buffer_size/8; i = i + 2){
		if (w_ctx->gpu_output_buffer[i] == 0 
				&& w_ctx->gpu_output_buffer[i+1] == 0
				&& w_ctx->gpu_output_buffer[i+2] == 0)
			break;
		emit_result(w_ctx, w_ctx->gpu_output_buffer[i]+r_first, /* r */
				w_ctx->gpu_output_buffer[i+1]+s_processed); /* s */
		w_ctx->gpu_output_buffer[i]   = 0;
		w_ctx->gpu_output_buffer[i+1] = 0;
	}
		
	/* Update processed tuples index */
	*(w_ctx->s_processed) += w_ctx->s_batch_size;
}

/* Process TUPLES_PER_CHUNK_R Tuples on the gpu with nested loop join
 * atomics version without cpu interprete step
 */
void process_r_gpu_atomics (worker_ctx_t *w_ctx){
    	const unsigned r_processed = *(w_ctx->r_processed);
    	const unsigned s_first = w_ctx->s_first;
    	const unsigned s_processed = *(w_ctx->s_processed);
	
	if (s_processed - s_first > 0){
		if (w_ctx->range_predicate){
			compare_kernel_new_r_atomics_range<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer,
			       	w_ctx->gpu_output_buffer_size/8,
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				w_ctx->r_batch_size);
		} else {
			compare_kernel_new_r_atomics<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>
				(w_ctx->gpu_output_buffer,
			       	w_ctx->gpu_output_buffer_size/8,
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				w_ctx->r_batch_size);

		}

		CUDA_SAFE(cudaDeviceSynchronize());
	}

	for (int i = 0; i < w_ctx->gpu_output_buffer_size/8; i = i + 2){
		if (w_ctx->gpu_output_buffer[i] == 0)
			break;
		emit_result(w_ctx, w_ctx->gpu_output_buffer[i]+r_processed, /* r */
				w_ctx->gpu_output_buffer[i+1]+s_first); /* s */
		w_ctx->gpu_output_buffer[i]   = 0;
		w_ctx->gpu_output_buffer[i+1] = 0;
	}
		
	/* Update processed tuples index */
	*(w_ctx->r_processed) += w_ctx->r_batch_size;

}
void interprete_s(worker_ctx_t *w_ctx, int *bitmap) {
	const int r_processed = *(w_ctx->r_processed);
	const int s_processed = *(w_ctx->s_processed);
	const int r_first = w_ctx->r_first;
	const int i = w_ctx->s_batch_size * (r_processed - r_first) / 32;
        for (int z = 0; z < i; z++) {
		/* Is there even a result in this int */
                if (bitmap[z] == 0) {
                        continue;
                } else {
#pragma unroll
                        for (int k = 0; k < 32; k++){
				/* Check inside of int */
                                if (std::bitset<32>(bitmap[z]).test(k)) { 
					const int s = (z/((r_processed -  r_first)/32));
					const int r = (z - s*((r_processed - r_first)/32))*32+k;
			 		emit_result(w_ctx, r+r_first, s+s_processed);
                                }
                        }
                        bitmap[z] = 0;
                }
    	}
}

void interprete_r(worker_ctx_t *w_ctx, int *bitmap) {
	const int s_processed = *(w_ctx->s_processed);
	const int r_processed = *(w_ctx->r_processed);
	const int s_first = w_ctx->s_first;
	const int i = w_ctx->r_batch_size * ((s_processed - s_first) / 32);
        for (int z = 0; z < i; z++) {
		/* Is there even a result in this int */
                if (bitmap[z] == 0) {
                        continue;
                } else {
#pragma unroll
                        for (int k = 0; k < 32; k++){
				/* Check inside of int */
                                if (std::bitset<32>(bitmap[z]).test(k)) { 
					const int r = (z/((s_processed - s_first)/32));
					const int s = (z - r*((s_processed - s_first)/32))*32+k;
					emit_result(w_ctx, r+r_processed, s+s_first);
                                }
                        }
                        bitmap[z] = 0;
                }
    	}
}

/* Process batch of tuples on the cpu with nested loop join */
void process_s_cpu (worker_ctx_t *w_ctx, unsigned threads){

#pragma omp parallel for
	for (unsigned int r = w_ctx->r_first; r < *(w_ctx->r_processed); r++)
	{
		for (unsigned int s = *(w_ctx->s_processed); s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
		    s++)
		{
			if (w_ctx->range_predicate){
				const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
				const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
				//if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
				    //emit_result (w_ctx, r, s);
			} else {
				//if (w_ctx->S.a[s] + w_ctx->S.b[s] == w_ctx->R.x[r] + w_ctx->R.y[r])
				    //emit_result (w_ctx, r, s);
			}
		}
	}

	*(w_ctx->s_processed) += w_ctx->s_batch_size;
}

/* Process batch of tuples on the cpu with nested loop join */
void process_r_cpu (worker_ctx_t *w_ctx, unsigned threads){

#pragma omp parallel for
	for (unsigned int s = w_ctx->s_first; s < *(w_ctx->s_processed); s++)
        {
		for (unsigned int r = *(w_ctx->r_processed); 
				r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
		   		r++)
		{
			if (w_ctx->range_predicate){
				const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
				const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
				//if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
				    //emit_result (w_ctx, r, s);
			} else {
				//if (w_ctx->S.a[s] + w_ctx->S.b[s] == w_ctx->R.x[r] + w_ctx->R.y[r])
				    //emit_result (w_ctx, r, s);
			}
		}
	}
	
	*(w_ctx->r_processed) += w_ctx->r_batch_size;
}

/*
 *  Emit result is called every time a new tuple output tuple is produced
 */ 
void emit_result (worker_ctx_t *w_ctx, unsigned int r, unsigned int s) 
{   	
	Timer::Timer timer = Timer::Timer();

	/* Choose the older tuple to calc the latency /
	auto now = timer.now();
	if (s_get_tns(ctx,s) < r_get_tns(ctx,r)){
		auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>
				(now -  w_ctx->stats.start_time) - r_get_tns(ctx,r));
		w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>
			(w_ctx->stats.summed_latency + i);
	} else { 
		auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>
				(now -  w_ctx->stats.start_time) - s_get_tns(ctx,s));
		w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>
			(w_ctx->stats.summed_latency + i);
	}

	/* Output tuple statistics /
	w_ctx->stats.processed_output_tuples++;
	
	//int sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - w_ctx->stats.start_time ).count();
	//w_ctx->stats.output_tuple_map[sec]++;

	// Write into resultfile
	fprintf (w_ctx->resultfile, "%d %d\n", r,s);
*/
}

void
expire_outdated_tuples_ (worker_ctx_t *w_ctx, master_ctx_t *ctx){
   	const int s_processed = *(w_ctx->s_processed);
   	const int r_processed = *(w_ctx->r_processed);

	while ((r_get_tns(ctx,w_ctx->r_first).count() + w_ctx->window_size_R*1000000000L) 
			< s_get_tns(ctx,s_processed).count()){
		 w_ctx->r_first++;
	}

	while ((s_get_tns(ctx,w_ctx->s_first).count() + w_ctx->window_size_S*1000000000L) 
			< r_get_tns(ctx,r_processed).count()){
		 w_ctx->s_first++;
	}
}

void
expire_outdated_tuples (worker_ctx_t *w_ctx, master_ctx_t *ctx){
	if (w_ctx->processing_mode == cpu1_mode
        	|| w_ctx->processing_mode == cpu2_mode
         	|| w_ctx->processing_mode == cpu3_mode 
       		|| w_ctx->processing_mode == cpu4_mode
       		|| w_ctx->processing_mode == gpu_mode
       		|| w_ctx->processing_mode == atomic_mode) {
		expire_outdated_tuples_ (w_ctx, ctx);
	}
}
