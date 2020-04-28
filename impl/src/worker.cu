#include "config.h"
#include "parameter.h"

#include "assert.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <unistd.h>
#include <sys/time.h>
#include <bitset>
#include <omp.h>

#include "data.h"
#include "master.h"
#include "ringbuffer.h"
#include "messages.h"
#include "worker.h"
#include "kernels.h"
#include "cuda_helper.h"
#include "dvs.h"

/* --- forward declarations --- */
static inline void emit_result (worker_ctx_t *ctx, unsigned int r,
                                unsigned int s);
void init_worker (worker_ctx_t *w_ctx);
void process_s (worker_ctx_t *w_ctx);
void process_r (worker_ctx_t *w_ctx);
void process_s_cpu (worker_ctx_t *w_ctx, unsigned threads);
void process_r_cpu (worker_ctx_t *w_ctx, unsigned threads);
void process_s_gpu (worker_ctx_t *w_ctx);
void process_r_gpu (worker_ctx_t *w_ctx);
void process_s_gpu_atomics (worker_ctx_t *w_ctx);
void process_r_gpu_atomics (worker_ctx_t *w_ctx);
void interprete_s (worker_ctx_t *w_ctx, int *bitmap);
void interprete_r (worker_ctx_t *w_ctx, int *bitmap);
void expire_outdated_tuples (worker_ctx_t *w_ctx);

/*
 * woker
 */
void *start_worker(void *ctx){
	worker_ctx_t *w_ctx = (worker_ctx_t *) ctx;
	init_worker(w_ctx);

	time_t start = time(0);
	auto idle_start_time = std::chrono::system_clock::now();
	auto proc_start_time = std::chrono::system_clock::now();
	
	if(w_ctx->enable_freq_scaling)
        	set_freq(w_ctx->frequency_mode, w_ctx->max_cpu_freq, w_ctx->max_gpu_freq);
	
	while (true)
   	{
		/* Stats */	
		proc_start_time = std::chrono::system_clock::now();
		w_ctx->stats.runtime_proc += std::chrono::duration_cast
			<std::chrono::milliseconds>(proc_start_time - idle_start_time).count();
		idle_start_time = std::chrono::system_clock::now();

		/* Wait until main releases the lock and enough data arrived
		 * Using conditional variables we avoid busy waiting
		 */
		if ((*(w_ctx->r_available) < *(w_ctx->r_processed) + w_ctx->r_batch_size)
                           && (*(w_ctx->s_available) < *(w_ctx->s_processed) + w_ctx->s_batch_size)){

				
			if(w_ctx->enable_freq_scaling)
				set_freq(w_ctx->frequency_mode, w_ctx->min_cpu_freq, w_ctx->min_gpu_freq);
				
			// Waiting signal for master 
			w_ctx->stop_signal = true;
			
			std::unique_lock<std::mutex> lk(*(w_ctx->data_mutex));
			//w_ctx->data_cv->wait(lk, [&](){return (true);});
			w_ctx->data_cv->wait(lk, [&](){return (*(w_ctx->r_available) >= *(w_ctx->r_processed) + w_ctx->r_batch_size)
                	           || (*(w_ctx->s_available) >= *(w_ctx->s_processed) + w_ctx->s_batch_size);});

			w_ctx->stop_signal = false;
		
			if(w_ctx->enable_freq_scaling)
				set_freq(w_ctx->frequency_mode, w_ctx->max_cpu_freq, w_ctx->max_gpu_freq);
		
			w_ctx->stats.switches_to_proc++;
		}

		/* Stats */	
		idle_start_time = std::chrono::system_clock::now();
		w_ctx->stats.runtime_idle += std::chrono::duration_cast
			<std::chrono::milliseconds>(idle_start_time - proc_start_time).count();
		proc_start_time = std::chrono::system_clock::now();

		/* process TUPLES_PER_CHUNK_R if there are that many tuples available */
		if (*(w_ctx->r_available) >= *(w_ctx->r_processed) + w_ctx->r_batch_size) {
		    process_r (w_ctx);
		}

		
		/* process TUPLES_PER_CHUNK_S if there are that many tuples available */
		if (*(w_ctx->s_available) >= *(w_ctx->s_processed) + w_ctx->s_batch_size){
		    process_s (w_ctx);
		}

		expire_outdated_tuples (w_ctx);
	}
}

void process_s (worker_ctx_t *w_ctx){
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
	}
	w_ctx->stats.processed_input_tuples += w_ctx->s_batch_size;
}

void process_r (worker_ctx_t *w_ctx){
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
}

/* Process TUPLES_PER_CHUNK_S Tuples on the gpu with nested loop join
 * Similar to HELLS JOIN
 */
void process_s_gpu (worker_ctx_t *w_ctx){
    	const unsigned s_processed = *(w_ctx->s_processed);
    	const unsigned r_first = w_ctx->r_first;
    	const unsigned r_processed = *(w_ctx->r_processed);
    	
	//assert (s_processed - *(w_ctx->s_available)) >= TUPLES_PER_CHUNK_S);

	if (r_processed - r_first > 0){
		
		/* Start kernel */
		compare_kernel_new_s<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				w_ctx->s_batch_size,
				r_processed - r_first);

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
    	
	//assert (s_processed - *(w_ctx->s_available)) >= TUPLES_PER_CHUNK_S);

	if (s_processed - s_first > 0){
		
		/* Start kernel */
		compare_kernel_new_r<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				w_ctx->r_batch_size);

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
		compare_kernel_new_s_atomics<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(w_ctx->gpu_output_buffer,
			       	w_ctx->gpu_output_buffer_size/8,
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				w_ctx->s_batch_size,
				r_processed - r_first);

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
		compare_kernel_new_r_atomics<<<w_ctx->gpu_gridsize, w_ctx->gpu_blocksize>>>(w_ctx->gpu_output_buffer,
			       	w_ctx->gpu_output_buffer_size/8,
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				w_ctx->r_batch_size);

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

/* Process TUPLES_PER_CHUNK_S Tuples on the cpu with nested loop join*/
void process_s_cpu (worker_ctx_t *w_ctx, unsigned threads){
	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (unsigned int r = w_ctx->r_first; r < *(w_ctx->r_processed); r++)
	{
		for (unsigned int s = *(w_ctx->s_processed); s < *(w_ctx->s_processed) + w_ctx->s_batch_size;
		    s++)
		{
			const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
			const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
			if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.)){
			    emit_result (w_ctx, r, s);
			}
		}
	}

	*(w_ctx->s_processed) += w_ctx->s_batch_size;
}

/* Process TUPLES_PER_CHUNK_R Tuples on the cpu with nested loop join*/
void process_r_cpu (worker_ctx_t *w_ctx, unsigned threads){
	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (unsigned int s = w_ctx->s_first; s < *(w_ctx->s_processed); s++)
        {
		for (unsigned int r = *(w_ctx->r_processed); 
				r < *(w_ctx->r_processed) + w_ctx->r_batch_size;
		   		r++)
		{
			const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
			const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
			if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
			    emit_result (w_ctx, r, s);
		}
	}
	
	*(w_ctx->r_processed) += w_ctx->r_batch_size;
}

/*
 *  TODO: Daten nur in den Outbuffer schreiben
 *  Nicht vom compiler optimiert
 */

/*
 *  Emit result is called every time a new tuple output tuple is produced
 *  We update our statistics data
 */ 
static inline void
emit_result (worker_ctx_t *w_ctx, unsigned int r, unsigned int s) 
{   
	auto now = std::chrono::system_clock::now();
	
	/* Choose the older tuple to calc the latency*/
	if (w_ctx->S.t_ns[s] < w_ctx->R.t_ns[r]){
		auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->R.t_ns[r]);
		w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
		//std::cout << "R: " << std::chrono::duration_cast <std::chrono::milliseconds>(i).count() << "\n";
	} else { 
		auto i = (std::chrono::duration_cast <std::chrono::nanoseconds>(now -  w_ctx->stats.start_time) - w_ctx->S.t_ns[s]);
		w_ctx->stats.summed_latency = std::chrono::duration_cast <std::chrono::nanoseconds>(w_ctx->stats.summed_latency + i);
		//std::cout << "S: " << std::chrono::duration_cast <std::chrono::milliseconds>(i).count() << "\n";
	}
	//std::cout << r <<  " " << s << " " << *(w_ctx->s_available) <<  " " <<  *(w_ctx->s_processed) << " " << *(w_ctx->r_available) <<  " " <<  *(w_ctx->r_processed) << "\n";

	/* Output tuple statistics */
	w_ctx->stats.processed_output_tuples++;
	//int sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - w_ctx->stats.start_time ).count();
	//w_ctx->stats.output_tuple_map[sec]++;
}


void
expire_outdated_tuples (worker_ctx_t *w_ctx){
   	const int s_processed = *(w_ctx->s_processed);
   	const int r_processed = *(w_ctx->r_processed);

	while ((w_ctx->R.t_ns[w_ctx->r_first].count() + w_ctx->window_size_R*1000000000L) < w_ctx->S.t_ns[s_processed].count()){
		 w_ctx->r_first++;
	}

	while ((w_ctx->S.t_ns[w_ctx->s_first].count() + w_ctx->window_size_S*1000000000L) < w_ctx->R.t_ns[r_processed].count()){
		 w_ctx->s_first++;
	}
}
