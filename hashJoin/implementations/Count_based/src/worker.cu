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

#include "data.h"
#include "master.h"
#include "ringbuffer.h"
#include "messages.h"
#include "worker.h"
#include "kernels.h"
#include "cuda_helper.h"

/* --- forward declarations --- */
static inline void emit_result (worker_ctx_t *ctx, unsigned int r,
                                unsigned int s);
void init_worker(worker_ctx_t *w_ctx);
void process_s (worker_ctx_t *w_ctx);
void process_r (worker_ctx_t *w_ctx);
void process_s_cpu (worker_ctx_t *w_ctx);
void process_r_cpu (worker_ctx_t *w_ctx);
void process_s_gpu (worker_ctx_t *w_ctx);
void process_r_gpu (worker_ctx_t *w_ctx);
void interprete_s (worker_ctx_t *w_ctx, int *bitmap);
void interprete_r (worker_ctx_t *w_ctx, int *bitmap);
//void expire_outdated_tuples (worker_ctx_t *w_ctx);

/*
 * woker
 */
void *start_worker(void *ctx){
	worker_ctx_t *w_ctx = (worker_ctx_t *) ctx;
	
	init_worker(w_ctx);

	time_t start = time(0);
	while (true)
   	{
		/* Wait until main releases the lock and enough data arrived
		 * Using conditional variables we avoid busy waiting
		 */
		if (MAIN_PROCESSING_LOCK){
			std::unique_lock<std::mutex> lk(*(w_ctx->data_mutex));
			w_ctx->data_cv->wait(lk, [&](){return (*(w_ctx->r_available) >= *(w_ctx->r_processed) + TUPLES_PER_CHUNK_R)
				   || (*(w_ctx->s_available) >= *(w_ctx->s_processed) + TUPLES_PER_CHUNK_S);});
		}
		
		/* process TUPLES_PER_CHUNK_R if there are that many tuples available */
		if (*(w_ctx->r_available) >= *(w_ctx->r_processed) + TUPLES_PER_CHUNK_R)
		    process_r (w_ctx);

		/* process TUPLES_PER_CHUNK_S if there are that many tuples available */
		if (*(w_ctx->s_available) >= *(w_ctx->s_processed) + TUPLES_PER_CHUNK_S)
		    process_s (w_ctx);

		/* check for tuple expiration TODO*/
		/* TODO: Auf GPU? */
		//expire_outdated_tuples (w_ctx);

		/* Check if we are still in the process time window */
		if (difftime( time(0), start) == w_ctx->process_window_time){
			start = time(0);

			/* Start idle time window */
			usleep(w_ctx->idle_window_time);
		}
	}
}

void process_s (worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == cpu_mode){
		process_s_cpu(w_ctx);
	} else if (w_ctx->processing_mode == gpu_mode){
		process_s_gpu(w_ctx);
	}
}

void process_r (worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == cpu_mode){
		process_r_cpu(w_ctx);
	} else if (w_ctx->processing_mode == gpu_mode){
		process_r_gpu(w_ctx);
	}
}

void init_worker (worker_ctx_t *w_ctx){

	/* Allocate output buffer */
	if (w_ctx->processing_mode == gpu_mode){
		
		unsigned buffer_size;

		// Estimated maximal buffersize
		if (TUPLES_PER_CHUNK_S > TUPLES_PER_CHUNK_R)
			buffer_size = w_ctx->num_tuples_R * TUPLES_PER_CHUNK_S;
		else
			buffer_size = w_ctx->num_tuples_S * TUPLES_PER_CHUNK_R;
		
		CUDA_SAFE(cudaHostAlloc((void**)&(w_ctx->gpu_output_buffer), buffer_size, 0));
		std::memset(w_ctx->gpu_output_buffer, 0, buffer_size);
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
		compare_kernel_new_s<<<1,GPU_THREAD_NUM>>>(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_processed]), &(w_ctx->S.b[s_processed]), 
				&(w_ctx->R.x[r_first]), &(w_ctx->R.y[r_first]), 
				TUPLES_PER_CHUNK_S, 
				r_processed - r_first);

		CUDA_SAFE(cudaDeviceSynchronize());

		interprete_s(w_ctx, w_ctx->gpu_output_buffer);
	}

	/* Update processed tuples index */
	*(w_ctx->s_processed) += TUPLES_PER_CHUNK_S;

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
		compare_kernel_new_r<<<1,GPU_THREAD_NUM>>>(w_ctx->gpu_output_buffer, 
				&(w_ctx->S.a[s_first]), &(w_ctx->S.b[s_first]), 
				&(w_ctx->R.x[r_processed]), &(w_ctx->R.y[r_processed]), 
				s_processed - s_first,
				TUPLES_PER_CHUNK_R);

		CUDA_SAFE(cudaDeviceSynchronize());

		interprete_r(w_ctx, w_ctx->gpu_output_buffer);
	}

	/* Update processed tuples index */
	*(w_ctx->r_processed) += TUPLES_PER_CHUNK_R;

}

void interprete_s(worker_ctx_t *w_ctx, int *bitmap) {
	const int r_processed = *(w_ctx->s_processed);
	const int s_processed = *(w_ctx->s_processed);
	const int i = TUPLES_PER_CHUNK_S * ((*(w_ctx->r_processed)) - w_ctx->r_first) / 32;
        for (int z = 0; z < i; z++) {
		/* Is there even a result in this int */
                if (bitmap[z] == 0) {
                        continue;
                } else {
#pragma unroll
                        for (int k = 0; k < 32; k++){
				/* Check inside of int */
                                if (std::bitset<32>(bitmap[z]).test(k)) { 
					const int s = (z/(r_processed/32));
					const int r = (z - s*(r_processed/32))*32+k;
					emit_result(w_ctx, r, s+s_processed);
                                }
                        }
                        bitmap[z] = 0;
                }
    	}
}

void interprete_r(worker_ctx_t *w_ctx, int *bitmap) {
	const int s_processed = *(w_ctx->s_processed);
	const int r_processed = *(w_ctx->r_processed);
	const int i = TUPLES_PER_CHUNK_R * ((*(w_ctx->s_processed)) - w_ctx->s_first) / 32;
        for (int z = 0; z < i; z++) {
		/* Is there even a result in this int */
                if (bitmap[z] == 0) {
                        continue;
                } else {
#pragma unroll
                        for (int k = 0; k < 32; k++){
				/* Check inside of int */
                                if (std::bitset<32>(bitmap[z]).test(k)) { 
					const int r = (z/(s_processed/32));
					const int s = (z - r*(s_processed/32))*32+k;
					emit_result(w_ctx, r+r_processed, s);
                                }
                        }
                        bitmap[z] = 0;
                }
    	}
}

/* Process TUPLES_PER_CHUNK_S Tuples on the cpu with nested loop join*/
void process_s_cpu (worker_ctx_t *w_ctx){

	for (unsigned int r = w_ctx->r_first; r < *(w_ctx->r_processed); r++)
	{
		for (unsigned int s = *(w_ctx->s_processed); s < *(w_ctx->s_processed) + TUPLES_PER_CHUNK_S;
		    s++)
		{
			const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
			const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
			if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
			    emit_result (w_ctx, r, s);
		}
	}

	*(w_ctx->s_processed) += TUPLES_PER_CHUNK_S;
}

/* Process TUPLES_PER_CHUNK_R Tuples on the cpu with nested loop join*/
void process_r_cpu (worker_ctx_t *w_ctx){
	
	for (unsigned int s = w_ctx->s_first; s < *(w_ctx->s_processed); s++)
        {
		for (unsigned int r = *(w_ctx->r_processed); 
				r < *(w_ctx->r_processed) + TUPLES_PER_CHUNK_R;
		   		r++)
		{
			const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
			const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
			if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
			    emit_result (w_ctx, r, s);
		}
	}
	
	*(w_ctx->r_processed) += TUPLES_PER_CHUNK_R;
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
	/* Update latency statistics */
	struct timeval t;
    	gettimeofday (&t, NULL);
	
	/* Choose the older tuple to calc the latency*/
	if (w_ctx->S.t[s].tv_sec*1000000000L + w_ctx->S.t[s].tv_nsec >
		w_ctx->R.t[r].tv_sec*1000000000L + w_ctx->R.t[r].tv_nsec){
		w_ctx->stats.summed_latency += ((t.tv_sec * 1000000 + t.tv_usec) - 
				((w_ctx->S.t[s].tv_sec + w_ctx->stats.start_time.tv_sec) * 1000000 + w_ctx->S.t[s].tv_nsec / 1000));
	} else {
		w_ctx->stats.summed_latency += ((t.tv_sec * 1000000 + t.tv_usec) - 
				((w_ctx->R.t[r].tv_sec + w_ctx->stats.start_time.tv_sec) * 1000000 + w_ctx->R.t[r].tv_nsec / 1000));
	}

	w_ctx->stats.processed_output_tuples++;
}


/*
void
expire_outdated_tuples (worker_ctx_t *w_ctx){
    struct timespec t_rel;

    t_rel = w_ctx->S.t[s_first];
    while (w_ctx->R.t[r_end].tv_sec*1000000000L + w_ctx->R.t[r_end].tv_nsec
         + w_ctx->window_size_R * 1000000000L) < (t_rel.tv_sec * 1000000000L + t_rel.tv_nsec){
	 w_ctx->r_end -= 1;
    }

    t_rel = w_ctx->R.t[r_first];
    while (w_ctx->S.t[r_end].tv_sec*1000000000L + w_ctx->S.t[r_end].tv_nsec
         + w_ctx->window_size_S * 1000000000L) < (t_rel.tv_sec * 1000000000L + t_rel.tv_nsec){
	 w_ctx->s_end -= 1;
    }
}
*/
