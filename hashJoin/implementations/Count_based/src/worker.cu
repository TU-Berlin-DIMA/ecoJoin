#include "config.h"
#include "parameter.h"

#include "assert.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <unistd.h>


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
static inline void flush_result (worker_ctx_t *ctx);	
void process_s (worker_ctx_t *w_ctx);
void process_r (worker_ctx_t *w_ctx);
void process_s_cpu (worker_ctx_t *w_ctx);
void process_r_cpu (worker_ctx_t *w_ctx);
void process_s_gpu (worker_ctx_t *w_ctx);
void process_r_gpu (worker_ctx_t *w_ctx);
//void expire_outdated_tuples (worker_ctx_t *w_ctx);

/*
 * woker
 */
void *start_worker(void *ctx){
	worker_ctx_t *w_ctx = (worker_ctx_t *) ctx;
	
	 time_t start = time(0);
	 while (true)
   	 {
		/* Wait until main releases the lock and enough data arrived
		 * Using conditional variables we avoid busy waiting
		 */
    		std::unique_lock<std::mutex> lk(*(w_ctx->data_mutex));
    		w_ctx->data_cv->wait(lk, [&](){return (*(w_ctx->r_available) >= *(w_ctx->r_processed) + TUPLES_PER_CHUNK_R)
				   || (*(w_ctx->s_available) >= *(w_ctx->s_processed) + TUPLES_PER_CHUNK_S);});
		
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
	return;
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

/* Process TUPLES_PER_CHUNK_S Tuples on the gpu with nested loop join*/
void process_s_gpu (worker_ctx_t *w_ctx){
	int  *output_buffer;

    	assert ((*(w_ctx->s_processed) - *(w_ctx->s_available)) >= TUPLES_PER_CHUNK_S);

	if ((*(w_ctx->r_processed) - w_ctx->r_first) > 0){
		/* Allocate Memory */
		// TODO: Worst case buffer Schlechteste Match rate
		CUDA_SAFE(cudaHostAlloc((void**)&output_buffer, (*(w_ctx->r_processed) - w_ctx->r_first) * TUPLES_PER_CHUNK_S * sizeof(int),0));
		std::memset(output_buffer, 0,  (*(w_ctx->r_processed) - w_ctx->r_first) * TUPLES_PER_CHUNK_S * sizeof(int));

		/* Start kernel */
		compare_kernel_new_s<<<1,GPU_THREAD_NUM>>>(output_buffer, 
				w_ctx->S.a, w_ctx->S.b, w_ctx->R.x, w_ctx->R.y, 
				*(w_ctx->s_processed), *(w_ctx->s_processed) + TUPLES_PER_CHUNK_S, 
				w_ctx->r_first, *(w_ctx->r_processed));
		CUDA_SAFE(cudaDeviceSynchronize());

		/* Emit result tuple */
		for(int r = 0; r < (*(w_ctx->r_processed) - w_ctx->r_first); r++){
			if (output_buffer[r] != 0)
				emit_result (w_ctx, r + w_ctx->r_first, 
						output_buffer[r] + w_ctx->s_first);
		}
		
		/* Free Memory */
		CUDA_SAFE(cudaFreeHost(output_buffer));
	}

	/* Update processed tuples */
	*(w_ctx->s_processed) += TUPLES_PER_CHUNK_S;

}

/* Process TUPLES_PER_CHUNK_R Tuples on the gpu with nested loop join
 */
void process_r_gpu (worker_ctx_t *w_ctx){
	int  *output_buffer;

    	assert ((*(w_ctx->r_processed) - *(w_ctx->r_available)) >= TUPLES_PER_CHUNK_R);

	if ((*(w_ctx->r_processed) - w_ctx->r_first) > 0){
		/* Allocate Memory */
		CUDA_SAFE(cudaHostAlloc((void**)&output_buffer, (*(w_ctx->r_processed) - w_ctx->r_first)*sizeof(int),0));
		std::memset(output_buffer, 0,  (*(w_ctx->r_processed) - w_ctx->r_first)*sizeof(int));

		/* Start kernel */
		compare_kernel_new_s<<<1,GPU_THREAD_NUM>>>(output_buffer, 
				w_ctx->S.a, w_ctx->S.b, w_ctx->R.x, w_ctx->R.y, 
				w_ctx->s_first, *(w_ctx->s_processed), 
				*(w_ctx->r_processed), *(w_ctx->r_processed) +  TUPLES_PER_CHUNK_R);
		CUDA_SAFE(cudaDeviceSynchronize());

		/* Emit result tuple */
		for(int r = 0; r < (*(w_ctx->r_processed) - w_ctx->r_first); r++){
			if (output_buffer[r] != 0)
				emit_result (w_ctx, r + w_ctx->r_first, 
						output_buffer[r] + w_ctx->s_first);
		}
		
		/* Free Memory */
		CUDA_SAFE(cudaFreeHost(output_buffer));
	}

	/* Update processed tuples */
	*(w_ctx->r_processed) += TUPLES_PER_CHUNK_R;

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

static inline void
emit_result (worker_ctx_t *ctx, unsigned int r, unsigned int s)
{   
    assert (!ctx->partial_result_msg.pos < RESULTS_PER_MESSAGE);
    ctx->partial_result_msg.msg[ctx->partial_result_msg.pos]
        = (result_t) { .r = r, .s = s };

    ctx->partial_result_msg.pos++;

    if (ctx->partial_result_msg.pos == RESULTS_PER_MESSAGE)
        flush_result (ctx);
}

static inline void
flush_result (worker_ctx_t *ctx)
{
    if (ctx->partial_result_msg.pos != 0)
    {
        if (! send (ctx->result_queue, &ctx->partial_result_msg.msg,
                    ctx->partial_result_msg.pos * sizeof (result_t)))
        {
            fprintf (stderr, "Cannot send result. FIFO full.\n");
            exit (EXIT_FAILURE);
        }

        ctx->partial_result_msg.pos = 0;
    }
    else
    {
        //LOG(ctx->log, "flushing requested, but nothing to flush");
    }
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
