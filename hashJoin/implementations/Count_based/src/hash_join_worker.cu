#include "config.h"
#include "parameter.h"

#include "assert.h"
#include <iostream>
#include <chrono>
#include <thread>

#include "data.h"
#include "master.h"
#include "ringbuffer.h"
#include "messages.h"
#include "hash_join_worker.h"

/* --- forward declarations --- */
static inline void emit_result (worker_ctx_t *ctx, unsigned int r,
                                unsigned int s);
static inline void flush_result (worker_ctx_t *ctx);	
void process_s (worker_ctx_t *w_ctx);
void process_r (worker_ctx_t *w_ctx);
void process_s_cpu (worker_ctx_t *w_ctx);
void process_r_cpu (worker_ctx_t *w_ctx);
//void expire_outdated_tuples (worker_ctx_t *w_ctx);

/*
 * woker
 */
void *start_worker(void *ctx){
	worker_ctx_t *w_ctx = (worker_ctx_t *) ctx;
	ringbuffer_t *r = w_ctx->data_R_queue;
	ringbuffer_t *s = w_ctx->data_S_queue;	

	 while (true)
   	 {
		/* process a message from the left queue if there's one waiting,
		 * but only if we are not blocked on the other side */
		/* if (!empty (ctx->left_recv_queue) && !full (ctx->right_send_queue))*/
		if (!empty_ (r))
		    process_r (w_ctx);

		/* likewise, handle messages from the right queue,
		 * but only if we are not blocked on the other side */
		/*if (!empty (ctx->right_recv_queue) && !full (ctx->left_send_queue))*/
		if (!empty_ (s))
		    process_s (w_ctx);

		/* check for tuple expiration */
		//expire_outdated_tuples (w_ctx);

		/* sleep for */
		//std::chrono::seconds sec(w_ctx->sleep_time);
	}
	return;
}

void process_s (worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == cpu_mode){
		process_s_cpu(w_ctx);
	}
}

void process_r (worker_ctx_t *w_ctx){
	if (w_ctx->processing_mode == cpu_mode){
		process_r_cpu(w_ctx);
	}
}

void process_s_cpu (worker_ctx_t *w_ctx){
	core2core_msg_t msg;
	receive(w_ctx->data_S_queue, &msg);
	for (unsigned int r = w_ctx->r_first; r < w_ctx->r_end; r++)
	{
		for (unsigned int s = msg.msg.chunk_S.start_idx;
		    s < msg.msg.chunk_S.start_idx +  TUPLES_PER_CHUNK_S;
		    s++)
		{
			const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
			const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
			if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
			    emit_result (w_ctx, r, s);
		}
	}
	w_ctx->s_end += TUPLES_PER_CHUNK_S;
}

void process_r_cpu (worker_ctx_t *w_ctx){
	core2core_msg_t msg;
	receive(w_ctx->data_R_queue, &msg);
	for (unsigned int s = w_ctx->s_first; s < w_ctx->s_end; s++)
        {
		for (unsigned int r = msg.msg.chunk_R.start_idx;
		    r < msg.msg.chunk_R.start_idx + TUPLES_PER_CHUNK_R;
		    r++)
		{
			const a_t a = w_ctx->S.a[s] - w_ctx->R.x[r];
			const b_t b = w_ctx->S.b[s] - w_ctx->R.y[r];
			if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
			    emit_result (w_ctx, r, s);
		}
	}
	w_ctx->r_end += TUPLES_PER_CHUNK_R;
	
}

static inline void
emit_result (worker_ctx_t *ctx, unsigned int r, unsigned int s)
{
    assert (ctx->partial_result_msg.pos < RESULTS_PER_MESSAGE);

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
