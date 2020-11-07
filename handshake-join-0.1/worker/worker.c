/**
 * @file
 *
 * Handshake join worker; to be executed as a thread on each core.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 Systems Group, ETH Zurich, Switzerland
 *
 * $Id: worker.c 602 2010-08-16 14:03:49Z jteubner $
 */

#include "config.h"
#include "parameters.h"

#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

//#include <xmmintrin.h>

#include "debug/debug.h"
#include "worker.h"
#include "messages/core2core.h"
#include "messages/result.h"

/* --- forward declarations --- */

static inline void process_left (worker_ctx_t *ctx);
static inline void process_right (worker_ctx_t *ctx);
static inline void expire_outdated_tuples (worker_ctx_t *ctx);
static inline void send_S_ack (ringbuffer_t *queue,
                               uint32_t idx, uint32_t size);
static inline void send_R_ack (worker_ctx_t *ctx, ringbuffer_t *queue,
                               uint32_t idx, uint32_t size);
static inline void emit_result (worker_ctx_t *ctx, unsigned int r,
                                unsigned int s);
static inline void flush_result (worker_ctx_t *ctx);

/**
 * Vector of four single-precision integers (128-bit SIMD registers
 * or 16-byte SIMD registers).
 */
typedef a_t v4si __attribute__((vector_size (16)));
typedef b_t v4sf __attribute__((vector_size (16)));

union a_vec {
    a_t   val[4];
    v4si  vec;
};
typedef union a_vec a_vec;

union b_vec {
    b_t   val[4];
    v4sf  vec;
};
typedef union b_vec b_vec;

union match_vec {
    int   val[4];
    v4si  vec;
};
typedef union match_vec match_vec;


/*static const a_vec a_band_min = { .val = { -10, -10, -10, -10 } };
static const a_vec a_band_max = { .val = { 10, 10, 10, 10 } };
static const b_vec b_band_min = { .val = { -10., -10., -10., -10. } };
static const b_vec b_band_max = { .val = { 10., 10., 10., 10. } };*/

/**
 * Worker loop; checks message boxes and invokes process_left(),
 * process_right(), and expire_outdated_tuples() accordingly.
 * This is essentially a 1:1 implementation of the algorithm shown
 * in the paper.
 *
 * @param ctx worker context
 */
void *
handshake_join (void *arg)
{
    worker_ctx_t     *ctx = (worker_ctx_t *) arg;
    void             *msgptr;

#ifndef NDEBUG
    char          logfilename[] = "log-core-000";

    /* open a log file */
    snprintf (logfilename, sizeof (logfilename), "log-core-%03u", ctx->id);
    ctx->log = fopen (logfilename, "w");
    assert (ctx->log);
#endif

    LOG(ctx->log, "Processing core %u starting up.", ctx->id);

#ifdef HAVE_LIBNUMA
    if (ctx->numa_node >= 0)
    {
        LOG(ctx->log, "Forcing core %u to run on NUMA node %i.\n",
                ctx->id, ctx->numa_node);

        if (numa_run_on_node (ctx->numa_node) < 0)
            LOG(ctx->log, "NUMA migration to node %i failed.\n",
                    ctx->numa_node);
    }
#endif

    while (true)
    {
        /* process a message from the left queue if there's one waiting,
         * but only if we are not blocked on the other side */
        /* if (!empty (ctx->left_recv_queue) && !full (ctx->right_send_queue))*/
        if (!empty (ctx->left_recv_queue)
                && (!full (ctx->right_send_queue)
                    || (peek (ctx->left_recv_queue, &msgptr)
                        && ((core2core_msg_t *) msgptr)->type == ack_S_msg)))
            process_left (ctx);

        /* likewise, handle messages from the right queue,
         * but only if we are not blocked on the other side */
        /*if (!empty (ctx->right_recv_queue) && !full (ctx->left_send_queue))*/
        if (!empty (ctx->right_recv_queue)
                && (!full (ctx->left_send_queue)
                    || (peek (ctx->right_recv_queue, &msgptr)
                        && ((core2core_msg_t *) msgptr)->type == ack_R_msg)))
            process_right (ctx);

        /* check for tuple expiration */
        expire_outdated_tuples (ctx);
    }

    return NULL;
}

/**
 * Process a message received from the left receive channel.  This
 * will be either a chunk of new data (beloning to input stream R)
 * or an acknowledgement for a chunk of data that we previously sent
 * to the left (for stream S).
 *
 * The left end of the pipeline is a tiny bit special.  Here,
 * acknowledgement messages indicate that a chunk of data should be
 * removed from the current join window.  Thus, such acknowledgements
 * may arrive even though we never sent out the respective data
 * chunk.  Thus, if we receive such acknowlegements, we leave them
 * in the channel until we have sent out the actual data item.
 *
 * @param ctx worker context
 *
 * @see paper submission
 */
static inline void
process_left (worker_ctx_t *ctx)
{
    void            *peek_msg;
    core2core_msg_t  msg;

    /* peek at the message to decide what type of message it is */
    peek (ctx->left_recv_queue, &peek_msg);

    if (((core2core_msg_t *) peek_msg)->type == new_R_msg)
    {
        receive (ctx->left_recv_queue, &msg);

        /* code has not been tested for arbitrary size values;
         * make sure there are no surprises */
        assert (msg.msg.chunk_R.size == TUPLES_PER_CHUNK_R);

        /* also, we assume that we receive data in order */
        assert (msg.msg.chunk_R.start_idx == ctx->wnd_R_end);

        LOG(ctx->log, "new R data [%u:%u] received from left.",
                msg.msg.chunk_R.start_idx, msg.msg.chunk_R.size);

        /* copy data if needed */
        if (ctx->copy_R)
        {
            memcpy (ctx->R.x
                        + (msg.msg.chunk_R.start_idx % ALLOCSIZE_PER_NUMA_R),
                    ctx->left_ctx->R.x
                        + (msg.msg.chunk_R.start_idx % ALLOCSIZE_PER_NUMA_R),
                    msg.msg.chunk_R.size * sizeof (*ctx->R.x));

            memcpy (ctx->R.y
                        + (msg.msg.chunk_R.start_idx % ALLOCSIZE_PER_NUMA_R),
                    ctx->left_ctx->R.y
                        + (msg.msg.chunk_R.start_idx % ALLOCSIZE_PER_NUMA_R),
                    msg.msg.chunk_R.size * sizeof (*ctx->R.y));
        }

        /*
         * SCAN / EVALUATE JOIN
         *
         * We use S here as the outer join relation.  The scanned piece
         * of R is very small and will thus be fully in L1.  Effectively,
         * we pay the price of scanning S once.
         *
         * NOTE: This is probably where we spent most of our CPU work
         *       (and memory bandwidth).  This may be a good starting
         *       point for low-level optimization.
         */
        for (unsigned int s = ctx->wnd_S_start; s < ctx->wnd_S_end; s++)
        {
/*#if ENABLE_SIMD

           /
             * These are loop invariant.  Keep them out of the following
             * loop.  (Compiler won't recognize this opportunity on its
             * own.)
             *
            const v4si a_s = (v4si) _mm_set1_epi32 (ctx->S.a[s]);
            const v4sf b_s = _mm_set1_ps (ctx->S.b[s]);

            for (unsigned int r = msg.msg.chunk_R.start_idx;
                    r < msg.msg.chunk_R.start_idx + msg.msg.chunk_R.size;
                    r+=4)
            {
                const v4si a_r = (v4si)
                    _mm_load_si128 ( (__m128i *) (ctx->R.x + r));

                const v4sf b_r = _mm_load_ps (ctx->R.y + r);

                const v4si a_diff = a_s - a_r;
                const v4sf b_diff = b_s - b_r;

                const v4si match1 = (v4si)
                    _mm_cmpgt_epi32 ((__m128i) a_diff, (__m128i) a_band_min.vec);
                const v4si match2 = (v4si)
                    _mm_cmplt_epi32 ((__m128i) a_diff, (__m128i) a_band_max.vec);
                const v4si match3 = (v4si)
                    _mm_cmpgt_ps (b_diff, b_band_min.vec);
                const v4si match4 = (v4si)
                    _mm_cmplt_ps (b_diff, b_band_max.vec);

                const match_vec match =
                    { .vec = match1 & match2 & match3 & match4 };

                const int short_match =
                    _mm_movemask_ps ((__m128) match.vec);

                if (short_match)
                {
                    for (unsigned int i = 0; i < 4; i++)
                        if (match.val[i])
                            emit_result (ctx, r+i, s);
                }
            }

#else*/
            for (unsigned int r = msg.msg.chunk_R.start_idx;
                    r < msg.msg.chunk_R.start_idx + msg.msg.chunk_R.size;
                    r++)
            {
                /*const a_t a = ctx->S.a[s] - ctx->R.x[r];
                const b_t b = ctx->S.b[s] - ctx->R.y[r];
                if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                    emit_result (ctx, r, s);*/

		// EESJ Prodicate
                if (ctx->R.x[r] == ctx->S.a[s])
                    emit_result (ctx, r, s);
            }
//#endif
        }


        ctx->wnd_R_end += msg.msg.chunk_R.size;

        LOG(ctx->log, "sending acknowledgement for R [%u:%u] to left.",
                msg.msg.chunk_R.start_idx, msg.msg.chunk_R.size);

        send_R_ack (ctx, ctx->left_send_queue, msg.msg.chunk_R.start_idx,
                msg.msg.chunk_R.size);
    }
    else if (((core2core_msg_t *) peek_msg)->type == ack_S_msg)
    {
        assert (((core2core_msg_t *) peek_msg)->type == ack_S_msg);

        /*
         * Only take out the acknowledgement if we actually sent
         * something out.
         *
         * This is relevant at the ends of the pipeline.  There, the
         * driver uses acknowledgement messages to indicate that a
         * tuple should be taken out of the system.  But it may happen
         * that we did not yet receive the the corresponding tuple.
         * If so, we simply wait until we sent out some more data
         * to the pipeline end.
         */
        if (ctx->wnd_S_start != ctx->wnd_S_sent)
        {
            receive (ctx->left_recv_queue, &msg);

            /* code has not been tested for arbitrary size values;
             * make sure there are no surprises */
            assert (msg.msg.ack_S.size == TUPLES_PER_CHUNK_S);

            /* also, we assume that we receive data in order */
            assert (msg.msg.ack_S.start_idx == ctx->wnd_S_start);

            /* fprintf (stderr, "core %u received ack from left\n", ctx->id); */
            LOG(ctx->log,
                    "acknowledgement (for S [%u:%u]) received from left.",
                    msg.msg.ack_S.start_idx, msg.msg.ack_S.size);

            /* remove oldest chunk from S-window */

            /* mock-up */
            ctx->wnd_S_start += msg.msg.ack_S.size;
        }
    }
    else
    {
        unsigned int msg_size;

        assert (((core2core_msg_t *) peek_msg)->type == flush_msg);

        msg_size = receive (ctx->left_recv_queue, &msg);

        flush_result (ctx);

        /* forward message to right */
        send (ctx->right_send_queue, &msg, msg_size);
    }
}

/**
 * Process a message received from the right receive channel.  This
 * will be either a chunk of new data (beloning to input stream S)
 * or an acknowledgement for a chunk of data that we previously sent
 * to the right (for stream R).
 *
 * The right end of the pipeline is a tiny bit special.  Here,
 * acknowledgement messages indicate that a chunk of data should be
 * removed from the current join window.  Thus, such acknowledgements
 * may arrive even though we never sent out the respective data
 * chunk.  Thus, if we receive such acknowlegements, we leave them
 * in the channel until we have sent out the actual data item.
 *
 * @param ctx worker context
 *
 * @see paper submission
 */
static inline void
process_right (worker_ctx_t *ctx)
{
    void            *peek_msg;
    core2core_msg_t  msg;

    /* peek at the message to decide what type of message it is */
    peek (ctx->right_recv_queue, &peek_msg);

    /* fprintf (stderr, "core %u received data from right\n", ctx->id); */

    if (((core2core_msg_t *) peek_msg)->type == new_S_msg)
    {
        /* receive message from the channel */
        receive (ctx->right_recv_queue, &msg);

        /* code has not been tested for arbitrary size values;
         * make sure there are no surprises */
        assert (msg.msg.chunk_S.size == TUPLES_PER_CHUNK_S);

        /* also, we assume that we receive data in order */
        assert (msg.msg.chunk_S.start_idx == ctx->wnd_S_end);

        LOG(ctx->log, "new S data [%u:%u] received from right.",
                msg.msg.chunk_S.start_idx, msg.msg.chunk_S.size);

        /* copy data if needed */
        if (ctx->copy_S)
        {
            memcpy (ctx->S.a
                        + (msg.msg.chunk_S.start_idx % ALLOCSIZE_PER_NUMA_S),
                    ctx->right_ctx->S.a
                        + (msg.msg.chunk_S.start_idx % ALLOCSIZE_PER_NUMA_S),
                    msg.msg.chunk_S.size * sizeof (*ctx->S.a));

            memcpy (ctx->S.b
                        + (msg.msg.chunk_S.start_idx % ALLOCSIZE_PER_NUMA_S),
                    ctx->right_ctx->S.b
                        + (msg.msg.chunk_S.start_idx % ALLOCSIZE_PER_NUMA_S),
                    msg.msg.chunk_S.size * sizeof (*ctx->S.b));
        }

        /* SCAN / EVALUTATE JOIN */

        /*
         * In the middle of the pipeline, we don't consider tuples that
         * have been sent out to the left neighbor, but for which we
         * did not yet receive an acknowledgement (because join pairs
         * will already be found by our left neighbor).  At the end of
         * the pipeline, there is no left neighbor who could find the
         * pair for us, so we do have to consider such tuples here.
         */
        const unsigned int r_first
            = (ctx->id == ctx->num_workers - 1)
              ? ctx->wnd_R_start : ctx->wnd_R_sent;

        /*
         * R is the outer join relation; see "symmetric" counterpart
         * in process_left().
         */
        for (unsigned int r = r_first; r < ctx->wnd_R_end; r++)
        {
/*#if ENABLE_SIMD

            const v4si a_r = (v4si) _mm_set1_epi32 (ctx->R.x[r]);
            const v4sf b_r = _mm_set1_ps (ctx->R.y[r]);

            for (unsigned int s = msg.msg.chunk_S.start_idx;
                    s < msg.msg.chunk_S.start_idx + msg.msg.chunk_R.size;
                    s+=4)
            {
                const v4si a_s = (v4si)
                    _mm_load_si128 ( (__m128i *) (ctx->S.a + s));

                const v4sf b_s = _mm_load_ps (ctx->S.b + s);

                const v4si a_diff = a_s - a_r;
                const v4sf b_diff = b_s - b_r;

                const v4si match1 = (v4si)
                    _mm_cmpgt_epi32 ((__m128i) a_diff, (__m128i) a_band_min.vec);
                const v4si match2 = (v4si)
                    _mm_cmplt_epi32 ((__m128i) a_diff, (__m128i) a_band_max.vec);
                const v4si match3 = (v4si)
                    _mm_cmpgt_ps (b_diff, b_band_min.vec);
                const v4si match4 = (v4si)
                    _mm_cmplt_ps (b_diff, b_band_max.vec);

                const match_vec match =
                    { .vec = match1 & match2 & match3 & match4 };

                const int short_match =
                    _mm_movemask_ps ((__m128) match.vec);

                if (short_match)
                {
                    for (unsigned int i = 0; i < 4; i++)
                        if (match.val[i])
                            emit_result (ctx, r, s+i);
                }
            }
#else*/
            for (unsigned int s = msg.msg.chunk_S.start_idx;
                    s < msg.msg.chunk_S.start_idx + msg.msg.chunk_R.size;
                    s++)
            {
                /*const a_t a = ctx->S.a[s] - ctx->R.x[r];
                const b_t b = ctx->S.b[s] - ctx->R.y[r];
                if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                    emit_result (ctx, r, s);*/
		// EESJ Predicate
                if (ctx->R.x[r] == ctx->S.a[s])
                    emit_result (ctx, r, s);
            }
//#endif
        }


        ctx->wnd_S_end += msg.msg.chunk_S.size;

        LOG(ctx->log, "sending acknowledgement for S [%u:%u] to right.",
                msg.msg.chunk_S.start_idx, msg.msg.chunk_S.size);

        /* FIXME: idx to acknowledge? */
        send_S_ack (ctx->right_send_queue, msg.msg.chunk_S.start_idx,
                msg.msg.chunk_S.size);
    }
    else if (((core2core_msg_t *) peek_msg)->type == ack_R_msg)
    {
        /*
         * Only take out the acknowledgement if we actually sent
         * something out.
         *
         * This is relevant at the ends of the pipeline.  There, the
         * driver uses acknowledgement messages to indicate that a
         * tuple should be taken out of the system.  But it may happen
         * that we did not yet receive the the corresponding tuple.
         * If so, we simply wait until we sent out some more data
         * to the pipeline end.
         */
        if (ctx->wnd_R_start != ctx->wnd_R_sent)
        {
            /* receive message from the channel */
            receive (ctx->right_recv_queue, &msg);

            /* code has not been tested for arbitrary size values;
             * make sure there are no surprises */
            assert (msg.msg.ack_R.size == TUPLES_PER_CHUNK_R);

            /* also, we assume that we receive data in order */
            assert (msg.msg.ack_R.start_idx == ctx->wnd_R_start);

            /* fprintf (stderr, "core %u received ack from left\n", ctx->id); */
            LOG(ctx->log,
                    "acknowledgement (for R [%u:%u]) received from right.",
                    msg.msg.ack_R.start_idx, msg.msg.ack_R.size);

            /* remove oldest chunk from R-window */

            /* mock-up */
            ctx->wnd_R_start += msg.msg.ack_R.size;
        }
    }
    else
    {
        unsigned int msg_size;

        assert (((core2core_msg_t *) peek_msg)->type == flush_msg);

        msg_size = receive (ctx->right_recv_queue, &msg);

        flush_result (ctx);

        /* forward message to left */
        send (ctx->left_send_queue, &msg, msg_size);
    }


}

/**
 * Send an acknowledgement for a chunk of S data.
 *
 * @todo This function and send_R_ack() currently do not have a proper
 *       failure model (and they are not implemented symmetrically).
 *
 * @param queue queue where to put the message
 * @param idx   index to acknowledge
 */
static inline void
send_S_ack (ringbuffer_t *queue, uint32_t idx, uint32_t size)
{
    core2core_msg_t msg;

    msg.type                = ack_S_msg;
    msg.msg.ack_S.start_idx = idx;
    msg.msg.ack_S.size      = size;

    /* FIXME: What to do here if channel is full? */
    if (! send (queue, &msg, sizeof (msg)))
    {
        fprintf (stderr, "Sending S acknowledgement failed. FIFO full.\n");
        exit (EXIT_FAILURE);
    }
}

/**
 * Send an acknowledgement for a chunk of R data.
 *
 * @param queue queue where to put the message
 * @param idx   index to acknowledge
 */
static inline void
send_R_ack (worker_ctx_t *ctx, ringbuffer_t *queue, uint32_t idx, uint32_t size)
{
    core2core_msg_t msg;

    msg.type                = ack_R_msg;
    msg.msg.ack_R.start_idx = idx;
    msg.msg.ack_R.size      = size;

    /* FIXME: What to do here if channel is full? */
    if (! send (queue, &msg, sizeof (msg)))
    {
        fprintf (stderr, "Could not send ACK for R (core %u). FIFO full.\n",
                ctx->id);
        exit (EXIT_FAILURE);
    }
}

/**
 * "Send" a result tuple to the master thread.  In reality, tuples
 * are batched up and #RESULTS_PER_MESSAGE tuples are sent to the
 * master in a single message.
 *
 * For each result tuple, only the row numbers of the R/S tuples
 * are reported to the master.  The master keeps its own copy of
 * all data (workers only know the join arguments anyway) and uses
 * row numbers to construct result tuples.
 *
 * @note This function blindly sends out tuples with send(), but
 *       does not verify that sending actually succeeded.
 *
 * @param ctx worker context (a partial message with batched-up
 *            tuples is held in the context, plus the FIFO to the
 *            master thread
 * @param r tuple position within input stream R
 * @param s tuple position within input stream S
 */
static inline void
emit_result (worker_ctx_t *ctx, unsigned int r, unsigned int s)
{
    LOG(ctx->log, "result: r = %u, s = %u", r, s);

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
        LOG(ctx->log, "flushing result buffer (%u tuples)",
                ctx->partial_result_msg.pos);

        if (! send (ctx->result_queue, &ctx->partial_result_msg.msg,
                    ctx->partial_result_msg.pos * sizeof (result_t)))
        {
            fprintf (stderr, "Cannot send result (core %u). FIFO full.\n",
                    ctx->id);
            exit (EXIT_FAILURE);
        }

        ctx->partial_result_msg.pos = 0;
    }
    else
    {
        LOG(ctx->log, "flushing requested, but nothing to flush");
    }
}

/**
 * See if any data has expired and, if so, try to send them to our
 * neighbors.  For proper load balancing, we don't just dump
 * arbitrary amounts of work our our neighbors.  Rather, we only
 * send data if
 *
 *  - the respective FIFO queue is not full (this avoids blocking
 *    or even deadlocks - hopefully),
 *    .
 *  - there are not too many acknowledgements outstanding for data
 *    that we already sent over the same channel (this avoids
 *    convoy effects; if we find, because of the following rule, that
 *    we can send data over to our neighbor, that workload will not
 *    immediately be visible there; thus, if we enter this method
 *    the next time, we will still see that we can send over some
 *    data—effectively, we'll send out large convoys at once before
 *    we see anything happening on the other end), and
 *    .
 *  - the window currently held by our neighbor is not significantly
 *    larger than our own (this leads to a balancing of work).
 *
 * In this particular implementation, all data is immediately "expired,"
 * i.e., available to be sent out to our neighbor.  The above three
 * rules make sure we keep some data here and lead to a balancing of
 * work.
 *
 * @param ctx worker context
 */
static inline void
expire_outdated_tuples (worker_ctx_t *ctx)
{

    /* is there something that we could send? */
    if (ctx->wnd_S_sent != ctx->wnd_S_end)
    {
        /*
         * Only dump work on our neighbors if they aren't more loaded
         * than we are (modulo a fudge factor MAX_S_LOAD_DIFFERENCE).
         * We also make sure we don't send out too much without having
         * received any answer.
         */
        if (! full (ctx->left_send_queue)
                && (ctx->wnd_S_sent - ctx->wnd_S_start < MAX_OUTSTANDING_S_ACKS)
                && (ctx->left_ctx->wnd_S_end - ctx->left_ctx->wnd_S_start)
                    < (ctx->wnd_S_end - ctx->wnd_S_start
                       + MAX_S_LOAD_DIFFERENCE))
        {
            core2core_msg_t msg;

            LOG(ctx->log, "sending %u S tuples to left (starting at "
                    "index %u).", TUPLES_PER_CHUNK_S, ctx->wnd_S_sent);

#ifndef NDEBUG
            /* debugging */
            ctx->did_not_send_to_left = false;
#endif

            msg.type                  = new_S_msg;
            msg.msg.chunk_S.start_idx = ctx->wnd_S_sent;
            msg.msg.chunk_S.size      = TUPLES_PER_CHUNK_S;

            /* send cannot fail, because we checked before */
            send (ctx->left_send_queue, &msg, sizeof (msg));

            /* mark tuples as forwarded */
            ctx->wnd_S_sent += msg.msg.chunk_S.size;
        }
        else
        {
#ifndef NDEBUG
            if (! ctx->did_not_send_to_left)
                LOG(ctx->log,
                        "did not send data to left (full: %s; loaded: %s)",
                        full (ctx->left_send_queue) ? "yes" : "no",
                        (ctx->left_ctx->wnd_S_end - ctx->left_ctx->wnd_S_start)
                          < (ctx->wnd_S_end - ctx->wnd_S_start
                              + MAX_S_LOAD_DIFFERENCE) ? "no" : "yes");

            ctx->did_not_send_to_left = true;
#endif
        }
    }

    /* is there something that we could send? */
    if (ctx->wnd_R_sent != ctx->wnd_R_end)
    {
        /*
         * Only dump work on our neighbors if they aren't more loaded
         * than we are (modulo a fudge factor MAX_R_LOAD_DIFFERENCE).
         * We also make sure we don't send out too much without having
         * received any answer.
         */
        if (! full (ctx->right_send_queue)
                && (ctx->wnd_R_sent - ctx->wnd_R_start < MAX_OUTSTANDING_R_ACKS)
                && (ctx->right_ctx->wnd_R_end - ctx->right_ctx->wnd_R_start)
                    < (ctx->wnd_R_end - ctx->wnd_R_start
                       + MAX_R_LOAD_DIFFERENCE))
        {
            core2core_msg_t msg;

            LOG(ctx->log, "sending %u R tuples to right (starting at "
                    "index %u).", TUPLES_PER_CHUNK_R, ctx->wnd_R_sent);

#ifndef NDEBUG
            /* debugging */
            ctx->did_not_send_to_right = false;
#endif

            msg.type                  = new_R_msg;
            msg.msg.chunk_R.start_idx = ctx->wnd_R_sent;
            msg.msg.chunk_R.size      = TUPLES_PER_CHUNK_R;

            /* send cannot fail, because we checked before */
            send (ctx->right_send_queue, &msg, sizeof (msg));

            /* mark tuple as forwarded */
            ctx->wnd_R_sent += msg.msg.chunk_R.size;
        }
        else
        {
#ifndef NDEBUG
            if (! ctx->did_not_send_to_right)
                LOG(ctx->log,
                        "did not send data to right (full: %s; loaded: %s)",
                        full (ctx->right_send_queue) ? "yes" : "no",
                        (ctx->right_ctx->wnd_R_end -ctx->right_ctx->wnd_R_start)
                          < (ctx->wnd_R_end - ctx->wnd_R_start
                              + MAX_R_LOAD_DIFFERENCE) ? "no" : "yes");

            ctx->did_not_send_to_right = true;
#endif
        }
    }
}
