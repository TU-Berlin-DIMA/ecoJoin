#include "data.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "master.h"
#include "cuda_helper.h"

/**
 * Generate (materialize) an input data stream; our main performance
 * experiment is to feed such a materialized stream into a handshake
 * join setting.
 *
 * @param ctx a master context; the #R and #S fields in this context
 *            will be populated
 */
void
generate_data (master_ctx_t *ctx)
{
    unsigned long t;      /* "current time" in nano-seconds */
    unsigned long range;  /* interval between two tuples is 0..range nsec */

    /* sanity check */
    
    /*if ((ctx->num_tuples_R % TUPLES_PER_CHUNK_R != 0)
            || (ctx->num_tuples_S % TUPLES_PER_CHUNK_S != 0))
    {
        fprintf (stderr,
                "WARNING: tuple counts that are not a multiple of the chunk\n"
                "         size may cause trouble with the join driver.\n");
    }*/
    

    /* allocate memory */

    if (ctx->processing_mode == cpu1_mode
		    || ctx->processing_mode == cpu2_mode
		    || ctx->processing_mode == cpu3_mode
		    || ctx->processing_mode == cpu4_mode
		    || ctx->processing_mode == ht_cpu1_mode
		    || ctx->processing_mode == ht_cpu2_mode
		    || ctx->processing_mode == ht_cpu3_mode
		    || ctx->processing_mode == ht_cpu4_mode){
	    ctx->R.t_ns = (std::chrono::nanoseconds*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.t_ns));
	    ctx->R.x = (x_t*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.x));
	    ctx->R.y = (y_t*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.y));

	    if (! (ctx->R.x && ctx->R.y ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }

	    ctx->S.t_ns = (std::chrono::nanoseconds*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.t_ns));
	    ctx->S.a = (a_t*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.a));
	    ctx->S.b = (b_t*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.b));

	    if (! (ctx->S.a && ctx->S.b ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }
    } else if (ctx->processing_mode == gpu_mode
		    || ctx->processing_mode == atomic_mode){
	    unsigned *i;
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->R.t_ns), (ctx->num_tuples_R + 1) * sizeof (*ctx->R.t_ns),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->R.x), (ctx->num_tuples_R + 1) * sizeof (*ctx->R.x),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->R.y), (ctx->num_tuples_R + 1) * sizeof (*ctx->R.y),0));

	    if (! (ctx->R.x && ctx->R.y ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }

	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->S.t_ns), (ctx->num_tuples_S + 1) * sizeof (*ctx->S.t_ns),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->S.a), (ctx->num_tuples_S + 1) * sizeof (*ctx->S.a),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->S.b), (ctx->num_tuples_S + 1) * sizeof (*ctx->S.b),0));

	    if (! (ctx->S.a && ctx->S.b ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }

    } else {
	fprintf (stderr, "processing mode not found\n");
	exit (EXIT_FAILURE);
    }

    /* generate data for R */
    t = 0;
    range = 2.e9 / ctx->rate_R;
    for (unsigned int i = 0; i < ctx->num_tuples_R; i++)
    {
        t = t + (random () % range);

	auto ns = std::chrono::nanoseconds(t % 1000000000L);
	auto s  = std::chrono::seconds(t / 1000000000L);
	ctx->R.t_ns[i] = s + ns;

        ctx->R.x[i] = random () % ctx->int_value_range;
        ctx->R.y[i] = (float) (random () % ctx->float_value_range);
    }

    /* generate data for S */
    t = 0;
    range = 2.e9 / ctx->rate_S;
    for (unsigned int i = 0; i < ctx->num_tuples_S; i++)
    {
        t = t + (random () % range);

	auto ns = std::chrono::nanoseconds(t % 1000000000L);
	auto s  = std::chrono::seconds(t / 1000000000L);
	ctx->S.t_ns[i] = s + ns;

        ctx->S.a[i] = random () % ctx->int_value_range;
        ctx->S.b[i] = (float) (random () % ctx->float_value_range);
    }

   
}

