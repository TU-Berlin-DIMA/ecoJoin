#include "data.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

#include "master.h"
#include "cuda_helper.h"

/* Limit for generated data, Tuples above the limit are not generated but simulated by
 * reloading again existing tuples */
//const size_t datasize_limit_mb = 16;
const size_t datasize_limit_mb = 256;

const unsigned long tuple_limit = datasize_limit_mb * 1024 * 1024 /*B*/ / 16;

/* Calculate current nano-seconds */
std::chrono::nanoseconds get_current_ns(master_ctx_t *ctx, std::chrono::nanoseconds t){
	unsigned long range = 1.e9 / ctx->rate_R;
	//unsigned long range = 1.e8 / ctx->rate_R;
	std::chrono::nanoseconds z(range);//(random() % range);
	return t + z;
}

/**
 * Generate (materialize) an input data stream; our main performance
 * experiment is to feed such a materialized stream into a handshake
 * join setting.
 *
 * @param ctx a master context; the #R and #S fields in this context
 *            will be populated
 */
void generate_data (master_ctx_t *ctx)
{
    std::cout << "# Tuple generation limit " << tuple_limit << " tuples / " << datasize_limit_mb << "MB\n";
    std::cout << "# Total datasize of stream is " << ctx->num_tuples_R*16/*B*/ / 1024 / 1024<< "MB\n";


    std::chrono::nanoseconds t(0);      /* "current time" in nano-seconds */
    unsigned long range;  /* interval between two tuples is 0..range nsec */
    
    if (tuple_limit > ctx->num_tuples_R) {
    	ctx->generate_tuples_R = ctx->num_tuples_R;
    } else {
    	ctx->generate_tuples_R = tuple_limit;
    	assert(ctx->generate_tuples_R % 2 == 0);
	std::cout << "# Generate data until tuple limit \n";
    }
    
    if (tuple_limit > ctx->num_tuples_S) {
    	ctx->generate_tuples_S = ctx->num_tuples_S;
    } else {
    	ctx->generate_tuples_S = tuple_limit;
    	assert(ctx->generate_tuples_S % 2 == 0);
	std::cout << "# Generate data until tuple limit \n";
    }

    /* allocate memory */
    if (ctx->processing_mode == cpu1_mode
		    || ctx->processing_mode == cpu2_mode
		    || ctx->processing_mode == cpu3_mode
		    || ctx->processing_mode == cpu4_mode
		    || ctx->processing_mode == ht_cpu1_mode
		    || ctx->processing_mode == ht_cpu2_mode
		    || ctx->processing_mode == ht_cpu3_mode
		    || ctx->processing_mode == ht_cpu4_mode){
	    ctx->R.t_ns = (std::chrono::nanoseconds*)malloc((ctx->generate_tuples_R + 1) * sizeof (*ctx->R.t_ns));
	    ctx->R.x = (x_t*)malloc((ctx->generate_tuples_R  + 1) * sizeof (*ctx->R.x));
	    ctx->R.y = (y_t*)malloc((ctx->generate_tuples_R  + 1) * sizeof (*ctx->R.y));

	    if (! (ctx->R.x && ctx->R.y ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }

	    ctx->S.t_ns = (std::chrono::nanoseconds*)malloc((ctx->generate_tuples_S + 1) * sizeof (*ctx->S.t_ns));
	    ctx->S.a = (a_t*)malloc((ctx->generate_tuples_S + 1) * sizeof (*ctx->S.a));
	    ctx->S.b = (b_t*)malloc((ctx->generate_tuples_S + 1) * sizeof (*ctx->S.b));

	    if (! (ctx->S.a && ctx->S.b ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }
    } else if (ctx->processing_mode == gpu_mode
		    || ctx->processing_mode == atomic_mode){
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->R.t_ns), (ctx->generate_tuples_R + 1) * sizeof (*ctx->R.t_ns),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->R.x), (ctx->generate_tuples_R + 1) * sizeof (*ctx->R.x),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->R.y), (ctx->generate_tuples_R + 1) * sizeof (*ctx->R.y),0));

	    if (! (ctx->R.x && ctx->R.y ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }

	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->S.t_ns), (ctx->generate_tuples_S + 1) * sizeof (*ctx->S.t_ns),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->S.a), (ctx->generate_tuples_S + 1) * sizeof (*ctx->S.a),0));
	    CUDA_SAFE(cudaHostAlloc((void**)&(ctx->S.b), (ctx->generate_tuples_S + 1) * sizeof (*ctx->S.b),0));

	    if (! (ctx->S.a && ctx->S.b ))
	    {
		fprintf (stderr, "memory allocation error\n");
		exit (EXIT_FAILURE);
	    }

    } else {
	fprintf (stderr, "processing mode not found\n");
	exit (EXIT_FAILURE);
    }

    if (!ctx->linear_data) { // random data
	    std::cout << "# Create Random Dataset\n";
	    /* generate data for R */
    	    t = std::chrono::nanoseconds(0);
	    for (unsigned int i = 0; i < ctx->generate_tuples_R; i++)
	    {
		t = get_current_ns(ctx,t);
		ctx->R.t_ns[i] = t;
		
		/*
		int j = random () % ctx->int_value_range;
		if (i % 2 == 0) {
			ctx->R.x[i] = j / 2;
			ctx->R.y[i] = j / 2;
		} else {
			ctx->R.x[i] = (j / 2) + 1;
			ctx->R.y[i] = j / 2;
		}
		*/

		
		ctx->R.x[i] = random () % ctx->int_value_range;
		ctx->R.y[i] = (float) (random () % ctx->float_value_range);
		
	    }

	    /* generate data for S */
    	    t = std::chrono::nanoseconds(0);
	    for (unsigned int i = 0; i < ctx->generate_tuples_S; i++)
	    {
		t = get_current_ns(ctx,t);
		ctx->S.t_ns[i] = t;
		/*
		int j = random () % ctx->int_value_range;
		if (j % 2 == 0) {
			ctx->S.a[i] = j / 2;
			ctx->S.b[i] = j / 2;
		} else {
			ctx->S.a[i] = (j / 2) + 1;
			ctx->S.b[i] = j / 2;
		}
		*/

		
		ctx->S.a[i] = random () % ctx->int_value_range;
		ctx->S.b[i] = (float) (random () % ctx->float_value_range);
	    }
    } else { // linear data
	    std::cout << "# Create Linear Dataset\n";
    	    /* generate data for R */
	    int x = 0;
	    int y = 0;
    	    t = std::chrono::nanoseconds(0);
	    for (unsigned int i = 0; i < ctx->generate_tuples_R; i++)
	    {
		t = get_current_ns(ctx,t);
		ctx->R.t_ns[i] = t;

		if (i % 2 == 0)
		   x++;
		else
		   y++;
		ctx->R.x[i] = x;
		ctx->R.y[i] = y;
	    }

	    int a = 0;
	    int b = 0;
	    /* generate data for S */
    	    t = std::chrono::nanoseconds(0);
	    for (unsigned int i = 0; i < ctx->generate_tuples_S; i++)
	    {
		t = get_current_ns(ctx,t);
		ctx->S.t_ns[i] = t;
		
		if (i % 2 == 0)
		   a++;
		else
		   b++;
		ctx->S.a[i] = a;
		ctx->S.b[i] = b;
	    }
    }
   
}

