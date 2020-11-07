/**
 * @file
 *
 * Analyze performance of the join kernel in isolation.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id$
 */

#include "config.h"
#include "parameters.h"

#include <stdio.h>
#include <stdlib.h>
//#include <xmmintrin.h>

#include "master/master.h"
#include "data/data.h"

#define SIMD 1

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

static int num_results = 0;
static long int compared = 0;

static inline void
emit_result (unsigned int r, unsigned int s)
{
    num_results++;

    if (num_results % 1000 == 0)
        fprintf (stderr, "%u results found.\n", num_results);
}

int
main (int argc, char **argv)
{
    master_ctx_t *ctx = malloc (sizeof (*ctx));

    ctx->num_tuples_R      = 25600;
    ctx->num_tuples_S      = 51200;
    ctx->rate_R            = 950;
    ctx->rate_S            = 950;
    ctx->window_size_R     = 600;
    ctx->window_size_S     = 600;
    ctx->int_value_range   = 10000;
    ctx->float_value_range = 10000;

    generate_data (ctx);

    for (unsigned int block = 0;
            block < ctx->num_tuples_R; block += TUPLES_PER_CHUNK_R)
    {

        /* This is the join loop, almost exactly as in worker.c */
        for (unsigned int s = 0; s < ctx->num_tuples_S; s++)
        {
/*#if SIMD
            const v4si a_s = (v4si) _mm_set1_epi32 (ctx->S.a[s]);
            const v4sf b_s = _mm_set1_ps (ctx->S.b[s]);

            for (unsigned int r = block; r < block + TUPLES_PER_CHUNK_R; r += 4)
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
                            emit_result (r+i, s);
                }

                compared += 4;
            }
#else*/
            for (unsigned int r = block; r < block + TUPLES_PER_CHUNK_R; r++)
            {
                const a_t a = ctx->S.a[s] - ctx->R.x[r];
                const b_t b = ctx->S.b[s] - ctx->R.y[r];

                if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                    emit_result (r, s);

                compared++;
            }
///#endif
        }
    }

    fprintf (stderr, "Compared %lu tuples, %u results.\n", compared,
            num_results);
}
