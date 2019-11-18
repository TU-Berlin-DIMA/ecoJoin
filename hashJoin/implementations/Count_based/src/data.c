/**
 * @file
 *
 * Generate input data for handshake join
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * $Id$
 */

#include "data.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "master.h"

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
    /*
    if ((ctx->num_tuples_R % TUPLES_PER_CHUNK_R != 0)
            || (ctx->num_tuples_S % TUPLES_PER_CHUNK_S != 0))
    {
        fprintf (stderr,
                "WARNING: tuple counts that are not a multiple of the chunk\n"
                "         size may cause trouble with the join driver.\n");
    }
    */

    /* allocate memory */

    /* FIXME: Consider NUMA here */
    ctx->R.t = (timespec*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.t));
    ctx->R.x = (x_t*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.x));
    ctx->R.y = (y_t*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.y));
    ctx->R.z = (z_t*)malloc((ctx->num_tuples_R + 1) * sizeof (*ctx->R.z));

    if (! (ctx->R.x && ctx->R.y && ctx->R.z))
    {
        fprintf (stderr, "memory allocation error\n");
        exit (EXIT_FAILURE);
    }

    /* FIXME: Consider NUMA here */
    ctx->S.t = (timespec*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.t));
    ctx->S.a = (a_t*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.a));
    ctx->S.b = (b_t*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.b));
    ctx->S.c = (c_t*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.c));
    ctx->S.d = (d_t*)malloc((ctx->num_tuples_S + 1) * sizeof (*ctx->S.d));

    if (! (ctx->S.a && ctx->S.b && ctx->S.c && ctx->S.d))
    {
        fprintf (stderr, "memory allocation error\n");
        exit (EXIT_FAILURE);
    }

    /* generate data for R */
    t = 0;
    range = 2.e9 / ctx->rate_R;
    for (unsigned int i = 0; i < ctx->num_tuples_R; i++)
    {
        t = t + (random () % range);

        ctx->R.t[i] = (struct timespec) { .tv_sec  = t / 1000000000L,
                                          .tv_nsec = t % 1000000000L };

        ctx->R.x[i] = random () % ctx->int_value_range;
        ctx->R.y[i] = (float) (random () % ctx->float_value_range);
        snprintf (ctx->R.z[i], sizeof (z_t), "%u", i);
    }

    /* add one dummy tuple with an "infinity" timestamp */
    ctx->R.t[ctx->num_tuples_R] = (struct timespec) { .tv_sec  = 100000,
                                                      .tv_nsec = 0 };

    /* dump data to a file if requested */
    /*
    if (ctx->data_prefix)
    {
        unsigned int   l = strlen (ctx->data_prefix);
        char          *s = malloc (l + 3);
        FILE          *f;

        assert (s);
        strncpy (s, ctx->data_prefix, l);
        s[l++] = '.'; s[l++] = 'R'; s[l++] = '\0';

        if (!(f = fopen (s, "w")))
        {
            fprintf (stderr, "error dumping R to file %s.\n", s);
            exit (EXIT_FAILURE);
        }

        fprintf (f, "#\n");
        fprintf (f, "# Handshake Join data dump; stream R\n");
        fprintf (f, "# $Id: main.c 598 2010-08-13 08:39:12Z jteubner $\n");
        fprintf (f, "#\n");
        fprintf (f, "#   timestamp    |     x     |      y     |           z\n");
        fprintf (f, "# ---------------+-----------+------------+----------------------\n");

        for (unsigned int i = 0; i < ctx->num_tuples_R; i++)
        {
            fprintf (f, "%6lu.%09lu | %9u | %10.2f | %20s\n",
                    ctx->R.t[i].tv_sec, ctx->R.t[i].tv_nsec,
                    ctx->R.x[i], ctx->R.y[i], ctx->R.z[i]);
        }

        fclose (f);
    }
    */

    /* generate data for S */
    t = 0;
    range = 2.e9 / ctx->rate_S;
    for (unsigned int i = 0; i < ctx->num_tuples_S; i++)
    {
        t = t + (random () % range);

        ctx->S.t[i] = (struct timespec) { .tv_sec  = t / 1000000000L,
                                          .tv_nsec = t % 1000000000L };

        ctx->S.a[i] = random () % ctx->int_value_range;
        ctx->S.b[i] = (float) (random () % ctx->float_value_range);
        ctx->S.c[i] = (double) 2 * i;
        ctx->S.d[i] = i % 2 == 0;
    }

    /* add one dummy tuple with an "infinity" timestamp */
    ctx->S.t[ctx->num_tuples_S] = (struct timespec) { .tv_sec  = 100000,
                                                      .tv_nsec = 0 };

    /* dump data to a file if requested */
    /*
    if (ctx->data_prefix)
    {
        unsigned int   l = strlen (ctx->data_prefix);
        char          *s = malloc (l + 3);
        FILE          *f;

        assert (s);
        strncpy (s, ctx->data_prefix, l);
        s[l++] = '.'; s[l++] = 'S'; s[l++] = '\0';

        if (!(f = fopen (s, "w")))
        {
            fprintf (stderr, "error dumping S to file %s.\n", s);
            exit (EXIT_FAILURE);
        }

        fprintf (f, "#\n");
        fprintf (f, "# Handshake Join data dump; stream S\n");
        fprintf (f, "# $Id: main.c 598 2010-08-13 08:39:12Z jteubner $\n");
        fprintf (f, "#\n");
        fprintf (f, "#   timestamp    |     a     |      b     |     c      |   d\n");
        fprintf (f, "# ---------------+-----------+------------+--------------------\n");

        for (unsigned int i = 0; i < ctx->num_tuples_S; i++)
        {
            fprintf (f, "%6lu.%09lu | %9u | %10.2f | %10.2f | %5s\n",
                    ctx->S.t[i].tv_sec, ctx->S.t[i].tv_nsec,
                    ctx->S.a[i], ctx->S.b[i], ctx->S.c[i],
                    ctx->S.d[i] ? "true" : "false");
        }

        fclose (f);
    }
    */
}

