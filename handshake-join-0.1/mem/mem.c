/**
 * @file
 *
 * Wrappers for memory management; this is to work with NUMA support
 * on machines that support it, without breaking support for other
 * machines.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id: mem.c 588 2010-08-02 13:21:37Z jteubner $
 */

#include "config.h"
#include "parameters.h"

#include <stdlib.h>

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

/* stderr, fprintf */
#include <stdio.h>

#include "mem.h"

void *
alloc_onnode (size_t size, int node __attribute__((unused)) )
{
    void *ret;

#ifdef HAVE_LIBNUMA
    ret = numa_alloc_onnode (size, node);
#else
    ret = malloc (size);
#endif

    if (ret == NULL)
    {
        fprintf (stderr, "Error during memory allocation.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}
