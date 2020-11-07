/**
 * @file
 *
 * Test program to extract some NUMA information from the system.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id: numainfo.c 583 2010-08-02 06:52:52Z jteubner $
 */

#include "config.h"

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif  /* HAVE_LIBNUMA */

#include <stdlib.h>
#include <stdio.h>

int
main (void)
{

#ifndef HAVE_LIBNUMA
    fprintf (stderr, "You did not have libnuma available when you "
            "compiled this program.\n");
    exit (EXIT_FAILURE);
#else

    int max_node;         /* output of numa_max_node() */

    printf ("Trying to get some NUMA information out of your system.\n");

    /* See whether this system has NUMA support at all. */
    if (numa_available () < 0)
    {
        fprintf (stderr, "Sorry, no NUMA support available on your system.\n");
        exit (EXIT_FAILURE);
    }
    else
        printf ("NUMA is available.  Good.\n");

    /* How many nodes are there in the system? */
    max_node = numa_max_node ();
    printf ("%i NUMA nodes found on your system.\n", max_node + 1);

    /* Analyze memory configuration */
    printf ("\nMemory configuration:\n");

    for (int node = 0; node <= max_node; node++)
    {
        long node_size;
        long free;

        node_size = numa_node_size (node, &free);

        if (node_size < 0)
            fprintf (stderr,
                    "  !Problem figuring out memory size of node %i!\n", node);
        else
            printf ("  Node %i has %li MB of memory (%li MB free)\n",
                    node, node_size / (1024 * 1024), free / (1024 * 1024));
    }

    /* Analyze NUMA distances (as reported by kernel) */
    printf ("\nNUMA distances (as reported by kernel):\n");

    printf ("      ");
    for (int node2 = 0; node2 <= max_node; node2++)
        printf (" %3i", node2);
    printf ("\n");

    printf ("-----+");
    for (int node2 = 0; node2 <= max_node; node2++)
        printf ("----");
    printf ("\n");

    for (int node1 = 0; node1 <= max_node; node1++)
    {
        printf (" %3i |", node1);
        for (int node2 = 0; node2 <= max_node; node2++)
        {
            int distance = numa_distance (node1, node2);
            printf (" %3i", distance);
        }
        printf ("\n");
    }

#endif  /* HAVE_LIBNUMA */

}
