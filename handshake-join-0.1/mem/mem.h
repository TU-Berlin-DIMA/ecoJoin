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
 * $Id: mem.h 588 2010-08-02 13:21:37Z jteubner $
 */

#include "config.h"
#include "parameters.h"

#include <stdlib.h>

#ifndef MEM_H
#define MEM_H

/**
 * Allocate a chunk of memory on node @a node.  If the system does
 * not support NUMA, the @a node argument will be ignored.  Also, if
 * no memory can be acquired (i.e., when malloc() or numa_alloc_onnode()
 * return @c NULL), the function will abort the program.
 *
 * @param size size of memory region to allocate (as in malloc() or
 *             numa_alloc_onnode())
 * @param node NUMA node where region should be allocated (argument
 *             ignored on non-NUMA machines)
 *
 * @return pointer to allocated memory
 */
void *alloc_onnode (size_t size, int node);

#endif  /* MEM_H */
