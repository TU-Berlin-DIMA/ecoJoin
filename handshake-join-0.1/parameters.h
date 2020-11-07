/**
 * @file
 *
 * Consolidate various types of parameters here.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * $Id: parameters.h 609 2010-08-17 11:36:33Z jteubner $
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

/**
 * Chunk size in which we sent data between processing cores.
 * Our implementation does not send individual tuples around, but
 * batches them up into chunks.  This parameter specifies that
 * chunk size for data from stream R.
 */
#define TUPLES_PER_CHUNK_R 64

/**
 * Chunk size in which we sent data between processing cores.
 * Our implementation does not send individual tuples around, but
 * batches them up into chunks.  This parameter specifies that
 * chunk size for data from stream S.
 */
#define TUPLES_PER_CHUNK_S 64

/**
 * Amount of memory (measured in number of tuples) allocated per NUMA
 * region for relation R.
 */
#define ALLOCSIZE_PER_NUMA_R (2 << 28)

/**
 * Amount of memory (measured in number of tuples) allocated per NUMA
 * region for relation S.
 */
#define ALLOCSIZE_PER_NUMA_S (2 << 28)

/**
 * Size of each FIFO message queue (in number of messages).
 */
#define MESSAGE_QUEUE_LENGTH 64

/**
 * Time interval (in nano-seconds) with which main.c:collect_results()
 * will wake up and collect result data.
 *
 * If you set this parameter too high, result FIFOs might run full and
 * throughput might be constrained on the result end.  Setting it too
 * low will cause higher CPU load on the collection thread.
 *
 * Currently: 50ms
 */
#define COLLECT_INTERVAL (50 * 1000000L)

/**
 * A queue send request will be attempted this many times, until
 * we give up because the queue is full.
 */
#define SEND_TIMEOUT 100000

/**
 * Before dumping work on their neighbors, processing cores will
 * check how loaded they are.  They will compare the size of the
 * neighbor's window to the size of their own.  Only if it's smaller
 * within this factor, there will be data sent.
 *
 * This number is measured in tuples; typically it is a small
 * multiple of the chunk size.
 */
#define MAX_R_LOAD_DIFFERENCE (5 * TUPLES_PER_CHUNK_R)

/** @see #MAX_R_LOAD_DIFFERENCE */
#define MAX_S_LOAD_DIFFERENCE (5 * TUPLES_PER_CHUNK_S)

/**
 * If we sent out this many tuples/chunks to our neighbors without
 * having received any acknowledgement messages, we don't send over
 * any new data.
 *
 * This number is measured in tuples; typically it is a small
 * multiple of the chunk size.
 */
#define MAX_OUTSTANDING_R_ACKS (5 * TUPLES_PER_CHUNK_R)

/** @see #MAX_OUTSTANDING_S_ACKS */
#define MAX_OUTSTANDING_S_ACKS (5 * TUPLES_PER_CHUNK_S)

/**
 * For simulation only.  Time in micro-seconds that we need per
 * tuple/chunk comparison.
 */
#define JOIN_COST 1

#endif  /* PARAMETERS_H */
