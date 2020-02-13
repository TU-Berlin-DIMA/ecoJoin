// Number of tuples processed per send Message for Stream R
//#define TUPLES_PER_CHUNK_R 64
// Number of tuples processed per send Message for Stream S
//#define TUPLES_PER_CHUNK_S 64

/* Thread number for GPU processing*/
//#define GPU_THREAD_NUM 128

/* Lock the worker until the master notifies*/
#define MAIN_PROCESSING_LOCK 0

/**
 * Size of each FIFO message queue (in number of messages).
 */
#define MESSAGE_QUEUE_LENGTH 64


/**
 * We use one byte for status information.  Hence, messages may be
 * at most (#CACHELINESIZE - 1) bytes long.
 */
#define MAX_MESSAGESIZE ((CACHELINESIZE) - 1)


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
