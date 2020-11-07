/**
 * @file
 *
 * `main' routine for the `Handshake Join' stream join prototype.
 *
 * @mainpage How Soccer Players Would Do Stream Joins
 *
 * @section compilation Compilation (or Even Installation)
 *
 * This is a prototype implementation to experiment with ``Handshake
 * Join,'' an algorithm proposed in our SIGMOD 2011 paper.  The
 * implementation includes the join operator itself, but also the
 * driver code to run the experimental setup described in the paper.
 *
 * The code has been written using standard GNU tools and uses autotools
 * for configuration.  Thus, compilation should be as simple as:
 *
 * @verbatim
     ./configure --disable-assert
     make
@endverbatim
 *
 * (You can run <tt>make install</tt>, too, if you really want to write
 * the binary somewhere in your system.)
 *
 * Besides the usual <tt>./configure</tt> options, the only really
 * interesting option that you can play with is <tt>--enable-simd</tt>,
 * which enables the SIMD optimizations discussed in the paper.
 *
 * Our code makes use of the following libraries if they can be found on
 * your system:
 *
 *  - POSIX real-time functionality (librt).
 *    Most importantly, we use the clock_nanosleep() from the POSIX
 *    real-time library.  If it cannot be found, we use gettimeofday()
 *    instead, but performance may suffer from that.
 *  - libnuma.
 *    To place threads and data properly on NUMA systems, we make use of
 *    the <tt>libnuma</tt> library.  Please make sure the library is
 *    installed in your system and can be found by <tt>./configure</tt>.
 *    Again, the code will run also without <tt>libnuma</tt>, but performance
 *    may suffer, particularly on large NUMA systems.
 *
 * We have successfully compiled and run our code on different Linux
 * variants; the experiments in the paper were performed on a Debian
 * system.  Most of the code was developed on Mac OS X Snow Leopard,
 * though neither real-time functionality nor libnuma are available there.
 *
 *
 * @section usage Usage
 *
 * @subsection invocation Invocation
 *
 * The <tt>sj</tt> binary understands the following command line options:
 *
 *  - <tt>-h</tt> prints a little help screen
 *  - <tt>-c NUM</tt> configures the system to use <tt>NUM</tt> cores/threads
 *    for <em>processing</em>.  At runtime there will be two additional
 *    threads: a join driver that pushes input data into handshake join
 *    and a result collector that collects and counts all result tuples.
 *  - <tt>-d PREFIX</tt>: To run the experiment, two random input streams
 *    will be generated.  With this command line option, the content of
 *    both streams will be printed to two files named <tt>[PREFIX].R</tt>
 *    and <tt>[PREFIX].S</tt>, respectively.  Both dump files can be used
 *    to compute the join result off-line and validate correctness.
 *  - <tt>-l FILE</tt> prints logging/debugging information to a file
 *    named <tt>FILE</tt>.
 *  - <tt>-n NUM</tt> number of input tuples to generate for input stream R.
 *  - <tt>-N NUM</tt> number of input tuples to generate for input stream S.
 *  - <tt>-o FILE</tt>: When running in ``GUI mode'', print information
 *    about individual cores' state to <tt>FILE</tt>.  A GUI can read this
 *    information and show a graphical representation of it to the user.
 *    This functionality is likely broken.
 *  - <tt>-O FILE</tt> writes the join result to file <tt>FILE</tt>.  Don't
 *    use this for performance experiments!
 *  - <tt>-r RATE</tt> tuple rate (in tuples/sec) for input stream R.
 *  - <tt>-R RATE</tt> tuple rate (in tuples/sec) for input stream S.
 *  - <tt>-v</tt> makes our system a tiny bit more verbose.
 *  - <tt>-w SIZE</tt> window size for stream R (in seconds).
 *  - <tt>-W SIZE</tt> window size for stream S (in seconds).
 *  - <tt>-g</tt> turns on GUI mode.  This is likely broken, use at your
 *    own risk!  (We had a primitive GUI for this at some point, but never
 *    maintained it.  We might revive this code and write up a demo paper
 *    for some conference.)
 *  - <tt>-i FILE</tt>: When in GUI mode, read commands from this file
 *    (usually a named pipe).  Don't use, the world might collapse if
 *    you do!
 *
 * @subsection running-experiments Running Experiments
 *
 * The above command line options can be used to instantiate a certain
 * configuration of handshake join and feed data into it.  If the
 * configuration can sustain the applied load, the join output will
 * be emitted and the program terminates normally.  Overload situations
 * will lead to overruns in the internal FIFO queues.  Once FIFOs overflow,
 * the program will terminate abnormally and print an error message.
 *
 * For the experiments in the paper, we wrote some Perl scripts that
 * determine the maximum possible load for every configuration using
 * ``try and error''.  Given upper and lower bounds for the stream rate
 * (known to be lower and higher than the maximum sustained throughput),
 * the tool <tt>find-max-rate.pl</tt> iteratively finds the maximum
 * supported rate for a given configuration (narrowing down the interval
 * bounds on every iteration).
 *
 * Another Perl script, <tt>iterate-cores.pl</tt>, iterates different
 * configurations.  For each configuration a prediction is made for the
 * uppoer and lower bounds (based on the previous result and an assumed
 * scalability) and <tt>find-max-rate.pl</tt> is invoked.
 *
 * Inherent to the problem, handshake join needs a certain amount of
 * warmup time (until both join windows are entirely full).  Therefore,
 * make sure you let the system run long enough (i.e., use enough input
 * tuples per stream) to obtain meaningful numbers.  We always made sure
 * the experiment runs at least three times as long as the window size.
 *
 * @subsection repeating Repeating Our Experiments
 *
 * The main entry point to repeat our experiments is the Perl script
 * <tt>iterate-cores.pl</tt>.  E.g., throughput numbers for a 15 minute
 * window and 4, 8, 16, and 32 CPU cores can be obtained as follows:
 *
 * @verbatim
     cd tools
     ./iterate-cores.pl --min 100 --max 2000 -w 900 -W 900 -d 2700 4 8 16 32
@endverbatim
 *
 * The parameters in the end specify the numbers of cores that should be
 * configured (first run will be with 4 cores, then 8, etc.).  Handshake
 * join instances (or better: the join driver) will be configured to use
 * a window size of 900 seconds for stream R (<tt>-w 900</tt>) and a window
 * size of 900 seconds for stream S (<tt>-W 900</tt>).  Input streams will
 * be generated to last 2700 seconds (note: streams should be significantly
 * longer than the window size; during the first 900 seconds inherently the
 * join windows will not have filled up, yet; effectively the system will
 * run for 1800 seconds under full load).
 *
 * Parameters <tt>--min 100</tt> and <tt>--max 2000</tt> assume lower and
 * upper bounds for the throughput of the first configuration (here: 4 cores).
 * (The script will blindly assume that 100 tuples per second can be
 * sustained, while 2000 cannot.)  The script will then, via bisection,
 * try to find the maximum throughput that the particular configuration
 * can sustain (i.e., it'll try a throughput of (100+2000)/2 = 1050 tuples
 * per second; if that succeeds, it'll try (1050+2000)/2 = 1525 tuples per
 * second, otherwise it'll try (100+1050)/2 = 575 tuples per second, etc.).
 *
 * After the throughput of the first configuration has been determined
 * (on our machine, we were able to sustain 995 tuples/sec on 4 cores),
 * the script makes an estimate for the throughput that can be sustained
 * in the next configuration.  To that estimate, it adds a safety margin
 * below and above the estimated throughput, then performs another bisection
 * search for the new configuration.
 *
 * At the end, the script will emit a summary like
 *
 * @verbatim
SUMMARY:
32      2806.9921875
28      2623.2421875
36      2966.96875
40      3129.13671875
12      1720.4140625
20      2215.765625
8       1404.6875
4       995.4140625
24      2434.65234375
16      1409.48828125
44      3290.10546875
@endverbatim
 *
 * @subsubsection internals-iterate Internals of the iterate-cores.pl Script
 *
 * The <tt>iterate-cores.pl</tt> script mainly does the iteration over
 * the given core configurations, and it computes the estimates for
 * configurations after the first one.
 *
 * The script will use the Linux <tt>script</tt> command to produce a
 * number of detailed log files.
 *
 * @subsubsection internals-find-max-rate Internals of the find-max-rate.pl Script
 *
 * <tt>iterate-cores.pl</tt> depends on another Perl script named
 * <tt>find-max-rate.pl</tt>.  The latter is the one that does the actual
 * bisection search.  It takes <tt>--min</tt>, <tt>--max</tt>, <tt>-w</tt>,
 * <tt>-W</tt>, and <tt>-d</tt> arguments much like <tt>iterate-cores.pl</tt>.
 * In addition it takes one single number of cores as the <tt>-c</tt>
 * argument.
 *
 *
 * @section implementation Implementation
 *
 * @subsection join-driver Join Driver
 *
 * During a run of an experiment, the code will first generate all data
 * for both input streams in memory.  Each tuple in those streams is
 * annotated with a time stamp (determined by the stream rate, with some
 * variations in the inter-tuple delay).  At runtime, the join driver
 * interprets those time stamps and feeds tuples into handshake join
 * accordingly.
 *
 * Data generation is implemented in generate_data().  Generated data
 * can be dumped to a file with a command line option (see above).
 *
 * To play the materialized streams at runtime, the join driver in
 * join_driver_exp() uses hj_nanosleep() to implement the inter-tuple
 * delay.  This uses the POSIX real-time function clock_nanosleep()
 * function if possible.  Otherwise, the functionality is simulated
 * with help of gettimeofday() (not a recommended setting).
 *
 * @subsection join-workers Join Workers
 *
 * Worker threads are created in setup_workers().  This routine sets up
 * all the FIFO queues, determines where (i.e., CPU in which NUMA region)
 * to place each worker, and where to allocate memories.  Worker threads
 * will be spawned and each worker thread gets its own worker context.
 *
 * @subsubsection memory Memory
 *
 * setup_workers() will allocate memory regions in all participating NUMA
 * regions.  The workers in this region will use the memory in the following
 * way.  The first worker that receives a chunk of data will read this data
 * from the originating NUMA region.  It will do its join processing while
 * reading the data and, as a side effect, it will physically copy the data
 * also into the local memory region.  Subsequent worker threads in the same
 * NUMA region will only read this data (and thus do NUMA-local memory
 * operations only).  To enable this functionality, setup_workers() will
 * set worker_ctx_t#copy_R and worker_ctx_t#copy_S flags in the worker
 * contexts accordingly.
 *
 * This is a prototype only, so we keep memory management <em>very</em>
 * simple.  Basically we just allocate enough memory in each NUMA region
 * that we can hold the entire stream there (our test machine has enough
 * memory so there's no need to be conservative here).  A real-world
 * implementation would probably allocate memory when data is copied into
 * the NUMA region.  De-allocation is a little bit more tricky, because
 * there are two destinations where the data (potentially) goes: (a) the
 * next NUMA region in the processing chain and (b) the result collector.
 *
 * If you turn on the verbose flag on the command line (<tt>-v</tt>), you
 * will see where each worker is being placed and whether or not it copies
 * any data for R or S.
 *
 * @subsubsection worker-implementation Worker Implementation
 *
 * The join workers themselves are implemented in handshake_join().  The
 * code there is a relatively straightforward implementation of the pseudo
 * code in the paper.
 *
 * If SIMD acceleration was requested during compilation (invoke
 * <tt>./configure</tt> with the <tt>--enable-simd</tt> option), the join
 * kernel is implemented with help of SIMD intrinsics.  See the code regions
 * that are predicated on the macro #ENABLE_SIMD.
 * 
 * Note: If have not tested any of the pathological configurations, such as
 *       a handshake join instance with only a single join worker (that
 *       worker would be the left-most and right-most worker at the same
 *       time).  Likely such configurations are broken.
 *
 * @subsection queues FIFO Queues
 *
 * FIFO queues are implemented as lock-free ringbuffers.  Sender and
 * receiver both spin when the queue is full/empty.  To avoid deadlock
 * situations, the sender aborts spinning after some timeout.  Since this
 * indicates a clear overload situation (we use FIFO queues of length 64,
 * configured via #MESSAGE_QUEUE_LENGTH in parameters.h), we
 * usually abort the program when a FIFO send() times out.
 *
 * All FIFO code is implemented in ringbuffer.c.
 *
 *
 * @subsection result-handling Result Handling
 *
 * Besides the FIFO queue to its neighbors, each join worker also has a
 * message queue to send result tuples to a central result collector.
 * This collector is implemented in collect_results().  The collector is
 * a separate thread.
 *
 * The collector vacuums all its input queues, puts itself to sleep for
 * a short moment (#COLLECT_INTERVAL), then repeats.
 *
 * If the writing of results to a file has been requested on the command
 * line (option <tt>-O</tt>), collect_results() will print all result
 * tuples to that file.  In any case, the collector will report the
 * current result tuple rate approximately every second.
 *
 *
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010, ETH Zurich, Systems Group
 *
 * $Id: main.c 954 2011-03-17 15:27:39Z jteubner $
 */

#include "config.h"
#include "parameters.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>

#ifdef HAVE_LIBNUMA
#include <numa.h>
#endif

#include "master/master.h"
#include "worker/worker.h"
#include "debug/debug.h"
#include "messages/core2core.h"
#include "data/data.h"
#include "time/time.h"
#include "mem/mem.h"

#define VERBOSE 1

/* ----- forward declarations ----- */
static void setup_workers (master_ctx_t *ctx);
static void * statistics_collector (void *arg);
static void * join_driver_gui (void *arg);
static void * join_driver_exp (void *arg);
static void * read_control_input (void *arg);
static void * collect_results (void *arg);
static void emit_result (master_ctx_t *ctx, unsigned int r, unsigned int s);
static inline void flush_result (master_ctx_t *ctx);

/**
 * Print usage information
 */
static void
usage (void)
{
    printf ("Program version $Id: main.c 954 2011-03-17 15:27:39Z jteubner $.\n");
    printf ("Usage:\n");
    printf ("  -h       this help screen\n");
    printf ("  -c NUM   number of processing cores/worker threads\n");
    printf ("  -d PREF  prefix for input stream dumps\n");
    printf ("  -l FILE  write a log file with many details (debugging)\n");
    printf ("  -n NUM   number of tuples to generate for stream R\n");
    printf ("  -N NUM   number of tuples to generate for stream S\n");
    printf ("  -o FILE  file name to write statistics to (default: stdout)\n");
    printf ("           (will append to FILE if it already exists)\n");
    printf ("  -O FILE  file name to write result stream to\n");
    printf ("  -r RATE  tuple rate for stream R (in tuples/sec)\n");
    printf ("  -R RATE  tuple rate for stream S (in tuples/sec)\n");
    printf ("  -v       be verbose\n");
    printf ("  -w SIZE  window size for stream R (in seconds)\n");
    printf ("  -W SIZE  window size for stream S (in seconds)\n");
    printf ("  -g       assume control by GUI (don't pre-generate data)\n");
    printf ("           This is likely broken at this time!  Don't use!\n");
    printf ("  -i FILE  will read commands from this file (req. for GUI)\n");
    printf ("\nHave fun!\n");
}

int
main (int argc, char **argv)
{
    master_ctx_t *ctx = malloc (sizeof (*ctx));
    int           status;
    pthread_t     collector;
    pthread_t     driver;
    int           ch;

    ctx->outfile     = stdout;
    ctx->infile      = NULL;
    ctx->logfile     = NULL;
    ctx->resultfile  = NULL;
    ctx->verbose     = false;
    ctx->num_workers = 8;

    ctx->use_gui           = false;
    ctx->num_tuples_R      = 1800000;
    ctx->num_tuples_S      = 1800000;
    ctx->rate_R            = 950;
    ctx->rate_S            = 950;
    ctx->window_size_R     = 600;
    ctx->window_size_S     = 600;
    ctx->int_value_range   = INT_MAX;//10000;
    ctx->float_value_range = INT_MAX;//10000;
    ctx->data_prefix       = NULL;

    /* parse command lines */
    while ((ch = getopt (argc, argv, "c:d:ghi:l:n:N:o:O:r:R:vw:W:")) != -1)
    {
        switch (ch)
        {
            case 'c':
                ctx->num_workers = strtol (optarg, NULL, 10);
                if (ctx->num_workers == 0)
                {
                    fprintf (stderr, "invalid number of cores\n");
                    exit (EXIT_FAILURE);
                }
                break;

            case 'd':
                {
                    const unsigned int l = strlen (optarg);
                    ctx->data_prefix = malloc (l + 1);
                    assert (ctx->data_prefix);
                    strncpy (ctx->data_prefix, optarg, l);
                    ctx->data_prefix[l] = '\0';
                }
                break;

            case 'g':
                ctx->use_gui = true;
                break;

            case 'i':
                if (!(ctx->infile = fopen (optarg, "r")))
                {
                    fprintf (stderr, "unable to open %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
                break;

            case 'l':
                if (!(ctx->logfile = fopen (optarg, "w")))
                {
                    fprintf (stderr, "unable to open %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
                break;

            case 'n':
                ctx->num_tuples_R = strtol (optarg, NULL, 10);
                break;

            case 'N':
                ctx->num_tuples_S = strtol (optarg, NULL, 10);
                break;

            case 'o':
                if (!(ctx->outfile = fopen (optarg, "a")))
                {
                    fprintf (stderr, "unable to open %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
                break;

            case 'O':
                if (!(ctx->resultfile = fopen (optarg, "w")))
                {
                    fprintf (stderr, "unable to open %s\n", optarg);
                    exit (EXIT_FAILURE);
                }
                break;

            case 'r':
                ctx->rate_R = strtol (optarg, NULL, 10);
                break;

            case 'R':
                ctx->rate_S = strtol (optarg, NULL, 10);
                break;

            case 'v':
                ctx->verbose = true;
                break;

            case 'w':
                ctx->window_size_R = strtol (optarg, NULL, 10);
                break;

            case 'W':
                ctx->window_size_S = strtol (optarg, NULL, 10);
                break;

            case 'h':
            case '?':
            default:
                usage (); exit (EXIT_SUCCESS);
        }
    }

    /* a few words of welcome */
    fprintf (ctx->outfile, "# Stream Join (“Handshake Join”) Prototype.\n");
    fprintf (ctx->outfile, "# Brought to you by Jens Teubner <jens.teubner@inf.ethz.ch>.\n");
    fprintf (ctx->outfile, "# (c) 2010 ETH Zurich, Systems Group\n");
    fprintf (ctx->outfile, "# \n");

    fflush (ctx->outfile);

    if (ctx->use_gui && ! ctx->infile)
    {
        fprintf (stderr, "No input file (command file) given.\n");
        usage ();
        exit (EXIT_FAILURE);
    }

#ifdef HAVE_LIBNUMA
    fprintf (ctx->outfile, "# Migrating to NUMA node 0...\n");
    numa_run_on_node (0);
    fprintf (ctx->outfile, "# NUMA migration done.\n");
#endif

    /*
     * Sanitize input parameters a bit; make sure that number of tuples
     * is a multiple of the chunk size.  The implementation of our join
     * driver will make trouble otherwise.
     */
    if (ctx->num_tuples_R % TUPLES_PER_CHUNK_R)
        ctx->num_tuples_R += TUPLES_PER_CHUNK_R
            - (ctx->num_tuples_R % TUPLES_PER_CHUNK_R);

    if (ctx->num_tuples_S % TUPLES_PER_CHUNK_S)
        ctx->num_tuples_S += TUPLES_PER_CHUNK_S
            - (ctx->num_tuples_S % TUPLES_PER_CHUNK_S);

    fprintf (ctx->outfile, "# Generating input data...\n");
    fprintf (ctx->outfile, "# Using parameters:\n");
    fprintf (ctx->outfile, "#   - num_tuples_R: %u; num_tuples_S: %u\n",
            ctx->num_tuples_R, ctx->num_tuples_S);
    fprintf (ctx->outfile, "#   - rate_R: %u; rate_S: %u\n",
            ctx->rate_R, ctx->rate_S);
    fprintf (ctx->outfile, "#   - window_size_R: %u; window_size_S: %u\n",
            ctx->window_size_R, ctx->window_size_S);
    generate_data (ctx);
    fprintf (ctx->outfile, "# Data generation done.\n");

    fprintf (ctx->outfile, "# Setting up handshake join workers...\n");
    setup_workers (ctx);
    fprintf (ctx->outfile, "# Worker setup done.\n");

    ctx->collect_interval = 1;

    if (ctx->use_gui)
    {
        fprintf (ctx->outfile, "# Setting up statistics collector...\n");
        status = pthread_create (&collector, NULL, statistics_collector, ctx);
        assert (status == 0);
        fprintf (ctx->outfile, "# Collector setup done.\n");
        printf ("#\n");
    }
    else
    {
        fprintf (ctx->outfile, "# Setting up result collector...\n");
        status = pthread_create (&collector, NULL, collect_results, ctx);
        assert (status == 0);
        fprintf (ctx->outfile, "# Collector setup done.\n");
        printf ("#\n");
    }

    fflush (ctx->outfile);

    /* TODO: initialize driver parameters and spawn driver thread */
    ctx->driver_interval = 0;
    ctx->r_divider = 10;
    ctx->s_divider = 10;
    ctx->r_size    = 1000;
    ctx->s_size    = 1000;

    ctx->left_dummy_ctx->wnd_S_start = 0;
    ctx->left_dummy_ctx->wnd_S_sent  = 0;
    ctx->right_dummy_ctx->wnd_R_start = 0;
    ctx->right_dummy_ctx->wnd_R_sent  = 0;

    if (ctx->use_gui)
    {
        fprintf (ctx->outfile,
                "# Setting up driver thread (GUI-controlled)...");
        status = pthread_create (&driver, NULL, join_driver_gui, ctx);
        assert (status == 0);
        fprintf (ctx->outfile, "done.\n");
        fprintf (ctx->outfile, "#\n");

        fflush (ctx->outfile);

        read_control_input ((void *) ctx);
    }
    else
        join_driver_exp (ctx);

    return EXIT_SUCCESS;
}

/**
 * Setup all worker contexts and spawn worker threads.
 *
 * @todo This is where most of the NUMA stuff has to be done.
 *
 * @param ctx a master context; must be populated already with the
 *            #num_workers information; this function will instantiate
 *            queues, setup all worker contexts, and add such information
 *            to #ctx.
 */
static void
setup_workers (master_ctx_t *ctx)
{
    int            status;
    unsigned int   num_workers;

    num_workers = ctx->num_workers;

    /*
     * Queues used to send data from worker to worker left-to-right.
     * Each of the NUM_WORKERS threads gets one left input queue and
     * one right input queue.
     */
    ctx->left_queues = malloc (num_workers * sizeof (ringbuffer_t *));
    assert (ctx->left_queues);

    ctx->right_queues = malloc (num_workers * sizeof (ringbuffer_t *));
    assert (ctx->right_queues);

    /* one extra queue from join driver to result collector */
    ctx->result_queues = malloc ((num_workers+1) * sizeof (ringbuffer_t *));
    assert (ctx->result_queues);

    ctx->worker_ctxs = malloc (num_workers * sizeof (worker_ctx_t *));
    assert (ctx->worker_ctxs);

    ctx->workers = malloc (num_workers * sizeof (pthread_t));
    assert (ctx->workers);

    ctx->left_dummy_ctx = malloc (sizeof (worker_ctx_t));
    assert (ctx->left_dummy_ctx);

    ctx->right_dummy_ctx = malloc (sizeof (worker_ctx_t));
    assert (ctx->right_dummy_ctx);

    /* Determine NUMA availability */
#ifdef HAVE_LIBNUMA
    if (numa_available () >= 0)
        ctx->use_numa = true;
    else
    {
        fprintf (stderr,
                "WARNING: Your system does not seem to be NUMA-capable.\n");
        ctx->use_numa = false;
    }
#else
    ctx->use_numa = false;
#endif

#ifdef HAVE_LIBNUMA
    if (ctx->use_numa)
    {
        ctx->numa_nodes = numa_max_node () + 1;

        assert (ctx->numa_nodes >= 1);

        if (ctx->verbose)
            fprintf (ctx->outfile, "# Your system has %u NUMA nodes.\n",
                    ctx->numa_nodes);
    }
#endif

    /*
     * allocate memory for worker contexts
     */
    for (unsigned int i = 0; i < num_workers; i++)
    {
        const int numa_node = (i+1) * ctx->numa_nodes / (num_workers + 2);

        ctx->worker_ctxs[i]
            = alloc_onnode (sizeof (*(ctx->worker_ctxs[i])), numa_node);

        assert (ctx->worker_ctxs[i]);
    }

#ifdef HAVE_LIBNUMA
    if (ctx->use_numa)
    {

        /*
         * (personal notes, ignore them if they don't make sense to you)
         *
         * Total number of threads:
         *
         *  - N Worker threads
         *  - driver thread
         *  - statistics/result collection
         *
         * Thus: x = (N + 2) / ctx->numa_nodes threads per NUMA node.
         *
         * Current thread index: n (0-based)
         *
         * NUMA region for n: (n+1) * ctx->numa_nodes / (N+2)
         */

        /* "Place" workers on NUMA nodes. */
        for (unsigned int i = 0; i < num_workers; i++)
        {
            ctx->worker_ctxs[i]->numa_node
                = (i+1) * ctx->numa_nodes / (num_workers + 2);

            ctx->worker_ctxs[i]->copy_R
                = (i == 0)
                  || ( (i * ctx->numa_nodes / (num_workers + 2))
                          != ((i+1) * ctx->numa_nodes / (num_workers + 2)));

            ctx->worker_ctxs[i]->copy_S
                = (i == num_workers - 1)
                  || ( ((i+2) * ctx->numa_nodes / (num_workers + 2))
                          != ((i+1) * ctx->numa_nodes / (num_workers + 2)));

            if (ctx->verbose)
                fprintf (ctx->outfile,
                        "# Worker %u placed on NUMA node %i%s%s.\n",
                        i, ctx->worker_ctxs[i]->numa_node,
                        ctx->worker_ctxs[i]->copy_R ? "; copies R" : "",
                        ctx->worker_ctxs[i]->copy_S ? "; copies S" : "");
        }
    }
    else
    {
        /* numa_node == -1 signifies to workers that there is no NUMA */
        for (unsigned int i = 0; i < num_workers; i++)
            ctx->worker_ctxs[i]->numa_node = -1;
    }
#else

    /* numa_node == -1 signifies to workers that there is no NUMA */
    for (unsigned int i = 0; i < num_workers; i++)
        ctx->worker_ctxs[i]->numa_node = -1;

#endif

    /* ring buffers */
    for (unsigned int i = 0; i < num_workers; i++)
    {
        /* FIXME: Consider NUMA here */
        ctx->left_queues[i]
            = new_ringbuffer (MESSAGE_QUEUE_LENGTH,
                              ctx->worker_ctxs[i]->numa_node);
        ctx->right_queues[i]
            = new_ringbuffer (MESSAGE_QUEUE_LENGTH,
                              ctx->worker_ctxs[i]->numa_node);
        ctx->result_queues[i]
            = new_ringbuffer (MESSAGE_QUEUE_LENGTH, ctx->numa_nodes - 1);
    }

    /* worker contexts */
    for (unsigned int i = 0; i < num_workers; i++)
    {
        /* setup context */
        ctx->worker_ctxs[i]->id               = i;
        ctx->worker_ctxs[i]->left_recv_queue  = ctx->left_queues[i];
        ctx->worker_ctxs[i]->right_recv_queue = ctx->right_queues[i];
        ctx->worker_ctxs[i]->left_send_queue
            = i > 0 ? ctx->right_queues[i-1] : NULL,
        ctx->worker_ctxs[i]->right_send_queue
            = i < (num_workers-1) ? ctx->left_queues[i+1] : NULL,
        ctx->worker_ctxs[i]->left_ctx
            = i > 0 ? ctx->worker_ctxs[i-1] : ctx->left_dummy_ctx,
        ctx->worker_ctxs[i]->right_ctx
            = i < (num_workers-1)
              ? ctx->worker_ctxs[i+1] : ctx->right_dummy_ctx;
        ctx->worker_ctxs[i]->num_workers      = num_workers;
        ctx->worker_ctxs[i]->result_queue     = ctx->result_queues[i];

        /* spawn a thread */
        status = pthread_create (&(ctx->workers[i]), NULL, handshake_join,
                                 ctx->worker_ctxs[i]);
        assert (status == 0);
    }

    /* allocate local memories (for input stream R) */
    x_t *x = ctx->R.x;
    y_t *y = ctx->R.y;
    for (int i = 0; i < num_workers; i++)
    {
        if (ctx->worker_ctxs[i]->copy_R)
        {
            /* FIXME: NUMA alloc */
            x = alloc_onnode (
                    ALLOCSIZE_PER_NUMA_R * sizeof (*x),
                    ctx->worker_ctxs[i]->numa_node);
            assert (x);
            y = alloc_onnode (
                    ALLOCSIZE_PER_NUMA_R * sizeof (*y),
                    ctx->worker_ctxs[i]->numa_node);
            assert (y);
        }
        ctx->worker_ctxs[i]->R.x = x;
        ctx->worker_ctxs[i]->R.y = y;
    }

    /* allocate local memories (for input stream S) */
    a_t *a = ctx->S.a;
    b_t *b = ctx->S.b;
    for (int i = num_workers-1; i >= 0; i--)
    {
        if (ctx->worker_ctxs[i]->copy_S)
        {
            /* FIXME: NUMA alloc */
            a = alloc_onnode (
                    ALLOCSIZE_PER_NUMA_S * sizeof (*a),
                    ctx->worker_ctxs[i]->numa_node);
            assert (a);
            b = alloc_onnode (
                    ALLOCSIZE_PER_NUMA_S * sizeof (*b),
                    ctx->worker_ctxs[i]->numa_node);
            assert (b);
        }
        ctx->worker_ctxs[i]->S.a = a;
        ctx->worker_ctxs[i]->S.b = b;
    }

    /* some nodes will copy data from their neighbors */
    ctx->left_dummy_ctx->R.x = ctx->R.x;
    ctx->left_dummy_ctx->R.y = ctx->R.y;
    ctx->right_dummy_ctx->S.a = ctx->S.a;
    ctx->right_dummy_ctx->S.b = ctx->S.b;

    /* access to the pipeline for the master */
    ctx->data_R_queue = ctx->worker_ctxs[0]->left_recv_queue;
    ctx->ack_R_queue  = ctx->worker_ctxs[0]->left_send_queue;
    ctx->data_S_queue = ctx->worker_ctxs[num_workers-1]->right_recv_queue;
    ctx->ack_S_queue  = ctx->worker_ctxs[num_workers-1]->right_send_queue;

    ctx->result_queue = ctx->result_queues[num_workers]
        = new_ringbuffer (MESSAGE_QUEUE_LENGTH, 0);

}


/**
 * Collector to be used in combination with GUI front-end; periodically
 * reads out some statistics about all iter-core queues and prints them
 * to a "file" (typically a FIFO that is consumed by the front-end).
 *
 * @param arg a master context (with @c void pointer type to comply with
 *            pthread_create() interface.
 *
 * @return This function never returns.
 */
static void *
statistics_collector (void *arg)
{
    master_ctx_t *ctx = (master_ctx_t *) arg;

    fprintf (ctx->outfile, "# core |  wnd_R  |  fwd_R  |  wnd_S  |  fwd_S\n");
    fprintf (ctx->outfile, "#------+---------+---------+---------+---------\n");
    fflush (ctx->outfile);

    while (true)
    {
        sleep (ctx->collect_interval);

        for (unsigned int i = 0; i < ctx->num_workers; i++)
        {
            fprintf (ctx->outfile, "   %2u  | %7u | %7u | %7u | %7u\n",
                    i,
                    ctx->worker_ctxs[i]->wnd_R_end
                        - ctx->worker_ctxs[i]->wnd_R_start,
                    ctx->worker_ctxs[i]->wnd_R_sent
                        - ctx->worker_ctxs[i]->wnd_R_start,
                    ctx->worker_ctxs[i]->wnd_S_end
                        - ctx->worker_ctxs[i]->wnd_S_start,
                    ctx->worker_ctxs[i]->wnd_S_sent
                        - ctx->worker_ctxs[i]->wnd_S_start);
        }

        fflush (ctx->outfile);
    }

    return NULL;
}

static inline bool
send_new_R_tuple (master_ctx_t *ctx, unsigned int start_idx, unsigned int size)
{
    core2core_msg_t msg = (core2core_msg_t) {
        .type = new_R_msg,
        .msg.chunk_R = (chunk_R_msg_t) { .start_idx = start_idx,
                                         .size      = size } };

    bool ret = send (ctx->data_R_queue, &msg, sizeof (msg));

    LOG(ctx->logfile, "sent new R [%u:%u] to queue %p (%s)",
            start_idx, size, ctx->data_R_queue,
            ret ? "successful" : "FAILED");

    if (!ret)
    {
        fprintf (stderr, "Cannot send R tuple. FIFO queue full.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}

static inline bool
send_R_ack (master_ctx_t *ctx, unsigned int start_idx, unsigned int size)
{
    core2core_msg_t msg = (core2core_msg_t) {
        .type = ack_R_msg,
        .msg.ack_R = (ack_R_msg_t) { .start_idx = start_idx,
                                     .size      = size } };

    bool ret = send (ctx->data_S_queue, &msg, sizeof (msg));

    LOG(ctx->logfile, "sent ACK R [%u:%u] to queue %p (%s)",
            start_idx, size, ctx->data_S_queue,
            ret ? "successful" : "FAILED");

    if (!ret)
    {
        fprintf (stderr, "Cannot send R acknowledgement. FIFO queue full.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}

static inline bool
send_new_S_tuple (master_ctx_t *ctx, unsigned int start_idx, unsigned int size)
{
    core2core_msg_t msg = (core2core_msg_t) {
        .type = new_S_msg,
        .msg.chunk_S = (chunk_S_msg_t) { .start_idx = start_idx,
                                         .size      = size } };

    bool ret = send (ctx->data_S_queue, &msg, sizeof (msg));

    LOG(ctx->logfile, "sent new S [%u:%u] to queue %p (%s)",
            start_idx, size, ctx->data_S_queue,
            ret ? "successful" : "FAILED");

    if (!ret)
    {
        fprintf (stderr, "Cannot send S tuple. FIFO queue full.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}

static inline bool
send_S_ack (master_ctx_t *ctx, unsigned int start_idx, unsigned int size)
{
    core2core_msg_t msg = (core2core_msg_t) {
        .type = ack_S_msg,
        .msg.ack_S = (ack_S_msg_t) { .start_idx = start_idx,
                                     .size      = size } };

    bool ret = send (ctx->data_R_queue, &msg, sizeof (msg));

    LOG(ctx->logfile, "sent ACK S [%u:%u] to queue %p (%s)",
            start_idx, size, ctx->data_R_queue,
            ret ? "successful" : "FAILED");

    if (!ret)
    {
        fprintf (stderr, "Cannot send S acknowledgement. FIFO queue full.\n");
        exit (EXIT_FAILURE);
    }

    return ret;
}

/**
 * Send a "flush result queues" message to the specified FIFO queue.
 * These messages will be propagated from core to core, and each core
 * that receives an instruction to flush will send a result message
 * to the collector process, no matter how full the result buffer is.
 */
static inline bool
send_flush (master_ctx_t *ctx, ringbuffer_t *buf)
{
    core2core_msg_t msg = (core2core_msg_t) { .type = flush_msg };

    bool ret = send (buf, &msg, sizeof (msg));

    LOG(ctx->logfile, "sent flush message to queue %p (%s)",
            buf, ret ? "successful" : "FAILED");

    return ret;
}

/**
 * Join driver process to be used in combination with the GUI front-end.
 *
 * The idea is a loop where the driver
 *
 *  # sleeps for a controllable amount of time
 *  # whenever a controllable number of iterations have been made,
 *    an R and/or S tuple is sent into the handshake join pipeline.
 *  # acknowledgement messages are created to keep a controllable
 *    window size (tuple-based, not time-based).
 *
 * @note This function is likely broken.  Use at your own risk.
 */
static void *
join_driver_gui (void *arg)
{
    master_ctx_t *ctx = (master_ctx_t *) arg;
    unsigned int  counter = 1;
    unsigned int  wnd_size_R = 0;
    unsigned int  wnd_size_S = 0;

    ctx->driver_interval = 0;
    ctx->r_divider = 10;
    ctx->s_divider = 10;
    ctx->r_size    = 1000;
    ctx->s_size    = 1000;

    ctx->left_dummy_ctx->wnd_S_start = 0;
    ctx->left_dummy_ctx->wnd_S_sent  = 0;
    ctx->right_dummy_ctx->wnd_R_start = 0;
    ctx->right_dummy_ctx->wnd_R_sent  = 0;

    while (true)
    {
        /*
         * If the driver interval is set to 0, we only send messages
         * upon explicit commands from the user.
         */
        if (ctx->driver_interval == 0)
        {
            usleep (10000);
            continue;
        }

        usleep (ctx->driver_interval);

        if (counter % ctx->r_divider == 0)
        {
            /* send an R tuple/chunk */
            if (! send_new_R_tuple (ctx, 0, 0))
                fprintf (stderr, "input queue for stream R seems full\n");
            else
                wnd_size_R++;
        }

        if (counter % ctx->s_divider == 0)
        {
            /* send an S tuple/chunk */
            if (! send_new_S_tuple (ctx, 0, 0))
                fprintf (stderr, "input queue for stream S seems full\n");
            else
                wnd_size_S++;
        }

        while (wnd_size_R > ctx->r_size)
        {
            bool result = true;

            result = send_R_ack (ctx, 0, 0);

            if (!result)
            {
                fprintf (stderr, "input queue for stream S seems full, "
                        "will retry to send ACK for R\n");
                break;
            }

            wnd_size_R--;
        }

        while (wnd_size_S > ctx->s_size)
        {
            bool result = true;
            
            result = send_S_ack (ctx, 0, 0);

            if (!result)
            {
                fprintf (stderr, "input queue for stream R seems full, "
                        "will retry to send ACK for S\n");
                break;
            }

            wnd_size_S--;
        }

        counter++;

        /*
         * Simulate load for left and right dummy contexts.
         */
        ctx->left_dummy_ctx->wnd_S_end = 0;
        ctx->right_dummy_ctx->wnd_R_end = 0;
        /*
        ctx->left_dummy_ctx->wnd_S_end = wnd_size_S / ctx->num_workers;
        ctx->right_dummy_ctx->wnd_R_end = wnd_size_R / ctx->num_workers;
        */

    }

    return NULL;
}

/**
 * Join driver to be used for actual experiments.
 *
 * We assume two pre-generated data streams.  This function iterates
 * and, for every iteration,
 *
 *  # it determines when the next tuple (an R or an S tuple) has to be
 *    sent to the workers.
 *  # It then sleeps for a moment, until that tuple is due (we use
 *    POSIX real-time functionality for this purpose).
 *  # Before actually sending the tuple, we see whether some other tuples
 *    have to expire first (from either stream) and send respective
 *    acknowledgement messages.
 *  # Finally, we send the tuple itself and go into the next iteration.
 *
 * Once we've replayed the full two streams, we wait for another while.
 * While waiting, we let tuples expire, but don't feed in any new data.
 * Afterward, both pipelines should be entirely empty.  Then we send
 * a "flush" instruction to the pipeline (to flush all result buffers),
 * wait a few seconds to make sure result messages have arrived, then
 * we return (which ultimately terminates the program).
 *
 * Sending of tuples and acknowledgements happens in chunks (their size
 * is configurable in parameters.h).  This requires some extra care to
 * make sure we adhere to the time-based window semantics.  To this end,
 * we perform some minor scans/joins here, and we have our own result
 * queue to the result collector.
 *
 * @param arg a master context (as a @c void pointer to comply with the
 *            pthread_create() interface
 */
static void *
join_driver_exp (void *arg)
{
    master_ctx_t *ctx = (master_ctx_t *) arg;

    /* is the next tuple to process an R tuple (if not, it's an S tuple) */
    bool next_is_R;

    /* "current" tuple, whose time stamp will expire next */
    unsigned int r = 0;

    /*
     * last tuple that we actually sent out; we'll always send tuples
     * in batches, i.e., when r - r_last_sent exceeds some threshold
     */
    unsigned int r_last_sent = 0;

    /* end of the current R window; used to manage tuple expiration */
    unsigned int r_end = 0;

    /* number of chunks that we had to drop, because input FIFOs were full */
    unsigned int r_dropped = 0;

    unsigned int s = 0;
    unsigned int s_last_sent = 0;
    unsigned int s_end = 0;
    unsigned int s_dropped = 0;

    /* timestamp for next tuple to send, relative to start of experiment */
    struct timespec t_rel;
    /* timestamp as real time, i.e., t_rel + time measured at begin */
    struct timespec t_real;
    /* difference t_real - t_rel in seconds */
    time_t t_offset;

    struct timespec t_start;
    struct timespec t_end;

    /*
     * get current time; we'll add that to the time stamps in the
     * input data before we call clock_nanosleep()
     */
    if (hj_gettime (&t_start))
    {
        fprintf (stderr,
                "Something went wrong with the real time interface.\n");
        fprintf (stderr, "A call to hj_gettime() failed.\n");
        exit (EXIT_FAILURE);
    }

    /* add a few seconds delay to play safe */
    t_start.tv_sec += 5;
    t_offset = t_start.tv_sec;

    LOG(ctx->logfile, "timer offset is %lu", t_offset);

    while (r < ctx->num_tuples_R || s < ctx->num_tuples_S)
    {
        /* is the next tuple an R or an S tuple? */
        next_is_R = (ctx->R.t[r].tv_sec * 1000000000L + ctx->R.t[r].tv_nsec)
            < (ctx->S.t[s].tv_sec * 1000000000L + ctx->S.t[s].tv_nsec);

        /* sleep until we have to send the next tuple */
        if (next_is_R)
            t_rel = ctx->R.t[r];
        else
            t_rel = ctx->S.t[s];

        t_real = (struct timespec) { .tv_sec  = t_rel.tv_sec + t_offset,
                                     .tv_nsec = t_rel.tv_nsec };

        LOG(ctx->logfile, "sleeping until %lu.%09li", t_real.tv_sec,
                t_real.tv_nsec);

        hj_nanosleep (&t_real);

        /* let old tuples expire; R tuples first */
        while (r_end < r
                && (ctx->R.t[r_end].tv_sec*1000000000L + ctx->R.t[r_end].tv_nsec
                    + ctx->window_size_R * 1000000000L)
                    < (t_rel.tv_sec * 1000000000L + t_rel.tv_nsec))
        {
            /*
             * Processing tuples in chunks can cause problems here.
             *
             * Situation:
             *
             *  1. An S tuple is logically sent, but only queued up in
             *     the next S chunk (that is, it will not actually be
             *     sent to a worker).
             *
             *  2. It is time to expire some old tuples, so we send an
             *     acknowledgement here.
             *
             *  3. When the S chunk is sent to the worker now, it will
             *     not see tuples that were expired with the above
             *     acknowledgement.
             *
             * We avoid this by explicitly looking at not-yet-sent tuples
             * before we send out the acknowledgement.
             */
            for (unsigned int r_ = r_end; r_ < r_end + TUPLES_PER_CHUNK_R; r_++)
            {
                for (unsigned int s_ = s_last_sent; s_ < s; s_++)
                {
                    const a_t a = ctx->S.a[s_] - ctx->R.x[r_];
                    const b_t b = ctx->S.b[s_] - ctx->R.y[r_];
                    if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                        emit_result (ctx, r_, s_);
                }
            }

            if (send_R_ack (ctx, r_end, TUPLES_PER_CHUNK_R))
                r_end += TUPLES_PER_CHUNK_R;
            else
                break;
        }

        /* let old tuples expire; now the S tuples */
        while (s_end < s
                && (ctx->S.t[s_end].tv_sec*1000000000L + ctx->S.t[s_end].tv_nsec
                    + ctx->window_size_S * 1000000000L)
                    < (t_rel.tv_sec * 1000000000L + t_rel.tv_nsec))
        {
            /* see above */
            for (unsigned int s_ = s_end; s_ < s_end + TUPLES_PER_CHUNK_S; s_++)
            {
                for (unsigned int r_ = r_last_sent; r_ < r; r_++)
                {
                    const a_t a = ctx->S.a[s_] - ctx->R.x[r_];
                    const b_t b = ctx->S.b[s_] - ctx->R.y[r_];
                    if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                        emit_result (ctx, r_, s_);
                }
            }


            if (send_S_ack (ctx, s_end, TUPLES_PER_CHUNK_S))
                s_end += TUPLES_PER_CHUNK_S;
            else
                break;
        }

        /* send tuple */
        if (next_is_R)
        {
            /*
             * Processing tuples in chunks leads to problems with the
             * exact time stamps.
             *
             * Since tuple expiration looks at the oldest tuple in each
             * chunk, some tuples in the last acknowledged chunk may
             * actually have to be joined with this new tuple.  This is
             * what we "repair" here.
             */
            for (int s_ = s_end-1;
                    s_ >= 0
                    && (ctx->S.t[s_].tv_sec * 1000000000L + ctx->S.t[s_].tv_nsec
                        + ctx->window_size_S * 1000000000L)
                       >= (ctx->R.t[r].tv_sec * 1000000000L
                           + ctx->R.t[r].tv_nsec);
                    s_--)
            {
                const a_t a = ctx->S.a[s_] - ctx->R.x[r];
                const b_t b = ctx->S.b[s_] - ctx->R.y[r];
                if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                    emit_result (ctx, r, s_);
            }

            r++;

            /* only actually send tuple when the chunk is full */
            if ((r - r_last_sent) == TUPLES_PER_CHUNK_R)
            {
                r_dropped +=
                    ! send_new_R_tuple (ctx, r_last_sent, TUPLES_PER_CHUNK_R);
                r_last_sent = r;
            }
        }
        else
        {
            /*
             * Processing tuples in chunks leads to problems with the
             * exact time stamps.
             *
             * Since tuple expiration looks at the oldest tuple in each
             * chunk, some tuples in the last acknowledged chunk may
             * actually have to be joined with this new tuple.  This is
             * what we "repair" here.
             */
            for (int r_ = r_end-1;
                    r_ >= 0
                    && (ctx->R.t[r_].tv_sec * 1000000000L + ctx->R.t[r_].tv_nsec
                        + ctx->window_size_R * 1000000000L)
                       >= (ctx->S.t[s].tv_sec * 1000000000L
                           + ctx->S.t[s].tv_nsec);
                    r_--)
            {
                const a_t a = ctx->S.a[s] - ctx->R.x[r_];
                const b_t b = ctx->S.b[s] - ctx->R.y[r_];
                if ((a > -10) & (a < 10) & (b > -10.) & (b < 10.))
                    emit_result (ctx, r_, s);
            }

            s++;

            /* only actually send tuple when the chunk is full */
            if ((s - s_last_sent) == TUPLES_PER_CHUNK_S)
            {
                s_dropped +=
                    ! send_new_S_tuple (ctx, s_last_sent, TUPLES_PER_CHUNK_S);
                s_last_sent = s;
            }
        }

    }

    /*
     * This is the time when we sent out all data items.  We use that
     * as the "finish time" of our experiment.
     */
    hj_gettime (&t_end);

    fprintf (stderr, "Data all fed into handshake join, now sending ACKs.\n");
    fflush (stderr);

    /*
     * Now flush the two handshake join pipelines.
     */
    while (r_end < ctx->num_tuples_R || s_end < ctx->num_tuples_S)
    {
        /* is the next tuple to acknowledge an R or an S tuple? */
        next_is_R =
            (ctx->R.t[r_end].tv_sec * 1000000000L + ctx->R.t[r_end].tv_nsec)
            < (ctx->S.t[s_end].tv_sec * 1000000000L + ctx->S.t[s_end].tv_nsec);

        /* sleep until we have to send the next acknowledgement */
        if (next_is_R)
            t_real = (struct timespec) {
                .tv_sec = ctx->R.t[r_end].tv_sec + t_offset +ctx->window_size_R,
                .tv_nsec = ctx->R.t[r_end].tv_nsec };
        else
            t_real = (struct timespec) {
                .tv_sec = ctx->S.t[s_end].tv_sec + t_offset +ctx->window_size_S,
                .tv_nsec = ctx->S.t[s_end].tv_nsec };

        LOG(ctx->logfile, "sleeping until %lu.%09li", t_real.tv_sec,
                t_real.tv_nsec);

        hj_nanosleep (&t_real);

        if (next_is_R)
        {
            if (send_R_ack (ctx, r_end, TUPLES_PER_CHUNK_R))
                r_end += TUPLES_PER_CHUNK_R;
        }
        else
        {
            if (send_S_ack (ctx, s_end, TUPLES_PER_CHUNK_S))
                s_end += TUPLES_PER_CHUNK_S;
        }

    }

    fprintf (stderr, "Flushing result buffers.\n");
    fflush (stderr);

    send_flush (ctx, ctx->data_R_queue);
    send_flush (ctx, ctx->data_S_queue);

    flush_result (ctx);

    sleep (5);

    fprintf (stderr, "Generated %u/%u tuples for R/S in %li.%09li sec.\n",
            ctx->num_tuples_R, ctx->num_tuples_S,
            ((t_end.tv_sec * 1000000000 + t_end.tv_nsec)
              - (t_start.tv_sec * 1000000000 + t_start.tv_nsec)) / 1000000000L,
            ((t_end.tv_sec * 1000000000 + t_end.tv_nsec)
              - (t_start.tv_sec * 1000000000 + t_start.tv_nsec)) % 1000000000L);
    fprintf (stderr, "This corresponds to a rate of %8.2f/%8.2f tuples/sec.\n",
            (double) ctx->num_tuples_R /
                ((t_end.tv_sec * 1000000000 + t_end.tv_nsec)
                 - (t_start.tv_sec * 1000000000 + t_start.tv_nsec)) / 1e9,
            (double) ctx->num_tuples_S /
                ((t_end.tv_sec * 1000000000 + t_end.tv_nsec)
                 - (t_start.tv_sec * 1000000000 + t_start.tv_nsec)) / 1e9);
    if (r_dropped || s_dropped)
        fprintf (stderr, "WARNING: %u/%u R/S tuples had to be dropped "
                "because FIFOs were full!\n",
                r_dropped * TUPLES_PER_CHUNK_R, s_dropped * TUPLES_PER_CHUNK_S);

    fflush (stderr);

    exit (EXIT_SUCCESS);

    return NULL;
}

/**
 * "Send" a result tuple to the result collector.  In reality, tuples
 * are batched up and #RESULTS_PER_MESSAGE tuples are sent to the
 * collector in a single message.
 *
 * For each result tuple, only the row numbers of the R/S tuples
 * are reported to the master.  The master keeps its own copy of
 * all data (workers only know the join arguments anyway) and uses
 * row numbers to construct result tuples.
 *
 * @note This function blindly sends out tuples with send(), but
 *       does not verify that sending actually succeeded.
 *
 * @see worker/worker.c:emit_result()
 *
 * @param ctx worker context (a partial message with batched-up
 *            tuples is held in the context, plus the FIFO to the
 *            master thread
 * @param r tuple position within input stream R
 * @param s tuple position within input stream S
 */
static void
emit_result (master_ctx_t *ctx, unsigned int r, unsigned int s)
{
    LOG(ctx->logfile, "result: r = %u, s = %u", r, s);

    assert (ctx->partial_result_msg.pos < RESULTS_PER_MESSAGE);

    ctx->partial_result_msg.msg[ctx->partial_result_msg.pos]
        = (result_t) { .r = r, .s = s };

    ctx->partial_result_msg.pos++;

    if (ctx->partial_result_msg.pos == RESULTS_PER_MESSAGE)
        flush_result (ctx);
}

/**
 * Flush queue to result collector; see emit_result().
 */
static inline void
flush_result (master_ctx_t *ctx)
{
    if (ctx->partial_result_msg.pos != 0)
    {
        LOG(ctx->logfile, "flushing result buffer (%u tuples)",
                ctx->partial_result_msg.pos);

        send (ctx->result_queue, &ctx->partial_result_msg.msg,
                ctx->partial_result_msg.pos * sizeof (result_t));

        ctx->partial_result_msg.pos = 0;
    }
    else
    {
        LOG(ctx->logfile, "flushing requested, but nothing to flush");
    }
}


/**
 * Serve the channel through which the GUI can send commands to the
 * driver process; read commands from the channel and set variables in the
 * master context accordingly.
 *
 * @note Support for the GUI front-end is not really maintained and
 *       likely broken.  Use at your own risk.
 */
static void *
read_control_input (void *arg)
{
    master_ctx_t  *ctx = (master_ctx_t *) arg;
    char           line[81];
    unsigned int   pos = 0;
    char           c;

    while (true)
    {
        c = getc (ctx->infile);

        if (c == EOF)
            break;

        if (c == '\n')
        {
            line[pos] = '\0';

            if (! strncmp (line, "driver_interval:",
                        sizeof ("driver_interval")))
            {
                unsigned int val = strtol (line + sizeof ("driver_interval"),
                        NULL, 10);
                fprintf (stderr, "setting driver_interval to %u.\n", val);
                fflush (stderr);
                ctx->driver_interval = val;
            }
            else if (! strncmp (line, "r_divider:",
                        sizeof ("r_divider")))
            {
                unsigned int val = strtol (line + sizeof ("r_divider"),
                        NULL, 10);
                fprintf (stderr, "setting r_divider to %u.\n", val);
                fflush (stderr);
                ctx->r_divider = val;
            }
            else if (! strncmp (line, "s_divider:",
                        sizeof ("s_divider")))
            {
                unsigned int val = strtol (line + sizeof ("s_divider"),
                        NULL, 10);
                fprintf (stderr, "setting s_divider to %u.\n", val);
                fflush (stderr);
                ctx->s_divider = val;
            }
            else if (! strncmp (line, "r_size:",
                        sizeof ("r_size")))
            {
                unsigned int val = strtol (line + sizeof ("r_size"),
                        NULL, 10);
                fprintf (stderr, "setting r_size to %u.\n", val);
                fflush (stderr);
                ctx->r_size = val;
            }
            else if (! strncmp (line, "s_size:",
                        sizeof ("s_size")))
            {
                unsigned int val = strtol (line + sizeof ("s_size"),
                        NULL, 10);
                fprintf (stderr, "setting s_size to %u.\n", val);
                fflush (stderr);
                ctx->s_size = val;
            }
            else if (! strncmp (line, "new R tuple",
                        sizeof ("new R tuple")-1))
            {
                bool ret;
                
                fprintf (stderr, "sending new R tuple");
                ret = send_new_R_tuple (ctx, 0, 0);
                fprintf (stderr, " (%s)\n", ret ? "successful" : "failed");
                fflush (stderr);
            }
            else if (! strncmp (line, "new S tuple",
                        sizeof ("new S tuple")-1))
            {
                bool ret;
                
                fprintf (stderr, "sending new S tuple");
                ret = send_new_S_tuple (ctx, 0, 0);
                fprintf (stderr, " (%s)\n", ret ? "successful" : "failed");
                fflush (stderr);
            }
            else
            {
                fprintf (stderr,
                        "WARNING: unrecognized command: `%s'.\n", line);
            }

            pos = 0;
            continue;
        }

        if (pos > 80)
        {
            fprintf (stderr,
                    "WARNING: Over-long input line.  Will ignore it.\n");
            pos = 0;
            continue;
        }

        line[pos++] = (char) c;

    }

    return NULL;
}

/**
 * Collect result tuples from all the workers.  This function is run
 * as its own thread.  It will spin and try to collect as many result
 * tuples from all the workers as it can.
 *
 * The function will serialize the result tuples to the output file
 * given on the command line (see master_ctx_t#outfile).  Also, the
 * function will report throughput statistics (i.e., result output
 * rates) approximately every second.
 *
 * @param arg a master context (signature of this function must match
 *            pthread_create(3).
 *
 * @return This function will never return.
 */
static void *
collect_results (void *arg)
{
    master_ctx_t    *ctx = (master_ctx_t *) arg;
    result_msg_t     msg;
    struct timespec  t_last;
    struct timespec  t_now;
    struct timespec  t_next;
    unsigned int     n_last = 0;
    unsigned int     n_now = 0;
    unsigned int     msg_size;
    unsigned int     num_result;

#ifdef HAVE_LIBNUMA
    numa_run_on_node (ctx->numa_nodes-1);
#endif

    hj_gettime (&t_last);

    t_next = t_last;

    while (true)
    {
        t_next.tv_nsec += COLLECT_INTERVAL;

        t_next.tv_sec  += t_next.tv_nsec / 1000000000L;
        t_next.tv_nsec %= 1000000000L;

        hj_nanosleep (&t_next);

        /* consume results from all result queues */
        for (unsigned int i = 0; i < ctx->num_workers+1; i++)
        {
            while (! empty (ctx->result_queues[i]))
            {
                msg_size = receive (ctx->result_queues[i], &msg);

                /*
                 * The message size tells us how many result tuples there
                 * are encoded in the message.
                 * See worker_ctx_t#partial_result_msg.
                 */
                num_result = msg_size / sizeof (result_t);

                for (unsigned int j = 0; j < num_result; j++)
                {
                    if (ctx->resultfile)
                    {
                        fprintf (ctx->resultfile,
                                "%4lu.%09lu | %8u | %8.2f || "//| %20s || "
                                "%4lu.%09lu | %8u | %8.2f || "//%10.2f | %5s || "
                                "%2u\n",
                                ctx->R.t[msg[j].r].tv_sec,
                                ctx->R.t[msg[j].r].tv_nsec,
                                ctx->R.x[msg[j].r], ctx->R.y[msg[j].r],
                                //ctx->R.z[msg[j].r],
                                ctx->S.t[msg[j].s].tv_sec,
                                ctx->S.t[msg[j].s].tv_nsec,
                                ctx->S.a[msg[j].s], ctx->S.b[msg[j].s],
                                //ctx->S.c[msg[j].s],
                                //ctx->S.d[msg[j].s] ? "true" : "false",
                                i);
                    }

                    n_now++;
                }
            }
        }

        /* check time stamp to report output data rate */
        hj_gettime (&t_now);

        /* report approximately every second */
        if (t_now.tv_sec != t_last.tv_sec)
        {
            const double sec
                = ((double) ((t_now.tv_sec * 1000000000L + t_now.tv_nsec)
                            - (t_last.tv_sec * 1000000000L + t_last.tv_nsec)))
                / 1e9;

            fprintf (stdout, "%u result tuples retrieved in %5.3f sec "
                    "(%6.2f tuples/sec)\n",
                    n_now - n_last, sec, ((double) (n_now - n_last)) / sec);

            if (ctx->resultfile && n_last != n_now)
                fflush (ctx->resultfile);

            t_last = t_now;
            n_last = n_now;
        }

    }

    /* never reached */
    return NULL;
}
