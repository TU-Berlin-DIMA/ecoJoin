/**
 * @file
 *
 * Debugging support
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id: debug.h 587 2010-08-02 12:11:28Z jteubner $
 */

#ifndef DEBUG_H
#define DEBUG_H

#include "config.h"

#ifdef HAVE_LIBRT
#include <time.h>
#else
#include <sys/time.h>
#endif

#ifndef NDEBUG

#ifdef HAVE_LIBRT

/**
 * For debugging purposes: log to a file what this worker is doing.
 * Will do nothing if NDEBUG is defined.
 *
 * @param f   a file handle
 * @param ... a printf() style argument list (format file plus arguments)
 */
#define LOG(f, ...) \
    do {                                                                      \
        if (f)                                                                \
        {                                                                     \
            char logmsg[128];                                                 \
            struct timespec t;                                                \
                                                                              \
            clock_gettime (CLOCK_REALTIME, &t);                               \
            snprintf (logmsg, sizeof (logmsg), __VA_ARGS__);                  \
                                                                              \
            fprintf (f, "%lu.%09lu: %s\n", t.tv_sec + t.tv_nsec / 1000000000, \
                    t.tv_nsec, logmsg);                                       \
            fflush (f);                                                       \
        }                                                                     \
    } while (0)
#else  /* HAVE_LIBRT? */
#define LOG(f, ...) \
    do {                                                                      \
        if (f)                                                                \
        {                                                                     \
            char logmsg[128];                                                 \
            struct timeval t;                                                 \
                                                                              \
            gettimeofday (&t, NULL);                                          \
            snprintf (logmsg, sizeof (logmsg), __VA_ARGS__);                  \
                                                                              \
            fprintf (f, "%li.%06i: %s\n", t.tv_sec + t.tv_usec / 1000000,     \
                    (int) t.tv_usec, logmsg);                                 \
            fflush (f);                                                       \
        }                                                                     \
    } while (0)
#endif  /* HAVE_LIBRT? */

#else  /* NDEBUG? */

#define LOG(f, ...)

#endif  /* NDEBUG? */

#endif  /* DEBUG_H */
