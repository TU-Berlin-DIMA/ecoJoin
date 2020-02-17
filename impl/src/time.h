/**
 * @file
 *
 * Wrapper for time-related stuff.  This is mainly because Mac OS X
 * does not support POSIX real-time functionality.  Here we provide
 * an interface that will support OS functionality as available.  Note
 * that for performance you will clearly want true real-time support
 * from the OS, that is, you want to run your code on Linux.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id: time.h 583 2010-08-02 06:52:52Z jteubner $
 */

#ifndef TIME_H
#define TIME_H


#include <time.h>
#include <sys/time.h>

void timespec_diff(struct timespec *start, struct timespec *stop,
                   struct timespec *result);

long timespec_to_ms(struct timespec *spec);

void print_timespec(struct timespec time);

#define hj_gettime(tm) \
    clock_gettime (CLOCK_REALTIME, (tm))

void hj_nanosleep (struct timespec *ts);


#endif  /* TIME_H */
