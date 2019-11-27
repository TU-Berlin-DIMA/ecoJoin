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

#define hj_gettime(tm) \
    clock_gettime (CLOCK_REALTIME, (tm))

//#define hj_nanosleep(ts) \
//    clock_nanosleep (CLOCK_REALTIME, TIMER_ABSTIME, (ts), NULL)



//int hj_gettime (struct timespec *ts);

void hj_nanosleep (struct timespec *ts);

int timespec2str(char *buf, unsigned int len, struct timespec *ts);

#endif  /* TIME_H */
