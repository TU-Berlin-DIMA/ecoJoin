/**
 * @file
 *
 * Our poor man's replacements for POSIX real-time functionality.
 * Mainly intended to make our code compile and run on Mac OS X
 * systems (@em not for productive use or experiments!).
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id: time.c 584 2010-08-02 08:00:20Z jteubner $
 */

#include "config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "string.h"
#include "time.h"

#include <iostream>
#include <unistd.h>

/*
 * Wait until timespec time is reached
 */
void hj_nanosleep(timespec* ts)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    long i = (ts->tv_sec * 1000000 + ts->tv_nsec / 1000) - (t.tv_sec * 1000000 + t.tv_usec);
    unsigned a = (ts->tv_sec * 1000000 + ts->tv_nsec / 1000) - (t.tv_sec * 1000000 + t.tv_usec);

    /*
    long s = (ts->tv_sec * 1000000 + ts->tv_nsec / 1000);
    long b = (t.tv_sec * 1000000 + t.tv_usec);
    std::cout << i << " " << s << " "<< b << "\n";
    */

    if (i > 0)
        usleep(a);
}

/*
 * Legacy implementation since busy waiting produces cpu overhead
 */
void hj_nanosleep_legacy(timespec* ts)
{
    struct timeval t;
    do {
        gettimeofday(&t, NULL);
    } while (t.tv_sec * 1000000000 + t.tv_usec * 1000
        < ts->tv_sec * 1000000000 + ts->tv_nsec);
}

void timespec_diff(struct timespec* start, struct timespec* stop,
    struct timespec* result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return;
}

long timespec_to_ms(struct timespec* spec)
{
    long ms; // Milliseconds
    time_t s; // Seconds

    s = spec->tv_sec;
    ms = round(spec->tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
    if (ms > 999) {
        s++;
        ms = 0;
    }
    return ms;
}

// buf needs to store 30 characters
int timespec2str(char* buf, unsigned int len, struct timespec* ts)
{
    int ret;
    struct tm t;

    tzset();
    if (localtime_r(&(ts->tv_sec), &t) == NULL)
        return 1;

    ret = strftime(buf, len, "%F %T", &t);
    if (ret == 0)
        return 2;
    len -= ret - 1;

    ret = snprintf(&buf[strlen(buf)], len, ".%09ld", ts->tv_nsec);
    if (ret >= len)
        return 3;

    return 0;
}

void print_timespec(struct timespec time)
{
    const uint TIME_FMT = strlen("2012-12-31 12:59:59.123456789") + 1;
    char timestr[TIME_FMT];

    struct timeval t;

    if (timespec2str(timestr, sizeof(timestr), &time) != 0) {
        printf("timespec2str failed!\n");
    } else {
        printf("CLOCK_REALTIME: time=%s\n", timestr);
    }
}
