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

#include <chrono>
#include <sys/time.h>
#include <time.h>

void timespec_diff(struct timespec* start, struct timespec* stop,
    struct timespec* result);

long timespec_to_ms(struct timespec* spec);

void print_timespec(struct timespec time);

#define hj_gettime(tm) \
    clock_gettime(CLOCK_REALTIME, (tm))

void hj_nanosleep(struct timespec* ts);

// We want to use clock_gettime()
// If it doesn't work, try setting CPP_FLAGS "--enable-libstdcxx-time=yes"
// This enables link-time checking for clock_gettime() in GCC/G++
// Checks taken from https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63400#c1
#ifdef _GLIBCXX_USE_CLOCK_REALTIME
#ifdef _GLIBCXX_USE_CLOCK_GETTIME_SYSCALL
#warning system_clock using syscall(SYS_clock_gettime, CLOCK_REALTIME, &tp);
#else
//#warning system_clock using clock_gettime(CLOCK_REALTIME, &tp);
#endif
#elif defined(_GLIBCXX_USE_GETTIMEOFDAY)
#warning system_clock using gettimeofday(&tv, 0);
#else
#warning system_clock using std::time(0);
#endif

#ifdef _GLIBCXX_USE_CLOCK_MONOTONIC
#ifdef _GLIBCXX_USE_CLOCK_GETTIME_SYSCALL
#warning steady_clock using syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &tp);
#else
//#warning steady_clock using clock_gettime(CLOCK_MONOTONIC, &tp);
#endif
#else
#warning steady_clock using time_point(system_clock::now().time_since_epoch());
#endif

namespace Timer {

// Timer for benchmarking
// Header-only implementation for compiler inlining
// Verified that G++ with -O2 optimization inlines these functions
class Timer {
public:
    void start()
    {
        start_epoch_ = std::chrono::steady_clock::now();
    }

    template <typename UnitT = std::chrono::nanoseconds>
    uint64_t stop()
    {
        std::chrono::steady_clock::time_point stop_epoch;
        UnitT time_span;

        stop_epoch = std::chrono::steady_clock::now();
        time_span = std::chrono::duration_cast<UnitT>(stop_epoch - start_epoch_);
        return time_span.count();
    }

    std::chrono::steady_clock::time_point now()
    {
        return std::chrono::steady_clock::now();
    }

private:
    std::chrono::steady_clock::time_point start_epoch_;
};

}

#endif /* TIME_H */
