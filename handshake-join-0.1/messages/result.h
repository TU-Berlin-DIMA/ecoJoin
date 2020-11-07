/**
 * @file
 *
 * Declarations for messages to send results back to master.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 Systems Group, ETH Zurich, Switzerland
 *
 * $Id: result.h 583 2010-08-02 06:52:52Z jteubner $
 */

#ifndef RESULT_H
#define RESULT_H

#include "config.h"
#include "parameters.h"

#include "comm/ringbuffer.h"

/**
 * Message that encodes a single result tuple (by stating the rows
 * numbers in input stream R and S).
 */
struct result_t {
    uint32_t    r;
    uint32_t    s;
};
typedef struct result_t result_t;

/**
 * Multiple result tuples can be packaged into a single message.
 * This value gives the number of tuples per message.
 */
#define RESULTS_PER_MESSAGE (MAX_MESSAGESIZE / sizeof (result_t))

/**
 * A single result message (encoding #RESULTS_PER_MESSAGE result tuples).
 */
typedef result_t result_msg_t[RESULTS_PER_MESSAGE];

#endif  /* RESULT_H */
