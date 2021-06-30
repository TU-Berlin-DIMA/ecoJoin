/**
 * @file
 *
 * Declarations for messages between processing cores (i.e., forwarded
 * tuples and acknowledgements).
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 Systems Group, ETH Zurich, Switzerland
 *
 * $Id: core2core.h 583 2010-08-02 06:52:52Z jteubner $
 */

#ifndef CORE2CORE_H
#define CORE2CORE_H

#include "config.h"
#include "parameter.h"
#include "stdint.h"

/**
 * Encode the type of a message.
 *
 * @note Should we ever get tight on space in messages, we could
 *       replace this by a @c char type (we'll than have to use
 *       macros instead of enum values then, however).
 */
enum core2core_msg_type_t {

    new_R_msg /**< tuple of stream R (sent to right, recvd from left) */
    ,
    new_S_msg /**< tuple of stream S (sent to left, recvd from right) */
    ,
    ack_R_msg /**< ack. of an R tuple (sent to left, recvd from right) */
    ,
    ack_S_msg /**< ack. of an S tuple (sent to right, recvd from left) */
    ,
    flush_msg /**< flush buffers with outgoing result tuples */

};
typedef enum core2core_msg_type_t core2core_msg_type_t;

/*
 * Message that describes a chunk of tuples from input stream R.
 */
struct chunk_R_msg_t {

    /**
     * Position of the first tuple of this chunk.
     * The receiver thread knows the base addresses of all column vectors
     * already (we'll want to communicate that information via messages,
     * too).  The size of each chunk is hard coded.
     */
    uint32_t start_idx;

    /** Number of tuples in this chunk. */
    uint32_t size;
};
typedef struct chunk_R_msg_t chunk_R_msg_t;

/**
 * Message that describes a chunk of tuples from input stream S.
 */
struct chunk_S_msg_t {

    /**
     * Position of the first tuple of this chunk.
     * The receiver thread knows the base addresses of all column vectors
     * already (we'll want to communicate that information via messages,
     * too).  The size of each chunk is hard coded.
     */
    uint32_t start_idx;

    /** Number of tuples in this chunk. */
    uint32_t size;
};
typedef struct chunk_S_msg_t chunk_S_msg_t;

/**
 * Message that describes the acknowledgement of a tuple chunk from
 * input stream R.
 */
struct ack_R_msg_t {

    /**
     * Position of the first tuple of this chunk, as in #chunk_R_msg_t.
     */
    uint32_t start_idx;

    /** Number of tuples in this chunk. */
    uint32_t size;
};
typedef struct ack_R_msg_t ack_R_msg_t;

/**
 * Message that describes the acknowledgement of a tuple chunk from
 * input stream S.
 */
struct ack_S_msg_t {

    /**
     * Position of the first tuple of this chunk, as in #chunk_S_msg_t.
     */
    uint32_t start_idx;

    /** Number of tuples in this chunk. */
    uint32_t size;
};
typedef struct ack_S_msg_t ack_S_msg_t;

/**
 * A message that is sent from one processing core to another.
 */
struct core2core_msg_t {

    /** The type of the message */
    core2core_msg_type_t type;

    /** The message itself */
    union {
        chunk_R_msg_t chunk_R;
        chunk_S_msg_t chunk_S;
        ack_R_msg_t ack_R;
        ack_S_msg_t ack_S;
    } msg;
};
typedef struct core2core_msg_t core2core_msg_t;

struct result_t {
    uint32_t r;
    uint32_t s;
};
typedef struct result_t result_t;

/**
 * Multiple result tuples can be packaged into a single message.
 * This value gives the number of tuples per message.
 */
#define RESULTS_PER_MESSAGE (MAX_MESSAGESIZE / sizeof(result_t))

/**
 * A single result message (encoding #RESULTS_PER_MESSAGE result tuples).
 */
typedef result_t result_msg_t[RESULTS_PER_MESSAGE];

#endif
