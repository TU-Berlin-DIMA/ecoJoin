/**
 * @file
 *
 * Declarations for FIFO queue (ringbuffer) implementation.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010 ETH Zurich, Systems Group
 *
 * $Id: ringbuffer.h 590 2010-08-11 09:07:39Z jteubner $
 */

#include "config.h"

#include "parameter.h"

#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <stdbool.h>
#include <stdint.h>

/**
 * We use one byte for status information.  Hence, messages may be
 * at most (#CACHELINESIZE - 1) bytes long.
 */
#define MAX_MESSAGESIZE ((CACHELINESIZE)-1)

/**
 * We store one message in one cache line.  The last byte is used
 * as a status/size byte.  The consumer will set the status byte to 0
 * after reading the message (and thus indicate that the slot is
 * free).  The producer will write the length of the message (in
 * bytes) into the status/size byte.  Note that messages must be at
 * least one byte in size.  #size is always written as the last
 * action when sending or receiving messages.
 *
 * Message and status byte are both marked as `volatile.'  Otherwise,
 * the C compiler would pull accesses out of the (spin) loops and the
 * spin loops would never terminate.
 *
 * By use of the `packed' attribute, we make sure that the compiler
 * actually packs the message together into one cache line.
 */
struct message_t {
    volatile uint8_t message[MAX_MESSAGESIZE];
    volatile uint8_t size;
} __attribute__((packed, aligned(CACHELINESIZE)));
typedef struct message_t message_t;

/**
 * A logical ring buffer, shared by one producer and one consumer.
 *
 * Producer and consumer each keep track of their current position
 * in the ring buffer using the #writer_pos and #reader_pos fields.
 * Since accesses to both fields will be highly concurrent, we make
 * sure that they are placed in different cache lines.
 */
struct ringbuffer_t {
    unsigned int num_msgs; /**< number of slots in this ring buffer */
    message_t* messages; /**< the actual messages */
    unsigned int writer_pos __attribute__((aligned(CACHELINESIZE)));
    /**< next slot that the producer will write to */
    unsigned int reader_pos __attribute__((aligned(CACHELINESIZE)));
    /**< next slot that the consumer will (try to) read */
};
typedef struct ringbuffer_t ringbuffer_t;

/**
 * Create a new ringbuffer.
 *
 * Allocates memory for a ringbuffer_t structure, as well as for
 * the respective messages array.
 */
ringbuffer_t* new_ringbuffer(unsigned int num_msgs, int node);

/**
 * Send a message.
 *
 * May have to block until there is a slot available in the ring buffer.
 * Blocking times out after we tried #SEND_TIMEOUT times to send the
 * data.
 *
 * @param buf      FIFO buffer to send the message to; if @a buf is @c NULL,
 *                 function will immediately return @c true and no data will
 *                 be sent anywhere (sort of like a /dev/null-style data sink)
 * @param message  pointer to the actual message
 * @param msg_size size of the message; starting from @a message,
 *                 @a msg_size bytes will be sent
 *
 * @return returns @c true if the data could successfully be sent (or
 *         if @a buf was @c NULL); @c false indicates that the FIFO queue
 *         was full
 */
bool send(ringbuffer_t* buf, const void* message, unsigned int msg_size);

/**
 * Return true if the queue is full, a send() to the queue might block.
 */
#define full(ringbuffer)  \
    ((ringbuffer) != NULL \
        && (ringbuffer)->messages[ringbuffer->writer_pos].size != 0)

/**
 * Return true if no messages are waiting in the buffer, otherwise false.
 */
#define empty_(ringbuffer) \
    ((ringbuffer)->messages[ringbuffer->reader_pos].size == 0)

/**
 * Receive a message (non-blocking).
 *
 * Writes received message into memory region pointed to by #message,
 * returns message size (in bytes).  Returns 0 if there was no message
 * in the ring buffer.
 */
unsigned int receive(ringbuffer_t* buf, void* message);

/**
 * Peek at a message, without actually receiving it.
 *
 * Returns the size of the stored message (0 if there is no message
 * waiting in the ring buffer).  Writes a pointer to the message into
 * the address given by @a msg.
 */
unsigned int peek(ringbuffer_t* buf, void** msg);

#endif /* RINGBUFFER_H */
