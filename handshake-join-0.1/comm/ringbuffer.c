/**
 * @file
 *
 * Asynchronous message queue implementation.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) 2010, ETH Zurich, Systems Group
 *
 * $Id: ringbuffer.c 952 2011-03-09 12:16:34Z jteubner $
 */

#include "config.h"

#include "parameters.h"

#include "comm/ringbuffer.h"
#include "mem/mem.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

ringbuffer_t *
new_ringbuffer (unsigned int num_msgs, int node)
{
    ringbuffer_t *ret = alloc_onnode (sizeof (ringbuffer_t), node);
    int           status;
    void         *mem;

    assert (sizeof (message_t) == CACHELINESIZE);

    assert (sizeof (ringbuffer_t) == 3 * CACHELINESIZE);

    ret->writer_pos = 0;
    ret->reader_pos = 0;
    ret->num_msgs   = num_msgs;

    /*
     * Use posix_memalign(3), since this allows us to request
     * aligned memory.
     */
    status = posix_memalign (&mem, CACHELINESIZE,
                             num_msgs * sizeof (*(ret->messages)));
    /*
    status = posix_memalign ((void **) &(ret->messages), CACHELINESIZE,
                             num_msgs * sizeof (*(ret->messages)));
    */
    assert (status == 0);
    ret->messages = mem;

    memset (ret->messages, 0, num_msgs * sizeof (*(ret->messages)));

    return ret;
}

bool
send (ringbuffer_t *buf, const void *message, unsigned int msg_size)
{
    unsigned int attempts = SEND_TIMEOUT;

    assert (msg_size && "messages must not have length 0");
    assert (msg_size <= MAX_MESSAGESIZE);

    if (buf == NULL)
        return true;

    while (buf->messages[buf->writer_pos].size)
        /* spin until there's a slot available or we time out */
        if (!attempts--)
            return false;

    for (unsigned int i = 0; i < msg_size; i++)
        buf->messages[buf->writer_pos].message[i]
            = ((char *) message)[i];

    buf->messages[buf->writer_pos].size = msg_size;

    buf->writer_pos = (buf->writer_pos + 1) % buf->num_msgs;

    return true;
}

unsigned int
receive (ringbuffer_t *buf, void *message)
{
    unsigned int msg_size = buf->messages[buf->reader_pos].size;

    if (msg_size == 0)
        return 0;

    for (unsigned int i = 0; i < msg_size; i++)
        ((char *) message)[i] = buf->messages[buf->reader_pos].message[i];

    buf->messages[buf->reader_pos].size = 0;
    buf->reader_pos = (buf->reader_pos + 1) % buf->num_msgs;

    return msg_size;
}

unsigned int
peek (ringbuffer_t *buf, void **msg)
{
    *msg = (void *) buf->messages[buf->reader_pos].message;
    return buf->messages[buf->reader_pos].size;
}
