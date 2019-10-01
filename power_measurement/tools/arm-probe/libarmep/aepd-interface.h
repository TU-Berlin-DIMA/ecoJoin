/*
 * Author: Andy Green <andy.green@linaro.org> 
 * Copyright (C) 2012 Linaro, LTD
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 *
 * These are the interface structs between the acquisition-specific
 * code and the shared memory buffers, which are just abstracted
 * sample and channel descriptions with no knowledge of what
 * produced them
 */

#define AEPD_SHARED_MAX_REAL_CHANNELS (8 * 3)
#define AEPD_SHARED_MAX_VIRTUAL_CHANNELS (8)

/*
 * this represents one set of samples over all channels for the same time
 */

struct aepd_interface_result {
	int triggered; /* 0 = pretrigger 1 = triggered */
	int chans; /* number of channels below with data */
	double samtime; /* sample time in s */

	/* For each channel: { common-mode voltage in V, current in A } */
	double buf[AEPD_SHARED_MAX_REAL_CHANNELS * 2];
};

/*
 * Your code instantiates one of these by calling aepd_interface_create()
 * and then passes it to device-specific struct as well as monitoring it
 * for samples using aep_wait_for_next_result()
 */

struct aepd_interface {

	/* ringbuffer to hold channel_results until we can deal with them */

	int head;
	int tail;
	struct aepd_interface_result aepd_interface_result[1000];

	/* synchronization object signalling new result written */
	sem_t * semaphore;
	char semname[64];

	/* metadata about the channels */

	char channel_name[AEPD_SHARED_MAX_VIRTUAL_CHANNELS + AEPD_SHARED_MAX_REAL_CHANNELS][64];
	char channel_name_pretty[AEPD_SHARED_MAX_VIRTUAL_CHANNELS + AEPD_SHARED_MAX_REAL_CHANNELS][64];
	char supply[AEPD_SHARED_MAX_VIRTUAL_CHANNELS + AEPD_SHARED_MAX_REAL_CHANNELS][64];
	char colour[AEPD_SHARED_MAX_VIRTUAL_CHANNELS + AEPD_SHARED_MAX_REAL_CHANNELS][16];
	char class[AEPD_SHARED_MAX_VIRTUAL_CHANNELS + AEPD_SHARED_MAX_REAL_CHANNELS][16];
	int chans; /* number of active acquisition channels above */
	int vchans; /* virtual channels (summing supplies) appear after probed physical channels in the arrays */
	int finished; /* are we going down? */

	int auto_zero;
};

/*
 * generic sample management api
 */

extern struct aepd_interface *aepd_interface_create(void);
extern void aepd_interface_destroy(struct aepd_interface *aepd_interface);
extern struct aepd_interface_result * aep_wait_for_next_result(struct aepd_interface *aepd_interface);
extern void aep_free_result(struct aepd_interface *aepd_interface);

