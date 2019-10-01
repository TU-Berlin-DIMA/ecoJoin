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
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <semaphore.h>
#include <sys/mman.h>

#include "aepd-interface.h"

struct aepd_interface *aepd_interface_create(void)
{
	struct aepd_interface *aepd_interface;
	int n;

	/*
	 * create our lump of shared memory
	 * which is usable by all forked threads
	 */

	n = open("/dev/zero", O_RDWR);
	aepd_interface = mmap(NULL, sizeof (struct aepd_interface),
		PROT_READ | PROT_WRITE, MAP_SHARED, n, 0);
	close(n);

	aepd_interface->head = 0;
	aepd_interface->tail = 0;

	sprintf(aepd_interface->semname, "linaro.aep.%u\n", getpid());

	aepd_interface->semaphore = sem_open(aepd_interface->semname, O_CREAT | O_RDWR, 0600, 0);
	if (aepd_interface->semaphore == SEM_FAILED) {
		fprintf(stderr, "Failed to open sem %s\n", aepd_interface->semname);
		return NULL;
	}

	return aepd_interface;
}

void aepd_interface_destroy(struct aepd_interface *aepd_interface)
{
	sem_close(aepd_interface->semaphore);
	if (munmap(aepd_interface, sizeof (struct aepd_interface)) < 0)
		fprintf(stderr, "munmap failed\n");
}

/*
 * helper for user code to block until next aep_result available
 */

struct aepd_interface_result * aep_wait_for_next_result(struct aepd_interface *aepd_interface)
{
	struct timespec ts;
	struct timeval tv;

	gettimeofday(&tv, NULL);
	ts.tv_sec = tv.tv_sec + 2;
	ts.tv_nsec = 0;

	if (sem_timedwait(aepd_interface->semaphore, &ts) < 0)
		return NULL;

	if (aepd_interface->tail == aepd_interface->head)
		return NULL;

	return &aepd_interface->aepd_interface_result[aepd_interface->tail];
}

/*
 * helper for user code to deal with ringbuffer
 */

void aep_free_result(struct aepd_interface *aepd_interface)
{
	if (aepd_interface->tail == sizeof(aepd_interface->aepd_interface_result) / sizeof(aepd_interface->aepd_interface_result[0]) - 1)
		aepd_interface->tail = 0;
	else
		aepd_interface->tail++;
}

