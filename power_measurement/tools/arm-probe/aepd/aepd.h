/*
 * Author: Andy Green <andy.green@linaro.org> 
 * Copyright (C) 2012 Linaro, LTD
 * Libwebsocket demo code (C) 2010-2012 Andy Green <andy@warmcat.com>
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include <sys/mman.h>

#include <libwebsockets.h>

#include "../libarmep/libarmep.h"

#define LOCAL_RESOURCE_PATH INSTALL_DATADIR"/aepd"
#define DOUBLES_PER_CH_SAMPLE 3

enum demo_protocols {
	/* always first */
	PROTOCOL_HTTP,

	PROTOCOL_AEPD,

	/* always last */
	DEMO_PROTOCOL_COUNT
};

struct aepd_shared {
	char fifo_filepath_stg[L_tmpnam + 1];
	char *fifo_filepath;
	int fd_fifo_write;
	int fd_fifo_read;
	int fifo_wrapped;
	int chans;
	unsigned long fifo_pos;
	double fifo_head_time;
	double fifo_tail_time;
	unsigned long modulo_integer_chan_size;
	int stop_flag;
	double trigger_level;
	int zero;
};

extern struct aepd_shared *aepd_shared;
