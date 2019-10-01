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

#include "aepd.h"

unsigned long max_fifo_extent_seconds = 120;

#define MAX_FIFO_EXTENT (MAX_PROBES * CHANNELS_PER_PROBE * sizeof(double) * \
			DOUBLES_PER_CH_SAMPLE * 10000 * max_fifo_extent_seconds)

struct aepd_shared *aepd_shared;
struct aepd_interface *aepd_interface;
static double sam[MAX_PROBES * CHANNELS_PER_PROBE * DOUBLES_PER_CH_SAMPLE];
extern struct lws_protocols protocols[];

void zero_fifo(void)
{
	int n;

	lseek(aepd_shared->fd_fifo_write, 0, SEEK_SET);
	aepd_shared->fifo_pos = 0;
	aepd_shared->fifo_wrapped = 0;
	for (n = 0; n < sizeof(sam) / sizeof(sam[0]); n++)
		sam[n] = 0.0;
	if (aepd_shared->chans > 0)
		aepd_shared->modulo_integer_chan_size =
			(MAX_FIFO_EXTENT / (aepd_shared->chans * sizeof(double) * 3)) *
			(aepd_shared->chans * sizeof(double) * 3);
	aepd_shared->fifo_head_time = 0;
	aepd_shared->fifo_tail_time = 0;
}

int add_fifo(void *data, unsigned int length)
{
	long l;

	l = write(aepd_shared->fd_fifo_write, data, length);
	if (l < length) {
		fprintf(stderr, "fifo write failed...\n");
		return (int)l;
	}

	aepd_shared->fifo_pos += length;
	if (aepd_shared->fifo_pos >= aepd_shared->modulo_integer_chan_size) {
		aepd_shared->fifo_wrapped = 1;
		aepd_shared->fifo_pos = 0;
		lseek(aepd_shared->fd_fifo_write, 0, SEEK_SET);
	}

	return 0;
}

struct aep_context aep_context = {
	.config_filepath = "./config",
	.highest = -1,
	.decimate = 1,
	.mv_min = 400,
	.trigger_filter_us = 400,
	.end_trigger_filter_us = 500000,
	.average_len = 1,
	.configuration_name = "default_configuration",
	.verbose = 0,
};


static struct option options[] = {
	{ "help",	no_argument,		NULL, 'h' },
	{ "port",	required_argument,	NULL, 'p' },
	{ "buffer",	required_argument,	NULL, 'b' },
	{ "ssl",	no_argument,		NULL, 's' },
	{ "interface",  required_argument,	NULL, 'i' },
	{ NULL, 0, 0, 0 }
};

static int loop = 1;

void sighandler(int sig)
{
	loop = 0;
}

static struct lws_context_creation_info info;

int main(int argc, char *argv[])
{
	int n = 0, m;
	const char *cert_path =
			    LOCAL_RESOURCE_PATH"/aepd.pem";
	const char *key_path =
			LOCAL_RESOURCE_PATH"/aepd.key.pem";
	int port = 15164;
	int use_ssl = 0;
	struct lws_context *context;
	int opts = LWS_SERVER_OPTION_SKIP_SERVER_CANONICAL_NAME;
	char interface_name[128] = "";
	const char *interface_ptr = NULL;
	struct aepd_interface_result *aepd_interface_result;
	struct timeval tv;
	unsigned long last = 0;
	unsigned long ms10 = -1;

	fprintf(stderr,
		"ARM Energy Probe Daemon  (C) Copyright 2012-2013 Linaro, LTD\n"
		"licensed under LGPL2.1\n");

	signal(SIGINT, sighandler);

	/*
	 * create our lump of shared memory
	 * which is usable by all forked processes
	 */

	n = open("/dev/zero", O_RDWR);
	aepd_shared = mmap(NULL, sizeof (struct aepd_shared),
		PROT_READ | PROT_WRITE, MAP_SHARED, n, 0);
	close(n);

	/*
	 * open spool fifo for result data we can serve
	 * it's a large ring buffer to deal with long term pretrigger
	 * websocket clients can make requests for variously-zoomed
	 * parts of this buffer.
	 */

	aepd_shared->fifo_filepath = tmpnam(aepd_shared->fifo_filepath_stg);
	aepd_shared->fd_fifo_write = open(aepd_shared->fifo_filepath,
						       O_CREAT | O_RDWR, 0600);
	if (aepd_shared->fd_fifo_write < 0) {
		fprintf(stderr, "Unable to open sample fifo file %s\n",
						    aepd_shared->fifo_filepath);
		return -1;
	}
	aepd_shared->fd_fifo_read = open(aepd_shared->fifo_filepath, O_RDONLY);
	if (aepd_shared->fd_fifo_read < 0) {
		fprintf(stderr, "Unable to open sample fifo file %s for read\n",
						    aepd_shared->fifo_filepath);
		return -1;
	}

	zero_fifo();

	while (n >= 0) {
		n = getopt_long(argc, argv, "b:si:p:", options, NULL);
		if (n < 0)
			continue;
		switch (n) {

		case 'b':
			max_fifo_extent_seconds = atoi(optarg);
			break;

		case 's':
			use_ssl = 1;
			break;
		case 'p':
			port = atoi(optarg);
			break;
		case 'i':
			strncpy(interface_name, optarg, sizeof interface_name);
			interface_name[(sizeof interface_name) - 1] = '\0';
			break;

		default:
		case 'h':
			fprintf(stderr, "Usage: arm-probe \n"
			    " [--buffer -b <secs>] duration of capture memory (5MB/sec!) default 120\n"
			    " [--ssl -s] Listen using SSL / Encrypted link\n"
			    " [--interface -i <if>] Listen on specific "
					"interface, eg, eth1\n"
			    " [--port -p -<port>] Port to listen on "
					"(default %u)\n", port
			);
			exit(1);
		}
	}

	/*
	 * generic sample interface
	 */

	aepd_interface = aepd_interface_create();
	if (aepd_interface == NULL) {
		fprintf(stderr, "failed to create aepd_interface region\n");
		return -1;
	}
	aep_context.aepd_interface = aepd_interface;

	/* libwebsockets context */

	if (!use_ssl)
		cert_path = key_path = NULL;

	info.port = port;
	info.iface = interface_ptr;
	info.protocols = protocols;
	info.ssl_cert_filepath = cert_path;
	info.ssl_private_key_filepath = key_path;
	info.uid = -1;
	info.gid = -1;
	info.options = opts;

	context = lws_create_context(&info);
	if (context == NULL) {
		fprintf(stderr, "libwebsocket init failed\n");
		return -1;
	}

	aep_context.original_count_channel_names = aep_context.count_channel_names;

	configure(&aep_context, NULL, "/virtual", aep_context.config_filepath, NULL);

	/*
	 * fork off the AEP service process
	 * runs in its own process to exploit SMP to dedicate one core for that
	 * what happens to samples is decoupled from capture process with large
	 * shared-memory buffer to allow for jitter
	 */

	if (aep_init_and_fork(&aep_context, argv) < 1)
		return 0; /* child process exit */

	
	n = fork();
	if (n < 0) {
		fprintf(stderr, "websockets service fork failed\n");
		return n;
	}
	if (!n) {
		strcpy(argv[0] + strlen(argv[0]), " - websockets server");

		/* websockets service process */

		while (1) {
			if (lws_service(context, 100))
				break;
			if (getppid() == 1)
				break;
			
			gettimeofday(&tv, NULL);

			ms10 = (tv.tv_sec * 10) + (tv.tv_usec / 100000);

			if (ms10 > last) {
				last = ms10;
				lws_callback_on_writable_all_protocol(context, &protocols[1]);
			}
		}

		lws_context_destroy(context);
		return 0;
	}

	/*
	 * this process just has to deal with collecting samples from
	 * any backend that acquired them into the shared memory
	 * buffer, and then make them available to the websocket
	 * callback via the aepd private fifo ring file
	 */

	aepd_shared->chans = -1;

	while (loop) {

		aepd_interface_result = aep_wait_for_next_result(aep_context.aepd_interface);
		if (!aepd_interface_result) {
			if (aepd_interface->finished)
				loop = 0;

			continue;
		}

		if (aepd_shared->stop_flag)
			goto done;

		if (aepd_shared->zero) {
			fprintf(stderr, "setting az in libaep thread\n");
			aep_context.aepd_interface->auto_zero = 2;
			aepd_shared->zero = 0;
		}

		/*
		 * we have the next result in aepd_interface_result..
		 * voltage and current per channel
		 */

		m = 0;
		for (n = 0; n < aepd_interface_result->chans * 2; n += 2) {

			/* 
			 * accumulate in V, A, W per-channel.
			 * We do summing like that so that we can compute
			 * averages over any distance without iteration, by
			 * (sample(x + distance) - sample(x)) / distance.
			 * We can recover individual intersample delta still
			 * by sample(x + 1) - sample(x) if we want it.
			 * Doubles are used to maximize dynamic range.
			 */

			sam[m++] += aepd_interface_result->buf[n];
			sam[m++] += aepd_interface_result->buf[n + 1];
			sam[m++] += aepd_interface_result->buf[n] * aepd_interface_result->buf[n + 1];
		}

		/*
		 * notice we have expanded the V/A 2 sample data
		 * into W as well.  That's because average of instantaneous W
		 * is not at all the same as average V * average A
		 */

		add_fifo(&sam[0], aepd_interface_result->chans * 3 * sizeof(double));
		aepd_shared->fifo_head_time = aepd_interface_result->samtime;
		if ((aepd_shared->fifo_head_time - aepd_shared->fifo_tail_time) > max_fifo_extent_seconds)
			aepd_shared->fifo_tail_time = aepd_shared->fifo_head_time - max_fifo_extent_seconds;
			

		if (aepd_shared->chans != aepd_interface_result->chans) {
			aepd_shared->chans = aepd_interface_result->chans;
			zero_fifo();
		}
done:
		/* done with it */

		aep_free_result(aepd_interface);
	}

	close(aepd_shared->fd_fifo_write);
	close(aepd_shared->fd_fifo_read);
	unlink(aepd_shared->fifo_filepath);

	aepd_interface_destroy(aepd_interface);

	fprintf(stderr, "exited\n");
	
	return 0;
}

