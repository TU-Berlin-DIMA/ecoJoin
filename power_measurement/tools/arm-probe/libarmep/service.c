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

#include "libarmep.h"

#include <linux/serial.h>


void add_pollfd(struct aep_context *aep_context, int fd, int events)
{
	aep_context->pollfds[aep_context->count_pollfds].fd = fd;
	aep_context->pollfds[aep_context->count_pollfds].events = events;
	aep_context->pollfds[aep_context->count_pollfds++].revents = 0;
}

void remove_pollfd(struct aep_context *aep_context, int fd)
{
	int n;

	for (n = 0; n < aep_context->count_pollfds; n++)
		if (aep_context->pollfds[n].fd == fd) {
			while (n < aep_context->count_pollfds) {
				aep_context->pollfds[n] = aep_context->pollfds[n + 1];
				n++;
			}
			aep_context->count_pollfds--;
		}
}

static void init_aep(struct aep_context *aep_context, struct aep *aep, const char *device_filepath)
{
	int n, m;
	struct aep_channel *ch;

	aep->head = 0;
	aep->tail = 0;
	aep->predicted_frame = 0;
	aep->state = APP_INIT_MAGIC;
	aep->invalid = 0;
	aep->counter = 0;
	aep->started = 0;
	aep->done_config = 0;

	aep->aep_context = aep_context;

	strncpy(aep->dev_filepath, device_filepath,
						sizeof aep->dev_filepath - 1);
	aep->dev_filepath[sizeof aep->dev_filepath - 1] = '\0';

	for (n = 0; n < CHANNELS_PER_PROBE; n++) {
		ch = &aep->ch[n];	

		ch->ignore = 0;
		ch->triggered = 0;
		ch->requested = 0;
		ch->summary[0] = '\0';
		ch->aep = aep;
		ch->trigger_slave = NULL;
		ch->out_head = ch->out_tail = 0;
		ch->voffset[0] = 0;
		ch->vnoise[0] = 0;
		ch->voffset[1] = 0;
		ch->vnoise[1] = 0;
		ch->rshunt = 0;
		ch->flag_was_configured = 0;
		sprintf(ch->channel_name, "%s-%d", device_filepath, n);
		ch->pretrig_ring = NULL;
		ch->ring_samples = 0;
		ch->trigger_filter = 0;
		ch->pretrigger_samples_taken = 0;
		avg_mean_us_init(&ch->avg_mean_voltage, aep->aep_context->average_len);
		avg_mean_us_init(&ch->avg_mean_current, aep->aep_context->average_len);
		ch->decimation_counter = 0;
		for (m = 0; m < sizeof(ch->min) / sizeof(ch->min[0]); m++) {
			ch->min[m] = 999;
			ch->max[m] = 0;
		}
		ch->samples_seen = 0;
		ch->simple_avg[0] = 0.0;
		ch->simple_avg[1] = 0.0;
		ch->simple_avg[2] = 0.0;
		ch->avg_count = 0;
		ch->samples = 0;
		strcpy(ch->supply, "none");
		sprintf(ch->colour, "#%06X", rand() & 0xffffff);
		strcpy(ch->class, "-");
	}
}


static void select_map(struct aep_channel *ch)
{
	int sel = -1;

	switch(ch->channel_num) {
	case 1:
		sel = 0;
		break;
	case 2:
		sel = 1;
		break;
	case 3:
		sel = 2;
		break;
	default:
		fprintf(stderr, "AEP channel number %d is wrong. It's only 3 channel, 1, 2 or 3\n",
			ch->channel_num);
	}

	if (ch->aep->aep_context->verbose)
		fprintf(stderr, "%s = corr map %d  %d\n",
			ch->channel_name, sel, ch->channel_num);
	ch->map_table = &interp_tables[sel];
}

void probe_close(struct aep *aep)
{
	unsigned char c = AEPC_STOP;
	int n;

	if (write(aep->fd, &c, 1) < 0)
		fprintf(stderr, "Couldn't ask probe to stop during close\n"); /* we don't care, we are closing */

	for (n = 0; n < CHANNELS_PER_PROBE; n++) {
		if (aep->ch[n].pretrig_ring)
			free(aep->ch[n].pretrig_ring);

		avg_mean_us_free(&aep->ch[n].avg_mean_voltage);
		avg_mean_us_free(&aep->ch[n].avg_mean_current);
	}

	if (aep->aep_context->verbose)
		fprintf(stderr, "closing %d\n", aep->fd);
	close(aep->fd);
}

static void copy_public_ch_info_to_shared(struct aepd_interface *aepd_interface, int chan, struct aep_channel *ch)
{
	strncpy(aepd_interface->channel_name[chan], ch->channel_name, sizeof(aepd_interface->channel_name[0]));
	aepd_interface->channel_name[chan][sizeof(aepd_interface->channel_name[0]) - 1] = '\0';
	strncpy(aepd_interface->channel_name_pretty[chan], ch->channel_name_pretty, sizeof(aepd_interface->channel_name_pretty[0]));
	aepd_interface->channel_name_pretty[chan][sizeof(aepd_interface->channel_name_pretty[0]) - 1] = '\0';
	strncpy(aepd_interface->supply[chan], ch->supply, sizeof(aepd_interface->supply[0]));
	aepd_interface->supply[chan][sizeof(aepd_interface->supply[0]) - 1] = '\0';
	strncpy(aepd_interface->colour[chan], ch->colour, sizeof(aepd_interface->colour[0]));
	aepd_interface->colour[chan][sizeof(aepd_interface->colour[0]) - 1] = '\0';
	strncpy(aepd_interface->class[chan], ch->class, sizeof(aepd_interface->class[0]));
	aepd_interface->class[chan][sizeof(aepd_interface->class[0]) - 1] = '\0';

}

int service_aeps(struct aep_context *aep_context, int fd_with_rx)
{
	static unsigned char zero[8] = { 0, 0, 0, 0, 0, 0, 0, 0, };
	int fd;
	int budget;
	int len;
	struct timeval tv;
	unsigned char *p;
	unsigned char c;
	struct termios tty;
	struct serial_struct sinfo;
	int n, m, i, chan;
	struct aep *aep;
	struct aep_channel *ch;
	unsigned char buf[AEP_INPUT_QUEUE];

	if (fd_with_rx >= 0)
		goto post_start;

	gettimeofday(&tv, NULL);

	if (tv.tv_sec == aep_context->last_scan_sec)
		goto bail;

	aep_context->last_scan_sec = tv.tv_sec;

	/* look for next device appearing */

	if (!aep_context->device_paths[aep_context->scan][0])
		goto bail;

	if (aep_context->aeps[aep_context->scan].fd > 0)
		goto bail;

	fd = open(aep_context->device_paths[aep_context->scan], O_RDWR | O_NONBLOCK | O_EXCL | O_NOCTTY);

	if (fd < 0)
		goto bail;

	if (ioctl(fd, TIOCEXCL) < 0) {
		fprintf(stderr, "Wasn't able to open %s exclusive",
				aep_context->device_paths[aep_context->scan]);
		close(fd);
		goto bail;
	}

	fprintf(stderr, "+ %s\n", aep_context->device_paths[aep_context->scan]);

	tcflush(fd, TCIOFLUSH);

	if (ioctl(fd, TIOCGSERIAL, &sinfo) == 0) {
		sinfo.closing_wait = ASYNC_CLOSING_WAIT_NONE;
		ioctl(fd, TIOCSSERIAL, &sinfo);
	}

	if (aep_context->verbose)
		fprintf(stderr, "initing %s fd=%d\n", aep_context->device_paths[aep_context->scan], fd);

	/* enforce suitable tty state */

	memset (&tty, 0, sizeof tty);
	if (tcgetattr (fd, &tty)) {
		fprintf(stderr, "tcgetattr failed on %s\n", aep_context->device_paths[aep_context->scan]);
		close(fd);
		goto bail;
	}

	tty.c_lflag &= ~(ISIG | ICANON | IEXTEN | ECHO | XCASE |
			ECHOE | ECHOK | ECHONL | ECHOCTL | ECHOKE);
	tty.c_iflag &= ~(INLCR | IGNBRK | IGNPAR | IGNCR | ICRNL |
			   IMAXBEL | IXON | IXOFF | IXANY | IUCLC);
	tty.c_oflag &= ~(ONLCR | OPOST | OLCUC | OCRNL | ONLRET);
	tty.c_cc[VMIN]  = 1;
	tty.c_cc[VTIME] = 0;
	tty.c_cc[VEOF] = 1;
	tty.c_cflag &=  ~(CBAUD | CSIZE | CSTOPB | PARENB | CRTSCTS);
	tty.c_cflag |= (8 | CREAD | 0 | 0 | 1 | CLOCAL);

	cfsetispeed(&tty, B115200);
	cfsetospeed(&tty, B115200);
	tcsetattr(fd, TCSANOW, &tty);

	init_aep(aep_context, &aep_context->aeps[aep_context->scan], aep_context->device_paths[aep_context->scan]);

	if (configure(aep_context, &aep_context->aeps[aep_context->scan], aep_context->device_paths[aep_context->scan],
					  aep_context->config_filepath, NULL) < 0) {
		fprintf(stderr, "config for %s failed\n", aep_context->device_paths[aep_context->scan]);
		close(fd);
		goto bail;
	}

	if (write(fd, zero, sizeof(zero)) != sizeof(zero)) {
		close(fd);
		goto bail;
	}
	if (aep_context->verbose)
		fprintf(stderr, "sending reset\n");
	c = AEPC_RESET;
	if (write(fd, &c, 1) != 1) {
		close(fd);
		goto bail;
	}

	if (aep_context->verbose)
		fprintf(stderr, "done reset\n");

	aep_context->aeps[aep_context->scan].fd = fd;
	aep_context->aeps[aep_context->scan].sec_last_traffic = tv.tv_sec;
	aep_context->aeps[aep_context->scan].index = aep_context->scan;

	if (aep_context->scan > aep_context->highest)
		aep_context->highest = aep_context->scan;

	add_pollfd(aep_context, fd, POLLIN | POLLERR);

	for (n = 0; n < CHANNELS_PER_PROBE; n++) {
		struct aep_channel *ch = &aep_context->aeps[aep_context->scan].ch[n];

		if (!ch->flag_was_configured)
			continue;

		select_map(ch);

		if (aep_context->original_count_channel_names) {
			for (m = 0; m < aep_context->count_channel_names; m++)
				if (!strcmp(ch->channel_name, aep_context->channel_name[m]))
					break;

			if (m == aep_context->count_channel_names)
				continue;
		} else
			if (aep_context->count_channel_names < (aep_context->matched_channel_names + 1))
				aep_context->count_channel_names++;

		ch->requested = 1;
		aep_context->matched_channel_names++;

		aep_context->awaiting_capture |= 7 << (aep_context->scan * 3);
	}

	aep_context->aeps[aep_context->scan].done_config |= 1;
	aep_context->scan++;

bail:
	aep_context->scans++;

	/* if we're waiting to start but saw every channel / probe */

	if (!aep_context->has_started &&
			aep_context->count_channel_names == aep_context->matched_channel_names) {
		/* nobody is waiting for config completion? */

		c = 0;
		for (m = 0; m <= aep_context->highest; m++)
			if (aep_context->aeps[m].fd > 0) {
				c = 1;
				if (aep_context->aeps[m].done_config != 3 || aep_context->aeps[m].invalid)
					goto post_start;
			}

		if (c == 0)
			goto post_start;

		/* 
		 * in the case we want to start all probes, don't know how many there are
		 * allow time to discover and recover from any initial comms error
		 */

		if (aep_context->scans < (MAX_PROBES * 2))
			goto post_start;

		/* 
		 * the number of guys we would start
		 * is all of them, is it?  Becuase we need to
		 * eliminate cross-channel latency as far as we
		 * can
		 */

		c = 0;
		for (m = 0; m <= aep_context->highest; m++)
			if (aep_context->aeps[m].fd > 0 && aep_context->aeps[m].done_config == 3) {
				for (i = 0; i < CHANNELS_PER_PROBE; i++)
					if (aep_context->aeps[m].ch[i].requested &&
						aep_context->aeps[m].ch[i].flag_was_configured)
						c++;
			}

		if (c != aep_context->matched_channel_names)
			goto post_start;

		/* still something expected? */
		if (aep_context->device_paths[aep_context->scan][0])
			goto post_start;

		/* tell everyone to start */

		fprintf(stderr, "Starting...\n");

		chan = 0;
		for (m = 0; m <= aep_context->highest; m++) {
			if (aep_context->aeps[m].fd <= 0 ||
					aep_context->aeps[m].done_config != 3 ) {
				if (aep_context->verbose)
					fprintf(stderr, "not starting %d fd %d done_config %d\n", m, aep_context->aeps[m].fd, aep_context->aeps[m].done_config);
				continue;
			}

//			if (aep_context->verbose)
				fprintf(stderr, "sending start to %d\n", m);
			c = AEPC_START;
			if (write(aep_context->aeps[m].fd, &c, 1) != 1) {
				fprintf(stderr,
				     "Failed to send start\n");
			}

			for (i = 0; i < CHANNELS_PER_PROBE; i++) {
				ch = &aep_context->aeps[m].ch[i];

				if (!ch->requested || !ch->flag_was_configured)
					continue;

				copy_public_ch_info_to_shared(aep_context->aepd_interface, chan, ch);

				chan++;
				aep_context->aepd_interface->chans = chan;
			}

			aep_context->aeps[m].started = 1;
			aep_context->has_started = 1;
			aep_context->aeps[m].sec_last_traffic = tv.tv_sec;
		}

		for (m = 0; m < aep_context->count_virtual_channels; m++) {
			copy_public_ch_info_to_shared(aep_context->aepd_interface, chan++, &aep_context->vch[m]);
			aep_context->aepd_interface->vchans++;
		}

	}
post_start:

	if (fd_with_rx < 0)
		goto service;

	gettimeofday(&tv, NULL);

	/* somebody had something for us */

	for (m = 0; m <= aep_context->highest; m++) {

		aep = &aep_context->aeps[m];

		if (aep->fd < 1)
			continue;

		/* check for timeout */

		if (tv.tv_sec > aep->sec_last_traffic + 2)
			if (aep->done_config != 3 || aep->started) {
				fprintf(stderr, "%s: timeout\n", aep->dev_filepath);
				goto died;
			}

		if (aep->fd != fd_with_rx)
			continue;

		if (aep_context->verbose)
			fprintf(stderr,"stuff for %d\n", m);

		aep->sec_last_traffic = tv.tv_sec;

		/* work out how much of the ring we can fill */

		if (aep->head <= aep->tail) {
			budget = (aep->head - 1) + (sizeof(aep->ring) - aep->tail);
			p = aep->ring + aep->tail;
			if (!budget)
				continue;

			len = read(aep->fd, buf, budget);
			if (len <= 0) {
				fprintf(stderr, "failed to read data\n");
				goto died;
			}

			if (aep_context->verbose)
				fprintf(stderr, "%d (a) fetched %d (budget %d)  head=%d tail=%d\n", m, len, budget, aep->head, aep->tail);

			n = sizeof(aep->ring) - aep->tail;
			if (len < n)
				n = len;

			memcpy(p, buf, n);
			aep->tail += n;
			if (aep->tail == sizeof(aep->ring))
				aep->tail = 0;

			len -= n;
			if (len) {
				memcpy(aep->ring, &buf[n], len);
				aep->tail += len;
			}

		} else {
			budget = aep->head - aep->tail - 1;
			p = aep->ring + aep->tail;
			if (!budget)
				continue;

			len = read(aep->fd, p, budget);
			if (len <= 0) {
				fprintf(stderr, "failed to read data\n");
				goto died;
			}

			if (aep_context->verbose)
				fprintf(stderr, "%d (b) fetched %d (budget %d) head=%d tail=%d\n", m, len, budget, aep->head, aep->tail);

			aep->tail += len;
		}

		if (aep->tail == sizeof(aep->ring))
			aep->tail = 0;

		continue;

died:
		fprintf(stderr, "removing probe due to error...\n");
		for (i = 0; i < CHANNELS_PER_PROBE; i++)
			if (aep->ch[i].requested) {
				aep_context->awaiting_capture &= ~(1 << (m * 3 + i));
				aep_context->matched_channel_names--;
			}
		probe_close(aep);
		remove_pollfd(aep_context, aep->fd);
		aep->fd = 0;
		m = aep_context->highest;
	}

service:

	/* 
	 * service the existing devices
	 * need to limit the amount of time spent in the service
	 * otherwise other probes will drop data
	 */

	for (n = 0; n <= aep_context->highest; n++) {

		aep = &aep_context->aeps[n];

		if (aep->fd <= 0)
			continue;
		m = aep_protocol_parser(aep, MAX_BYTES_PER_AEP_SERVICE);
		if (m >= 0)
			continue;
		if (m < -1) {
			n = aep_context->highest + 1;
			continue;
		}

		if (!aep_context->verbose)
			continue;
		fprintf(stderr, "service failed\n");
		for (i = 0; i < CHANNELS_PER_PROBE; i++) {
			if (!aep->ch[i].requested)
				continue;
			aep_context->awaiting_capture &= ~(1 << (n * 3 + i));
			aep_context->matched_channel_names--;
		}
		probe_close(aep);
		remove_pollfd(aep_context, aep->fd);
		aep->fd = 0;
	}

	if (aep_context->scans > (5 * MAX_PROBES) &&
			aep_context->awaiting_capture == 0 && aep_context->exit_after_capture) {
		fprintf(stderr, "done all capture\n");
		aep_context->aepd_interface->finished = 1;
		return -1;
	}

	/* if nothing waiting fail immediately if anything has buffered data to service */

	aep_context->poll_timeout_ms = 10;
	for (n = 0; n <= aep_context->highest; n++) {
		if (aep_context->aeps[n].fd <= 0)
			continue;

		if (aep_context->aeps[n].head != aep_context->aeps[n].tail)
			aep_context->poll_timeout_ms = 0;
	}

	return 0;
}

static int loop;

void sighandler(int sig)
{
	loop = 0;
}


/*
 * you should have called aepd_interface_create() above
 * and set aep_context->aepd_interface to the result
 * before calling this
 */

int aep_init_and_fork(struct aep_context *aep_context, char *argv[])
{
	int n, i;
	struct aep_channel *ch;

	loop = 1;

	init_interpolation();

	/* fork off aep service loop */

	n = fork();
	if (n)
		return n;

	signal(SIGINT, sighandler);

	/*
	 * child process runs independently
	 * fills the named pipe fifo with sample packets
	 */

	if (argv)
		strcpy(argv[0] + strlen(argv[0]), " - AEP server");

	aep_context->aepd_interface->semaphore = sem_open(aep_context->aepd_interface->semname, O_RDWR, 0600, 0);
	if (aep_context->aepd_interface->semaphore == SEM_FAILED) {
		fprintf(stderr, "Child failed to open sem %s\n", aep_context->aepd_interface->semname);
		return -1;
	}

	aep_context->aepd_interface->finished = 0;
	aep_context->poll_timeout_ms = 10;

	while (!aep_context->aepd_interface->finished && loop && getppid() != 1) {

		n = poll(aep_context->pollfds, aep_context->count_pollfds, aep_context->poll_timeout_ms);
		if (n < 0)
			aep_context->aepd_interface->finished = 1;

		for (n = 0; n < aep_context->count_pollfds; n++)
			if (aep_context->pollfds[n].revents)
				if (service_aeps(aep_context, aep_context->pollfds[n].fd) < 0)
					aep_context->aepd_interface->finished = 1;

		if (service_aeps(aep_context, -1) < 0)
			aep_context->aepd_interface->finished = 1;
	}

	sem_close(aep_context->aepd_interface->semaphore);

	/*
	 * Update channel config file
	 */

	for (n = 0; n <= aep_context->highest; n++) {
		if (aep_context->aeps[n].fd < 1)
			continue;

		for (i = 0; i < CHANNELS_PER_PROBE; i++) {
			ch = &aep_context->aeps[n].ch[i];			
;
			if (ch->flag_was_configured) {
				if (configure(aep_context, &aep_context->aeps[n],
					aep_context->aeps[n].dev_filepath,
					aep_context->config_filepath, ch) < 0)
				fprintf(stderr, "failed to update config\n");
			}
		}

		probe_close(&aep_context->aeps[n]);
	}

	return 0;
}



