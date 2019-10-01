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

/*
 * We take a strict whitelist approach to stop ../ attacks
 */

struct serveable {
	const char *urlpath;
	const char *mimetype;
}; 

static const struct serveable whitelist[] = {
	{ "/favicon.ico", "image/x-icon" },
	{ "/linaro-logo-32.png", "image/png" },
	{ "/caliper.png", "image/png" },
	{ "/trigger.png", "image/png" },
	{ "/indicator-bg.png", "image/png" },
	{ "/indicator-triggered.png", "image/png" },

	/* last one is the default served if no match */
	{ "/aepscope.html", "text/html" },
};

/* this protocol server (always the first one) just knows how to do HTTP */

static int
callback_http(struct lws *wsi, enum lws_callback_reasons reason, void *user,
	      void *in, size_t len)
{
	int n;
	char buf[512];

	switch (reason) {
	case LWS_CALLBACK_HTTP:

		for (n = 0; n < (ARRAY_SIZE(whitelist) - 1); n++)
			if (in && strcmp(in, whitelist[n].urlpath) == 0)
				break;

		sprintf(buf, LOCAL_RESOURCE_PATH"%s", whitelist[n].urlpath);

		if (lws_serve_http_file(wsi, buf, whitelist[n].mimetype,
					NULL, 0))
			//fprintf(stderr, "Failed to send HTTP file\n");
			break;
		break;

	default:
		break;
	}

	return 0;
}



/* linaro_aepd_protocol */

struct per_session_data__linaro_aepd {
	struct lws *wsi;
	long ringbuffer_tail;
	double sam[MAX_PROBES * CHANNELS_PER_PROBE * 3];
	int sam_valid;
	int stride;
	int channels_sent_flag;
	int issue_timestamp;
	int viewport_budget;
	double viewport_offset_time;
	long caliper_offset[2];
	double caliper_time[2];
	unsigned long ms10_last_caliper;
	unsigned long ms10_last_triglevel;
	int seen_rate;
};

extern struct aep_context aep_context;

static int
callback_linaro_aepd(struct lws *wsi, enum lws_callback_reasons reason,
		     void *user, void *in, size_t len)
{
	int n, m;
	struct per_session_data__linaro_aepd *pss = user;
	double sam[MAX_PROBES * CHANNELS_PER_PROBE * 3];
	double sam2[MAX_PROBES * CHANNELS_PER_PROBE * 3];
	double d[10];
	char buf[LWS_SEND_BUFFER_PRE_PADDING + 16384 + LWS_SEND_BUFFER_POST_PADDING];
	char *p = &buf[LWS_SEND_BUFFER_PRE_PADDING];
	int budget = 10;
	int no_valid_sam_flag = 0;
	long extent;
	long l;
	struct timeval tv;
	unsigned long ms10;

	switch (reason) {

	case LWS_CALLBACK_ESTABLISHED:
		pss->ringbuffer_tail = aepd_shared->fifo_pos;
		pss->sam_valid = 0;
		pss->wsi = wsi;
		pss->stride = 100;
		pss->channels_sent_flag = 0;
		pss->issue_timestamp = 1;
		pss->viewport_offset_time = 0;
		pss->viewport_budget = 0;
		pss->caliper_offset[0] = -1;
		pss->caliper_offset[1] = -1;
		pss->seen_rate = 0;
		break;

	case LWS_CALLBACK_SERVER_WRITEABLE:

		if (aepd_shared->chans < 0)
			break;

		gettimeofday(&tv, NULL);
		ms10 = (tv.tv_sec * 10) + (tv.tv_usec / 100000);

		/*
		 * is it time to do a caliper update?
		 */

		if (ms10 != pss->ms10_last_caliper && aepd_shared->chans) {

			pss->ms10_last_caliper = ms10;

			if (pss->caliper_time[0] > aepd_shared->fifo_head_time ||
				  pss->caliper_time[1] > aepd_shared->fifo_head_time)
				goto bad_caliper;

			if (pss->caliper_time[0] < 0 ||
				  pss->caliper_time[1] < 0)
				goto bad_caliper;

			l = pss->ringbuffer_tail - pss->caliper_offset[0];
			if (l < 0)
				l += aepd_shared->modulo_integer_chan_size;
			if (l < 0)
				goto bad_caliper;
			if (lseek(aepd_shared->fd_fifo_read, l, SEEK_SET) < 0)
				fprintf(stderr, "seek %ld failed\n", l);
			if (sizeof(sam) < (aepd_shared->chans * sizeof(double) * 3))
				fprintf(stderr, "reading too much %d\n", aepd_shared->chans);
			if (read(aepd_shared->fd_fifo_read, &sam[0],
					aepd_shared->chans * sizeof(double) * 3) < 0)
				fprintf(stderr, "fifo read %ld failed\n", l);

			l = pss->ringbuffer_tail - pss->caliper_offset[1];
			if (l < 0)
				l += aepd_shared->modulo_integer_chan_size;
			if (l < 0)
				goto bad_caliper;
			if (lseek(aepd_shared->fd_fifo_read, l, SEEK_SET) < 0)
				fprintf(stderr, "seek %ld failed\n", l);
			if (sizeof(sam2) < (aepd_shared->chans * sizeof(double) * 3))
				fprintf(stderr, "reading too much %d\n", aepd_shared->chans);
			if (read(aepd_shared->fd_fifo_read, &sam2[0],
					aepd_shared->chans * sizeof(double) * 3) < 0)
				fprintf(stderr, "fifo read %ld fail\n", l);

			extent = (pss->caliper_offset[1] - pss->caliper_offset[0]) /
						(aepd_shared->chans * sizeof(double) * 3);

			*p++ = 'c';
			p += sprintf(p, "%f,", aepd_shared->fifo_head_time);
			m = 0;
			for (n = 0; n < aepd_shared->chans; n++) {

				p += sprintf(p, "%f %f %f",
					(sam[m] - sam2[m]) / (double)extent,
					(sam[m + 1] - sam2[m + 1]) / (double)extent,
					(sam[m + 2] - sam2[m + 2]) / (double)extent
				);

				if (n + 1 != aepd_shared->chans) {
					*p++ = ',';
					*p = '\0';
				}
				m += 3;
			}

			*p = '\0';
			goto send;
bad_caliper:
			*p++ = 'C';
			*p = '\0';
			goto send;
		}

		if (ms10 != pss->ms10_last_triglevel && aepd_shared->chans) {
			pss->ms10_last_caliper = ms10;
		}

		/*
		 * do we need to issue a sync timestamp?
		 */

		if (pss->issue_timestamp && aepd_shared->chans) {
			pss->issue_timestamp = 0;

			/*
			 * we want to set the sample time context for what we are
			 * about to send
			 */

			pss->sam_valid = 0;

			l = pss->ringbuffer_tail;

			if (l <= (long)aepd_shared->fifo_pos)
				extent = aepd_shared->fifo_pos - l;
			else
				extent = (aepd_shared->modulo_integer_chan_size - l) + aepd_shared->fifo_pos;

			extent /= aepd_shared->chans * sizeof(double) * 3;


			p += sprintf(p, "t%f %f %f %d %d", aepd_shared->fifo_head_time - ((double)extent * 0.0001), aepd_shared->fifo_tail_time, aepd_shared->fifo_head_time, aepd_shared->stop_flag ^ 1, pss->viewport_budget);
			goto send;
		}

		/*
		 * do we need to dump channel info?
		 */

		if (!pss->channels_sent_flag && aepd_shared->chans) {

			/* signal it's a message with channel names */
			*p++ = '=';

			for (n = 0; n < aep_context.aepd_interface->chans + aep_context.aepd_interface->vchans; n++)
				p += sprintf(p, "%s,%s,%s,%s,;",
					aep_context.aepd_interface->channel_name[n],
					aep_context.aepd_interface->supply[n],
					aep_context.aepd_interface->colour[n],
					aep_context.aepd_interface->class[n]);

			pss->channels_sent_flag = 1;

			goto send;
		}

		/*
		 * if we're not following the samples head, we don't need to
		 * spam the viewport any more than one viewport's worth of samples.
		 *
		 * We're still collecting samples, he can get them by moving his
		 * viewport offset
		 */

		if (pss->viewport_offset_time < -0.00009)
			if (!pss->viewport_budget)
				break;

		/*
		 * if we didn't have to do any of the other tasks,
		 * aggregate up to 'budget' pending results in one websocket message
		 */

		if (!pss->seen_rate || !aepd_shared->chans)
			return 0;

		while (budget-- && (p - &buf[LWS_SEND_BUFFER_PRE_PADDING]) < (sizeof(buf) - 4096)) {

			if (pss->ringbuffer_tail <= (long)aepd_shared->fifo_pos)
				extent = aepd_shared->fifo_pos - pss->ringbuffer_tail;
			else
				extent = (aepd_shared->modulo_integer_chan_size - pss->ringbuffer_tail) + aepd_shared->fifo_pos;

			if (pss->ringbuffer_tail >= 0 && extent < (pss->stride * aepd_shared->chans * sizeof(double) * 3)) {
				budget = 0;
				continue;
			}

			if (pss->ringbuffer_tail < 0) {
				no_valid_sam_flag = 1;
				pss->sam_valid = 0;
				/* force a 0 sample */
				for (n = 0; n < aepd_shared->chans * 3; n++)
					sam[n] = pss->sam[n];
			} else {
				no_valid_sam_flag = 0;
				lseek(aepd_shared->fd_fifo_read, pss->ringbuffer_tail, SEEK_SET);
				if (read(aepd_shared->fd_fifo_read, &sam[0], aepd_shared->chans * sizeof(double) * 3) < 0)
					fprintf(stderr, "fifo read fail\n");
			}

			if (pss->viewport_budget)
				pss->viewport_budget--;
			else
				if (pss->viewport_offset_time < -0.00009)
					goto send;

			pss->ringbuffer_tail += (pss->stride * aepd_shared->chans * sizeof(double) * 3 );
			if (pss->ringbuffer_tail > (long)aepd_shared->modulo_integer_chan_size)
				pss->ringbuffer_tail -=  aepd_shared->modulo_integer_chan_size;

			if (pss->sam_valid || no_valid_sam_flag) {

				/*
				 * Javascript can't cope with binary, so we must ascii-fy it
				 */

				m = 0;
				for (n = 0; n < aepd_shared->chans; n++) {

					if ((sam[m] - pss->sam[m]) < 0)
						fprintf(stderr, "sam[m] %lf < pss->sam[m] %lf\n", sam[m], pss->sam[m]);

					p += sprintf(p, "%f %f %f",
						(sam[m] - pss->sam[m]) / (double)pss->stride,
						(sam[m + 1] - pss->sam[m + 1]) / (double)pss->stride,
						(sam[m + 2] - pss->sam[m + 2]) / (double)pss->stride
					);

					if (n + 1 != aepd_shared->chans) {
						*p++ = ',';
						*p = '\0';
					}
					m += 3;
				}

				*p++ = ';';
				*p = '\0';
			}
			memcpy(&pss->sam[0], &sam[0], aepd_shared->chans * sizeof(double) * 3);
			if (!no_valid_sam_flag)
				pss->sam_valid = 1;


			if ((p - &buf[LWS_SEND_BUFFER_PRE_PADDING]) > (sizeof(buf) - 2048)) {
				fprintf(stderr, "insane buffer usage %d! pss->ringbuffer_tail = %ld, budget=%d aepd_shared->chans=%d\n",
					(int)(p - &buf[LWS_SEND_BUFFER_PRE_PADDING]), pss->ringbuffer_tail, budget, aepd_shared->chans);
				budget = 0;
			}
		}
send:
		/*
		 * if we generated something, send it.  We are guaranteed not to block
		 * because we got here by POLLOUT seen on the socket
		 */

		if (p != &buf[LWS_SEND_BUFFER_PRE_PADDING]) {
			n = lws_write(wsi, (unsigned char *)
				   &buf[LWS_SEND_BUFFER_PRE_PADDING],
				   p - &buf[LWS_SEND_BUFFER_PRE_PADDING], LWS_WRITE_TEXT);
			if (n < 0) {
				fprintf(stderr, "ERROR writing to socket");
				return 1;
			}
			lws_callback_on_writable(wsi);
		}

		break;

	case LWS_CALLBACK_RECEIVE:

		switch (*(char *)in) {
		case 'Z': /* zero */
			fprintf(stderr, "zero\n");
			aepd_shared->zero = 1;
			break;
		case 't': /* set trigger level */
			if (sscanf(((char *)in) + 1, "%lf\n", &d[0]) == 1) {
				aepd_shared->trigger_level = d[0];
				/* set it */
			}
			break;

		case 'c': /* calipers changed */
			if (sscanf(((char *)in) + 1, "%lf %lf\n", &d[0], &d[1]) == 2) {

				if (d[0] != d[0] || d[1] != d[1]) { /* NaN */
					pss->caliper_offset[0] = -1;
					pss->caliper_time[0] = -1;
					pss->caliper_offset[1] = -1;
					pss->caliper_time[1] = -1;
					break;
				}

				if (d[0] > d[1]) {
					d[2] = d[1];
					d[1] = d[0];
					d[0] = d[2];
				}

				/*
				 * caliper positions in seconds behind rhs -->
				 * byte offset behind rhs in ringbuffer
				 */
				for (n = 0; n < 2; n++) {
					pss->caliper_offset[n] = (d[n] * 10000 * aepd_shared->chans * sizeof(double) * 3);
					pss->caliper_time[n] = d[n];
				}
			} else
				fprintf(stderr, "caliper sscanf failed\n");
			break;

		case 'r': /* rate or other change */
			if (sscanf(((char *)in) + 1, "%lf %lf %lf %lf\n", &d[0], &d[1], &d[2], &d[3]) == 4) {
				pss->stride = (int)d[0];
				pss->viewport_offset_time = d[3];

				l = aepd_shared->fifo_pos - ((int)d[1] * pss->stride * aepd_shared->chans * sizeof(double) * 3);
				l += (int)((pss->viewport_offset_time / 0.0001)) * aepd_shared->chans * sizeof(double) * 3;
				if (pss->viewport_offset_time)
					pss->viewport_budget = d[1];
				pss->ringbuffer_tail = l;
				aepd_shared->stop_flag = ((int)d[2]) ^ 1;
				pss->issue_timestamp = 1;
				pss->seen_rate = 1;
				pss->sam_valid = 0;
				lws_callback_on_writable(wsi);
			} else
				fprintf(stderr, "sscanf failed\n");
			break;
		}
		break;

	default:
		break;
	}

	return 0;
}

/* list of supported protocols and callbacks */

const struct lws_protocols protocols[] = {
	/* first protocol must always be HTTP handler */

	{
		"http-only",		/* name */
		callback_http,		/* callback */
		0			/* per_session_data_size */
	},
	{
		"linaro.aepd",
		callback_linaro_aepd,
		sizeof(struct per_session_data__linaro_aepd),
	},
	{
		NULL, NULL, 0		/* End of list */
	}
};

