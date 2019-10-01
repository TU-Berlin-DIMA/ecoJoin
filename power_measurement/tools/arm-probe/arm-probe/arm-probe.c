/*
 * Author: Daniel Lezcano <daniel.lezcano@linaro.org> (Initial version)
 *       : Andy Green <andy.green@linaro.org> 
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
 */

#include "../libarmep/libarmep.h"

int just_power = 0;
char stdinbuf[512];
char stdinline[512] = "";
int stdinpos = 0;
int stdinlen = 0;

struct aepd_interface *aepd_interface;

struct aep_context aep_context = {
	.config_filepath = "./config",
	.highest = -1,
	.decimate = 1,
	.mv_min = 400,
	.trigger_filter_us = 400,
	.end_trigger_filter_us = 500000,
	.average_len = 1,
	.configuration_name = "default_configuration",
};

void pull_stdin_if_needed(void)
{
	int n;

	if (stdinpos != stdinlen)
		return;

	n = read(0, stdinbuf, sizeof stdinbuf);
	if (n < 0) {
		if (aep_context.verbose)
			fprintf(stderr, "pull_stdin_if_needed: n=%d\n", n);
		return;
	}
	if (n == 0 && aep_context.verbose)
		fprintf(stderr, "pull_stdin_if_needed n = 0\n");
	stdinpos = 0;
	stdinlen = n;
}

int get_stdin_line(char *buf, int maxlen)
{
	while (maxlen--) {
		if (stdinpos == stdinlen) {
			pull_stdin_if_needed();
			if (stdinpos == stdinlen)
				return -1;
		}
		*buf++ = stdinbuf[stdinpos++];
		if (buf[-1] == '\n') {
			buf[-1] = '\0';
			return 0;
		}
	}
	return -1;
}

static struct option options[] = {
	{ "help",	no_argument,		NULL, 'h' },
	{ "verbose",	no_argument,		NULL, 'v' },
	{ "raw",	no_argument,		NULL, 'r' }, 
	{ "mean",	required_argument,	NULL, 'm' },
	{ "decimate",	required_argument,	NULL, 'd' },
	{ "mvtrigger",	required_argument,	NULL, 'q' },
	{ "mwtrigger",	required_argument,	NULL, 'w' },
	{ "channel",	required_argument,	NULL, 'c' },
	{ "mstrighold",	required_argument, 	NULL, 't' },
	{ "ustrigfilter", required_argument,	NULL, 'f' },
	{ "mslength",	required_argument, 	NULL, 'l' },
	{ "exitoncap",	no_argument,		NULL, 'x' },
	{ "justpower",	no_argument,		NULL, 'j' },
	{ "config",	required_argument,	NULL, 'C' },
	{ "offexit",	no_argument,		NULL, 'o' },
	{ "mspretrigger", required_argument,	NULL, 'p' },
	{ "nocorrection", no_argument,		NULL, 'n' },
	{ "autozero",	no_argument,		NULL, 'z' },
	{ "average",	required_argument,	NULL, 'a' },
	{ NULL, 0, 0, 0 }
};

static int loop;

void sighandler(int sig)
{
	loop = 0;
}

int main(int argc, char *argv[])
{
	time_t t;
	struct tm *tmp;
	char hostname[200];
	char date[200];
	int n = 0;
	int m, i;
	int periodic = 0;
	double tt;
	struct aepd_interface_result *aepd_interface_result;
	int first = 1;

	/* simple stats on what was captured */

	double averages[AEPD_SHARED_MAX_REAL_CHANNELS][3];
	double min[AEPD_SHARED_MAX_REAL_CHANNELS][3];
	double max[AEPD_SHARED_MAX_REAL_CHANNELS][3];
	double count_post_trigger = 0;

	loop = 1;

	signal(SIGINT, sighandler);

	while (n >= 0) {
		n = getopt_long(argc, argv, "xl:t:c:q:hvm:jC:w:f:op:rd:nza:",
							options, NULL);
		if (n < 0)
			continue;
		switch (n) {
		case 'a':
			aep_context.do_average = atof(optarg);
			break;
		case 'z':
			fprintf(stderr, "\n\n *** Note: for autozero, you "
				"should short both sides of the "
				"\n     channel sense "
				"leads to 0V... in some cases a powered-down "
				"target will do ***\n\n");
			aep_context.no_correction = 1;
			aep_context.mv_min = 0;
			aep_context.mw_min = 0;
			aep_context.auto_zero = 1;
			aep_context.average_len = 50000;
			aep_context.ms_capture = aep_context.average_len / 10;
			aep_context.decimate = aep_context.average_len;
			aep_context.exit_after_capture = 1;
			break;
		case 'n':
			aep_context.no_correction = 1;
			break;
		case 'd':
			aep_context.decimate = atoi(optarg);
			break;
		case 'r':
			aep_context.show_raw = 1;
			break;
		case 'p':
			aep_context.ms_pretrigger = atoi(optarg);
			break;
		case 'o':
			aep_context.require_off = 1;
			break;
		case 'f':
			aep_context.trigger_filter_us = atoi(optarg);
			break;
		case 'w':
			aep_context.mw_min = atoi(optarg);
			if (aep_context.mw_min)
				aep_context.mw_min_plus_hyst =
				 aep_context.mw_min + (aep_context.mw_min / 10);
			break;
		case 'j':
			just_power = 1;
			break;
		case 'C':
			strncpy(aep_context.config_filepath, optarg,
					sizeof aep_context.config_filepath - 1);
			aep_context.config_filepath[
				sizeof aep_context.config_filepath - 1] = '\0';
			break;
		case 't':
			aep_context.ms_holdoff = atoi(optarg);
			break;
		case 'l':
			aep_context.ms_capture = atoi(optarg);
			break;
		case 'c':
			if (aep_context.count_channel_names == MAX_PROBES) {
				fprintf(stderr, "too many channels named\n");
				return -1;
			}
			strncpy(aep_context.channel_name[aep_context.count_channel_names], optarg,
						sizeof aep_context.channel_name[0] - 1);
			aep_context.channel_name[aep_context.count_channel_names]
					[sizeof aep_context.channel_name[0] - 1] = '\0';
			aep_context.count_channel_names++;
			break;
		case 'q':
			aep_context.mv_min = atoi(optarg);
			break;
		case 'm':
			aep_context.average_len = atoi(optarg);
			break;
		case 'v':
			aep_context.verbose++;
			break;
		case 'x':
			aep_context.exit_after_capture = 1;
			break;
		default:
		case 'h':
			fprintf(stderr, "Usage: arm-probe \n"
				" [--verbose -v] Increase debug on stderr\n"
				" [--config -C <file>] use alternate config "
					"(default ./config)\n"
				" [--mean -m <averagng depth>] (default 1)\n"
				" [--decimate -d <report interval>] only "
					"output every n samples\n"
				" [--mvtrigger -q <min mV>] suppress output "
					"below this voltage (defaut 400mV)\n"
				" [--mwtrigger -w <min mW>] suppress output "
					"below this power (default 0mW)\n"
				" [--channel -c <name>] select channel\n"
				" [--mstrighold -t <ms>] delay in ms after "
					"power seen before capture\n"
				" [--ustrigfilter -f <us>] require trigger "
					"threshold to be exceeded for at "
					"least this amount of time\n"
				" [--mspretrigger -p <ms>] at trigger, issue "
					"this many samples from BEFORE "
					"trigger first\n"
				" [--length -l <ms>] period to capture after "
					"trigger (+holdoff if any)\n"
				" [--exitoncap -x] after requested capture "
					"(-l) completes on all channels, "
					"exit app\n"
				" [--offexit -o] wait for voltage to reduce "
					"below trigger before exiting\n"
				" [--justpower -j] just write power\n"
				" [--autozero -z] measure channel and store "
					"result as zero offset\n"
				" [--average -a <float>] at the end of the "
					"capture, add a simple mean average "
					"to the \n"
					"        stdout dataset, starting at "
					"<float> seconds\n"
			);
			exit(1);
		}
	}

	aep_context.original_count_channel_names = aep_context.count_channel_names;

	if (!isatty(0)) {

		m = 1;
		n = 0;
		while (m) {
			n++;
			if (n > 1000) {
				fprintf(stderr, "problem with stdin format\n");
				return 1;
			}
			if (get_stdin_line(stdinline, sizeof stdinline)) {
				fprintf(stderr, "get_stdin_line failed\n");
				m = 0;
				continue;
			}
			if (stdinline[0] != '#') {
				if (aep_context.verbose)
					fprintf(stderr, "seen '%s'\n", stdinline);
				puts("#");
				m = 0;
				continue;
			}
			puts(stdinline);
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

	/* configure */

	configure(&aep_context, NULL, "/virtual", aep_context.config_filepath, NULL);

	/*
	 * fork off the AEP service process
	 * runs in its own process to exploit SMP to dedicate one core for that
	 * what happens to samples is decoupled from capture process with large
	 * shared-memory buffer to allow for jitter
	 */

	if (aep_init_and_fork(&aep_context, argv) < 1)
		return 0; /* child process exit */

	
	printf("# configuration: %s\n", aep_context.config_filepath);
	printf("# config_name: %s\n", aep_context.configuration_name);
	printf("# trigger: %fV (hyst %fV) %fW (hyst %fW) %uus\n",
		aep_context.mv_min / 1000.0, TRIG_HYSTERESIS_MV / 1000.0, 
		aep_context.mw_min / 1000.0, TRIG_HYSTERESIS_MV / 1000.0,
		aep_context.trigger_filter_us);

	fprintf(stderr, "Configuration: %s\n", aep_context.configuration_name);

	t = time(NULL);
	tmp = localtime(&t);
	if (tmp) {
		if (strftime(date, sizeof date, "%a, %d %b %Y %T %z", tmp))
			printf("# date: %s\n", date);
	}

	if (!gethostname(hostname, sizeof hostname))
		printf("# host: %s\n", hostname);

	printf("#\n");

	for (n = 0; n < AEPD_SHARED_MAX_REAL_CHANNELS; n++)
		for (m = 0; m < 3; m++) {
			averages[n][m] = 0.0;
			min[n][m] = 999;
			max[n][m] = 0;
		}

	/*
	 * service any AEP results
	 */

	while (loop) {

		aepd_interface_result = aep_wait_for_next_result(aepd_interface);
		if (!aepd_interface_result) {
			if (aepd_interface->finished)
				loop = 0;

			continue;
		}

		if (first) {
			first = 0;
			for (n = 0; n < aepd_interface->chans + aepd_interface->vchans; n++)
				printf("# %s\t%s\t%s\t%s\t%s\n",
					aepd_interface->channel_name[n],
					aepd_interface->channel_name_pretty[n],
					aepd_interface->supply[n],
					aepd_interface->colour[n],
					aepd_interface->class[n]);

			printf("#\n#\ntime ");
			for (n = 0; n < aepd_interface->chans; n++)
				if (just_power)
					printf(" %s(W)", aepd_interface->channel_name_pretty[n]);
				else
					printf(" %s(V) %s(A) %s(W)",
						aepd_interface->channel_name_pretty[n],
						aepd_interface->channel_name_pretty[n],
						aepd_interface->channel_name_pretty[n]);
			printf("\n");
		}

		periodic++;
		if ((periodic & 0x1ff) == 0) {
			if (aepd_interface_result->triggered)
				fprintf(stderr, "TRIGD ");
			else
				fprintf(stderr, "ARMED ");
		}
		if (aepd_interface_result->triggered)
			printf("%f ", aepd_interface_result->samtime);

		m = 0;
		for (i = 0; i < aepd_interface_result->chans * 2; i += 2) {
			if (aepd_interface_result->triggered) {
				if (just_power)
					printf(" %.5f", aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1]);
				else
					printf(" %.2f %.4f %.5f",
						aepd_interface_result->buf[i], aepd_interface_result->buf[i + 1], aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1]);
			}

			averages[m][0] += aepd_interface_result->buf[i];
			averages[m][1] += aepd_interface_result->buf[i + 1];
			averages[m][2] += aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1];

			if (min[m][0] > aepd_interface_result->buf[i])
				min[m][0] = aepd_interface_result->buf[i];
			if (min[m][1] > aepd_interface_result->buf[i + 1])
				min[m][1] = aepd_interface_result->buf[i + 1];
			if (min[m][2] > aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1])
				min[m][2] = aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1];

			if (max[m][0] < aepd_interface_result->buf[i])
				max[m][0] = aepd_interface_result->buf[i];
			if (max[m][1] < aepd_interface_result->buf[i + 1])
				max[m][1] = aepd_interface_result->buf[i + 1];
			if (max[m][2] < aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1])
				max[m][2] = aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1];

			m++;

			if (periodic & 0x1ff)
				continue;
			if (just_power)
				fprintf(stderr, " %.5f", aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1]);
			else
				fprintf(stderr, " %.2f %.4f %.5f",
						aepd_interface_result->buf[i], aepd_interface_result->buf[i + 1], aepd_interface_result->buf[i] * aepd_interface_result->buf[i + 1]);
		}

		count_post_trigger++;

		if (aepd_interface_result->triggered) {
			printf("\n");
		}

		aep_free_result(aepd_interface);

		if (periodic & 0x1ff)
			continue;
		fprintf(stderr, "\n");

	}

	aep_context.aepd_interface->finished = 1;

	/*
	 * we are finished if we reach here...
	 * append simple average to results if requested
	 */

	if (aep_context.do_average) {

		tt = aep_context.do_average;

		for (m = 0; m < 2; m++) {

			if (stdinline[0]) {
				printf("%s", stdinline);
				if (get_stdin_line(stdinline, sizeof stdinline))
					stdinline[0] = '\0';
			} else
				printf("%f", tt);

			for (n = 0; n <= aep_context.aepd_interface->chans; n++) {
				if (just_power)
					printf(" %.5f", averages[n][2]);
				else
					printf(" %.2f %.4f %.5f",
						averages[n][0] / count_post_trigger,
						averages[n][1] / count_post_trigger,
						averages[n][2] / count_post_trigger);
			}
			printf("\n");

			tt += aep_context.do_average / 3;
		}
	}

	/*
	 * dump a summary of extents and mean of all channels now we are done
	 */

	fprintf(stderr, "\n\n");

	for (n = 0; n < aepd_interface->chans; n++) {

		if (min[n][0] == 999)
			continue;

		printf("%12s:  %4.2fV < %4.3fVavg < %4.2fV, "
			"%6.4fA < %6.5fAavg < %6.4fA, "
			"%9.6fW < %9.6fWavg < %9.6fW\n",
				aepd_interface->channel_name[n],
				min[n][0], averages[n][0] / count_post_trigger, max[n][0],
				min[n][1], averages[n][1] / count_post_trigger, max[n][1],
				min[n][2], averages[n][2] / count_post_trigger, max[n][2]);


		/*fprintf(stderr, "%12s:  %4.2fV < %4.3fVavg < %4.2fV, "
				"%6.4fA < %6.5fAavg < %6.4fA, "
				"%9.6fW < %9.6fWavg < %9.6fW\n",
					aepd_interface->channel_name[n],
					min[n][0], averages[n][0] / count_post_trigger, max[n][0],
					min[n][1], averages[n][1] / count_post_trigger, max[n][1],
					min[n][2], averages[n][2] / count_post_trigger, max[n][2]);*/
	}

	aepd_interface_destroy(aepd_interface);

	fprintf(stderr, "exited\n");
	
	return 0;
}

