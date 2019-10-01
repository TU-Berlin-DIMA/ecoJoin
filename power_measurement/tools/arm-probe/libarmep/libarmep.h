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
 */

#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/time.h>
#include <termio.h>
#include <time.h>
#include <semaphore.h>

#include "aepd-interface.h"

#define MAX_PROBES 8
#define CHANNELS_PER_PROBE 3

#define AEP_INPUT_QUEUE 163840

#define MAX_RESULT_QUEUE_DEPTH (512 * 8)
#define MAX_BYTES_PER_AEP_SERVICE 512

#define TRIG_HYSTERESIS_MV 200
#define TRIG_HYSTERESIS_MW 200

#define ADC_COUNTS_PER_VOLT_CH1 1000.0
#define ADC_COUNTS_PER_VOLT_CH2 10000.0

#define ADC_COUNT_CURRENT_CLIP_LIMIT 1640
#define REJECT_AUTOZERO_VOLTAGE 1.0
#define IGNORE_AUTOZERO_VOLTAGE 0.2

#define AEP_INVALID_COVER 100

/* policy for modifying offset when seeing below zero */
#define IGNORE_NEG_PRIOR_TO_SAMPLES 500                                    
#define NEGATIVE_ADJUST_SCALING_FACTOR 0.1

#define MAX_VIRTUAL_CHANNELS 8

struct avg_mean_us {
        unsigned short *ring;
        int width;
        int count;
        int next;
        unsigned long sum;
};

enum samples_actions {
	AEPSA_START_PRE,
	AEPSA_SAMPLE_PRE,
	AEPSA_END_PRE,
	AEPSA_START,
	AEPSA_SAMPLE,
	AEPSA_END,
	AEPSA_DATA_STARTING,
	AEPSA_DATA_STARTING_CHANNELS,
	AEPSA_DATA_STARTING_DONE,
	AEPSA_ADD_POLLFD,
	AEPSA_REMOVE_POLLFD
};

struct aep_channel;
struct aep;


enum state {
	APP_INIT_MAGIC,
	APP_INIT_VERSION,
	APP_INIT_VENDOR,
	APP_INIT_RATE,
	APP_INIT_CONFIG_1,
	APP_INIT_CONFIG_2,
	APP_INIT_CONFIG_3,

	APP_INIT_START_ACK,

	APP_FRAME_L,
	APP_FRAME_H_BAD,
	APP_FRAME_H,
	APP_VOLTAGE_L1_BAD,
	APP_VOLTAGE_H1_BAD,
	APP_CURRENT_L1_BAD,
	APP_CURRENT_H1_BAD,
	APP_VOLTAGE_L2_BAD,
	APP_VOLTAGE_H2_BAD,
	APP_CURRENT_L2_BAD,
	APP_CURRENT_H2_BAD,
	APP_VOLTAGE_L3_BAD,
	APP_VOLTAGE_H3_BAD,
	APP_CURRENT_L3_BAD,
	APP_CURRENT_H3_BAD,
	APP_SWALLOW,

	APP_VOLTAGE_L1,
	APP_VOLTAGE_H1,
	APP_CURRENT_L1,
	APP_CURRENT_H1,
	APP_VOLTAGE_L2,
	APP_VOLTAGE_H2,
	APP_CURRENT_L2,
	APP_CURRENT_H2,
	APP_VOLTAGE_L3,
	APP_VOLTAGE_H3,
	APP_CURRENT_L3,
	APP_CURRENT_H3,
};

struct samples {
	unsigned short v;
	unsigned short i;
};

/*
 * reference percentage is the error the channel exhibits at
 * 5.0V, 10mA 470mR (approx 3.8mV shunt voltage)
 */

struct interp_tables {
        struct interp *map;
        int len;
        double percentage_error_ref;
	/* this boils down to 10.8KBytes per interp table (currently 3) */
	unsigned short precomputed_common_10mV[600][9];
};


struct aep_channel {
	struct aep *aep;
	char channel_name[256];
	char channel_name_pretty[256];
	char supply[64]; /* parent channel name */
	char colour[16]; /* #xxxxxx HTML-style colour */
	char class[16]; /* optional class of power, eg, Soc */
	int pretrigger_ms;
	int pretrigger_samples_taken;
	int channel_num;
	double rshunt;
	double voffset[2];
	double vnoise[2];

	int requested;
	unsigned long samples;

	struct interp_tables *map_table;

	struct samples *pretrig_ring;
	int ring_samples;
	int head;
	int tail;
	int trigger_filter;
	int triggered;
	struct aep_channel *trigger_slave;

	struct avg_mean_us avg_mean_voltage;
	struct avg_mean_us avg_mean_current;

	/*
	 * used to detect and adjust -ve drift below 0
	 * 0: uncorrected V1,
	 * 1: uncorrected V2
	 */	

	double min[2];
	double max[2];

	int decimation_counter;
	unsigned long samples_seen;

	double out_ring[2][MAX_RESULT_QUEUE_DEPTH];
	int out_head;
	int out_tail;
	int ignore;

	double simple_avg[3];
	double avg_count;

	char summary[128];
	int flag_was_configured;
};

struct aep {
	struct aep_context *aep_context;
	int fd;
	char name[64];
	char dev_filepath[128];
	int index;
	int head;
	int tail;
	unsigned int version;
	unsigned int rate;
	enum state state;
	unsigned short predicted_frame;
	unsigned short voltage;
	unsigned short current;
	int invalid;
	unsigned char ring[AEP_INPUT_QUEUE];
	int counter;
	int sec_last_traffic;
	int started;
	int done_config;
	int rx_count;
	struct aep_channel ch[CHANNELS_PER_PROBE];
};


struct aep_context {
	struct aep aeps[MAX_PROBES];
	struct aep_channel vch[MAX_VIRTUAL_CHANNELS];
	int count_virtual_channels;
	int auto_zero;
	unsigned int last_scan_sec;
	char config_filepath[256];
	char configuration_name[64];
	char device_paths[MAX_PROBES][256];
	int highest;
	int decimate;
	int no_correction;
	unsigned int mv_min;
	unsigned int mw_min;
	unsigned int mw_min_plus_hyst;
	unsigned int trigger_filter_us;
	unsigned int end_trigger_filter_us;
	unsigned int ms_pretrigger;
	int ms_holdoff;
	int ms_capture;
	int require_off;
	int show_raw;
	int awaiting_capture;
	int scan;
	int scans;
	int has_started;
	int original_count_channel_names;
	int count_channel_names;
	int matched_channel_names;
	char channel_name[MAX_PROBES][64];
	int verbose;
	int average_len;
	struct pollfd pollfds[MAX_PROBES];
	int count_pollfds;
	int exit_after_capture;
	int poll_timeout_ms;
	double do_average;
	int done_az_channels;

	/* shared memory output buffer + metadata */

	struct aepd_interface  *aepd_interface;
};


enum aep_commands {
	AEPC_RESET = 0xff,
	AEPC_VERSION = 0x01,
	AEPC_VENDOR = 0x03,
	AEPC_RATE = 0x05,
	AEPC_CONFIG = 0x07,
	AEPC_START = 0x09,
	AEPC_STOP = 0x0b,
};

/* define interpolated correction tables */

struct interp {
	double common_mode;
	double actual;
	double raw;
};

/*
 * internal
 */

extern int aep_protocol_parser(struct aep *aep, int samples);
extern int process_sample(struct aep *aep, int ch_index);
extern int avg_mean_us_init(struct avg_mean_us *amu, int width);                       
extern void avg_mean_us_add(struct avg_mean_us *amu, unsigned short s);                
extern double avg_mean_us(struct avg_mean_us *amu);
extern void avg_mean_us_free(struct avg_mean_us *amu);
extern void avg_mean_us_flush(struct avg_mean_us *amu);
extern struct interp_tables interp_tables[3];
extern double correct(int no_correction, double common_mode, double in, struct interp_tables *itable);
extern int configure(struct aep_context *aep_context, struct aep *aep, const char *dev_filepath,
	const char *config_filepath, struct aep_channel *wch);
extern void probe_close(struct aep *aep);
extern void init_interpolation(void);

/*
 * external apis
 */

extern int service_aeps(struct aep_context *aep_context, int fd);
extern int aep_init_and_fork(struct aep_context *aep_context, char *argv[]);

