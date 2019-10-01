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
#include <fcntl.h> 
#include "libarmep.h"

enum fields {
	AEPC_FIELD_NAME,
	AEPC_FIELD_RSHUNT,
	AEPC_FIELD_INTERCHANNEL_ERROR_ESTIMATE,
	AEPC_FIELD_ZERO_OFFSET_V1,
	AEPC_FIELD_ZERO_NOISE_V1,
	AEPC_FIELD_ZERO_OFFSET_V2,
	AEPC_FIELD_ZERO_NOISE_V2,
	AEPC_FIELD_RELATIVE_PRETRIG,
	AEPC_FIELD_PRETTY_NAME,
	AEPC_FIELD_SUPPLY,
	AEPC_FIELD_COLOUR,
	AEPC_FIELD_CLASS,

	/* always last */
	FIELDS_PER_CHANNEL
};

static void process_token(struct aep_channel *ch, char *marshall, enum fields field)
{

	switch (field) {
	case AEPC_FIELD_RSHUNT:
		ch->rshunt = atof(marshall);
		if (!ch->rshunt)
			fprintf(stderr, "**** channel cannot have 0R shunt\n");
		break;
	case AEPC_FIELD_INTERCHANNEL_ERROR_ESTIMATE:
		ch->channel_num = atoi(marshall);
		break;
	case AEPC_FIELD_ZERO_OFFSET_V1:
		ch->voffset[0] = atof(marshall);
		break;
	case AEPC_FIELD_ZERO_NOISE_V1:
		ch->vnoise[0] = atof(marshall);
		break;
	case AEPC_FIELD_ZERO_OFFSET_V2:
		ch->voffset[1] = atof(marshall);
		break;
	case AEPC_FIELD_ZERO_NOISE_V2:
		ch->vnoise[1] = atof(marshall);
		break;
	case AEPC_FIELD_RELATIVE_PRETRIG:
		ch->pretrigger_ms = atoi(marshall);
		if (ch->aep) {
			ch->ring_samples = (ch->pretrigger_ms +
					   ch->aep->aep_context->ms_pretrigger) * 10;
			ch->pretrig_ring = NULL;
			ch->head = 0;
			ch->tail = 0;
			if (ch->ring_samples)
				ch->pretrig_ring = malloc (
				     sizeof(struct samples) *
					     ch->ring_samples);		
		}				
		break;
	case AEPC_FIELD_SUPPLY:
		strncpy(ch->supply, marshall, sizeof(ch->supply) - 1);
		ch->supply[sizeof(ch->supply) - 1] = '\0';
		break;
	case AEPC_FIELD_COLOUR:
		strncpy(ch->colour, marshall, sizeof(ch->colour) - 1);
		ch->colour[sizeof(ch->colour) - 1] = '\0';
		break;
	case AEPC_FIELD_CLASS:
		strncpy(ch->class, marshall, sizeof(ch->class) - 1);
		ch->class[sizeof(ch->class) - 1] = '\0';
		break;
	default:
		break;
	}
}

int configure(struct aep_context *aep_context, struct aep *aep, const char *dev_filepath,
	const char *config_filepath, struct aep_channel *wch)
{
	char buf[1024];
	char marshall[10];
	int len = 0;
	int pos = 0;
	int more = 1;
	int fd = open(config_filepath, O_RDONLY);
	char c;
	enum actions {
		APCA_GET_NAME,
		APCA_CHECK_DEVPATH,
		APCA_MATCH_DEVPATH,
		APCA_PULL_CHANNELS,
	};
	enum actions actions = APCA_GET_NAME;
	enum parser {
		ACPP_FIRST,
		ACPP_SKIP,
		ACPP_EAT,
		ACPP_SWALLOW_WHITE
	};
	enum parser parser = ACPP_FIRST;
	enum parser entry_parser;
	int stanza = -1;
	int index = 0;
	int copy_pos = 0;
	int field = 0;
	struct aep_channel *ch = NULL;
	int wfd = -1;
	char temp_config[1024];
	char linebuf[1024];
	int no_copy = 0;
	int ret = 0;
	int nope = 0;
	int n;

	if (wch) {
		strncpy(temp_config, config_filepath, sizeof temp_config - 2);
		temp_config[sizeof temp_config - 2] = '\0';
		temp_config[strlen(temp_config) + 1] = '\0';
		temp_config[strlen(temp_config)] = '~';
		unlink(temp_config);
		wfd = open(temp_config, O_RDWR | O_CREAT, 0664);
		if (wfd < 0) {
			fprintf(stderr, "Unable to open temp config file %s\n",
								  temp_config);
			return -1;
		}
	}

	if (aep)
		ch = &aep->ch[0];
	else {
		if (aep_context->count_virtual_channels >= MAX_VIRTUAL_CHANNELS)
			return -1;
		ch = &aep_context->vch[aep_context->count_virtual_channels];
		for (n = 0; n < MAX_PROBES; n++)
			aep_context->device_paths[n][0] = '\0';
	}

	if (fd < 0) {
		fprintf(stderr, "unable to open config file %s\n",
				config_filepath);
		return -1;
	}

	while (more) {
		if (pos == len) {
			len = read(fd, buf, sizeof buf);
			if (len <= 0) {
				more = 0;
				continue;
			}
			pos = 0;
		}

		c = buf[pos++];
		entry_parser = parser;

		if (!no_copy && wfd > 0)
			if (write(wfd, &c, 1) < 1) {
				fprintf(stderr, "Unable to write config file\n");
				goto bail;
			}

		switch (parser) {
		case ACPP_FIRST:
			if (c == '#') {
				parser = ACPP_SKIP;
				continue;
			}
			if (c != '\n')
				parser = ACPP_EAT;
			break;
		case ACPP_SKIP:
			if (c == '\n')
				parser = ACPP_FIRST;
			continue;
		case ACPP_EAT:
			if (c == '\n')
				parser = ACPP_FIRST;
			break;
		case ACPP_SWALLOW_WHITE:
			if (c == '\n') {
				parser = ACPP_FIRST;
				continue;
			}
			if (c != ' ' && c != '\t')
				parser = ACPP_EAT;
			else
				continue;
			break;
		}


		switch (actions) {
		case APCA_GET_NAME:
			if (parser == ACPP_FIRST) {
				aep_context->configuration_name[copy_pos] = '\0';
				copy_pos = 0;
				actions = APCA_CHECK_DEVPATH;
//				if (aep == NULL)
//					return 0;
			} else
				if (copy_pos < (sizeof aep_context->configuration_name) - 1)
					aep_context->configuration_name[copy_pos++] = c;
			break;
		case APCA_CHECK_DEVPATH:
			if (entry_parser == ACPP_FIRST) {
				copy_pos = 0;
				nope = 0;
				if (c != '/') {
					if (c != '\n')
						parser = ACPP_SKIP;
					continue;
				}
				aep_context->device_paths[++stanza][copy_pos++] = c;
				aep_context->device_paths[stanza][copy_pos] = '\0';
				actions = APCA_MATCH_DEVPATH;
			}
			break;

		case APCA_MATCH_DEVPATH:
			if (parser == ACPP_FIRST) {
				if (dev_filepath[copy_pos] != '\0' || nope) {
					parser = ACPP_SKIP;
					actions = APCA_CHECK_DEVPATH;
					copy_pos = 0;
					break;
				}
				actions = APCA_PULL_CHANNELS;
				copy_pos = 0;
				no_copy = ch == wch;
				break;
			}
			aep_context->device_paths[stanza][copy_pos] = c;
			if (dev_filepath[copy_pos++] != c)
				nope = 1;
			aep_context->device_paths[stanza][copy_pos] = '\0';
			break;

		case APCA_PULL_CHANNELS:
			if (entry_parser == ACPP_FIRST) {
				if (c != '\t' && c !=' ') {
					more = 0;
					continue;
				}
				parser = ACPP_SWALLOW_WHITE;
				copy_pos = 0;
				field = 0;
				break;
			}

			if (parser == ACPP_FIRST) {
				if (ch != wch)
					process_token(ch, marshall, field);

				index++;
				if (ch == wch) {
					sprintf(linebuf,
					  " %s\t%f\t%d\t%f\t%f\t%f\t%f\t%d\t%s\t%s\t%s\t%s\n",
						ch->channel_name,
						ch->rshunt,
						ch->channel_num,
						ch->voffset[0],
						ch->vnoise[0],
						ch->voffset[1],
						ch->vnoise[1],
						ch->pretrigger_ms,
						ch->channel_name_pretty,
						ch->supply,
						ch->colour,
						ch->class
					);
					if (aep->aep_context->verbose)
						fprintf(stderr, "Updating config "
						     "with \"%s\"\n", linebuf);
					if (write(wfd, linebuf, strlen(linebuf)) < 0) {
						fprintf(stderr, "Unable to write config file\n");
						goto bail;
					}
				}

				ch->flag_was_configured = 1;
				ch++;
				if (aep) {
					if (index == CHANNELS_PER_PROBE) {
						more = 0;
						continue;
					}
				} else {
					aep_context->count_virtual_channels++;
					if (aep_context->count_virtual_channels >= MAX_VIRTUAL_CHANNELS) {
						more = 0;
						continue;
					}
				}
				no_copy = ch == wch;
			}

			if (c == '\t' || c == ' ' || c == '\0') {

				if (ch != wch)
					process_token(ch, marshall, field);

				field++;
				parser = ACPP_SWALLOW_WHITE;
				copy_pos = 0;
				break;
			}

			if (no_copy)
				break;

			switch (field) {
			case AEPC_FIELD_NAME:
				if (copy_pos >= sizeof ch->channel_name - 1)
					break;
				ch->channel_name[copy_pos++] = c;
				ch->channel_name[copy_pos] = '\0';
				break;
			case AEPC_FIELD_PRETTY_NAME:
				if (copy_pos >=
					    sizeof ch->channel_name_pretty - 1)
					break;
				ch->channel_name_pretty[copy_pos++] = c;
				ch->channel_name_pretty[copy_pos] = '\0';
				break;
			default:
				if (copy_pos >= sizeof marshall - 1)
					break;
				marshall[copy_pos++] = c;
				marshall[copy_pos] = '\0';
				break;
			}
			break;
		}
	}

	if (wfd > 0) {
		more = len > 0;
		while (more) {
			if (pos == len) {
	                        len = read(fd, buf, sizeof buf);
	                        if (len <= 0) {                                      
	                                more = 0;
	                                continue;                                       
				}                                                       
				pos = 0;
			}

			c = buf[pos++];
			if (write(wfd, &c, 1) < 0) {
				fprintf(stderr, "Unable to write config file\n");
				goto bail;
			}
		}
	}
bail:
	close(fd);

	if (wfd > 0) {
		close(wfd);
		if (!ret) {
			unlink(config_filepath);
			rename(temp_config, config_filepath);
		}
	}

	return ret;
}


