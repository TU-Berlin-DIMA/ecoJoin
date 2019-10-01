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

int aep_protocol_parser(struct aep *aep, int samples)
{
	unsigned char v;
	static unsigned char cmd[] = {
		AEPC_CONFIG, 0, 6,
	};
	int bad = 0;

	while (aep->head != aep->tail && samples--) {

		v = aep->ring[aep->head++];
		if (aep->head == sizeof(aep->ring))
			aep->head = 0;

		if (aep->aep_context->verbose)
			fprintf(stderr, "%02X ", v);

		switch (aep->state) {

		/*
		 * this filters the input to sync, it will keep returning to
		 * look for the first byte if any of AC FF FF FF FF FF FF FF FF
		 * is not correctly seen.
		 */

		case APP_INIT_MAGIC:
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					aep->counter = 0;
				break;
			case 1:
			case 2:
			case 3:
			case 4:
			case 5:
			case 6:
			case 7:
			case 8:
				if (v != 0xff)
					aep->counter = 0;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
					     "magic ok, asking for version\n");
				v = AEPC_VERSION;
				if (write(aep->fd, &v, 1) != 1)
					return -1;
				aep->state++;
				aep->counter = 0;
				break;
			default:
				break;
			}
			break;

		case APP_INIT_VERSION:
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					return -1;
				break;
			case 1:
				aep->version = v;
				break;
			case 2:
				aep->version |= v << 8;
				break;
			case 3:
				aep->version |= v << 16;
				break;
			case 4:
				aep->version |= v << 24;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
					    "version ok, asking for vendor\n");
				v = AEPC_VENDOR;
				if (write(aep->fd, &v, 1) != 1)
					return -1;
	        		aep->state++;
				aep->counter = 0;
				break;
			default:
				break;
			}
			break;

		case APP_INIT_VENDOR:
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					return -1;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
					       "vendor ok, asking for rate\n");
				v = AEPC_RATE;
				if (write(aep->fd, &v, 1) != 1)
					return -1;
	        		aep->state++;
				aep->counter = 0;
				aep->name[63] = '\0';
				break;
			default:
				aep->name[aep->counter - 2] = v;
				break;
			}
			break;


		case APP_INIT_RATE:
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					return -1;
				break;
			case 1:
				aep->rate = v;
				break;
			case 2:
				aep->rate |= v << 8;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
						    "rate ok, doing config\n");
	        		aep->state++;
				aep->counter = 0;
				cmd[1] = 0;
				if (write(aep->fd, &cmd[0], sizeof cmd) !=
								    sizeof cmd)
					return -1;
				break;
			default:
				break;
			}
			break;

		case APP_INIT_CONFIG_1:
			
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					return -1;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
						    "rate ok, doing config\n");
	        		aep->state++;
				aep->counter = 0;
				cmd[1] = 1;
				if (write(aep->fd, &cmd[0], sizeof cmd) !=
								    sizeof cmd)
					return -1;
				break;
			default:
				break;
			}
			break;

		case APP_INIT_CONFIG_2:
			
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					return -1;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
						    "rate ok, doing config\n");
	        		aep->state++;
				aep->counter = 0;
				cmd[1] = 2;
				if (write(aep->fd, &cmd[0], sizeof cmd) !=
								    sizeof cmd)
					return -1;
				break;
			default:
				break;
			}
			break;



		case APP_INIT_CONFIG_3:
			
			switch (aep->counter++) {
			case 0:
				if (v != 0xac)
					return -1;
				break;
			case 63:
				if (aep->aep_context->verbose)
					fprintf(stderr,
						"Configured new probe instance "
						"%s %s %08X %d\n",
						aep->dev_filepath, aep->name,
						      aep->version, aep->rate);

				aep->done_config |= 2;
				aep->state = APP_INIT_START_ACK;
				aep->counter = 0;
				break;
			default:
				break;
			}
			break;

		case APP_INIT_START_ACK:
			if (v != 0xac)
				return -1;

			if (aep->aep_context->verbose)
				fprintf(stderr,
					"Start acknowledge seen "
					"%s %s %08X %d\n",
					aep->dev_filepath, aep->name,
					      aep->version, aep->rate);

			aep->done_config |= 4;
			aep->state = APP_FRAME_L;
			aep->counter = 0;
			break;

		case APP_FRAME_L:
			if (v == (aep->predicted_frame & 0xff)) {
				aep->state = APP_FRAME_H;
				break;
			}
			fprintf(stderr, "%d:.L(%d %d)", aep->index, v, aep->predicted_frame & 0xff);
			/* lost sync... */
			if (aep->invalid < AEP_INVALID_COVER)
				aep->invalid = AEP_INVALID_COVER;
			aep->state = APP_FRAME_H_BAD;
			break;

		case APP_FRAME_H:

			if (v == aep->predicted_frame >> 8) {
				aep->state = APP_VOLTAGE_L1;
				aep->predicted_frame++;
				if (aep->invalid) {
					aep->invalid--;
					if (aep->invalid == 0)
						fprintf(stderr, "%d:donetime\n", aep->index);
					else
						fprintf(stderr, "-%d:%d-", aep->index, aep->invalid);
				}
				break;
			}

			fprintf(stderr, "%d:.H(%d %d)", aep->index, v, aep->predicted_frame >> 8);

		/* fallthru */

		case APP_FRAME_H_BAD:
			/* lost sync - slip one byte and use that as frame prediction for next time */
			aep->predicted_frame = v;
			if (aep->invalid < AEP_INVALID_COVER)
				aep->invalid = AEP_INVALID_COVER;
			aep->state = APP_VOLTAGE_L1_BAD;
			break;

		case APP_VOLTAGE_L1_BAD:
			aep->predicted_frame |= v << 8;
			aep->predicted_frame++;
			aep->state = APP_VOLTAGE_H1_BAD;
			break;

		case APP_VOLTAGE_H1_BAD:
		case APP_CURRENT_L1_BAD:
		case APP_CURRENT_H1_BAD:
		case APP_VOLTAGE_L2_BAD:
		case APP_VOLTAGE_H2_BAD:
		case APP_CURRENT_L2_BAD:
		case APP_CURRENT_H2_BAD:
		case APP_VOLTAGE_L3_BAD:
		case APP_VOLTAGE_H3_BAD:
		case APP_CURRENT_L3_BAD:
		case APP_CURRENT_H3_BAD:
			aep->state++;
			break;

		case APP_SWALLOW:

			//if (aep->aep_context->verbose)
				fprintf(stderr, "%s: LOST SYNC *\n",
							aep->dev_filepath);
			bad++;
			if (bad > 16) {
				fprintf(stderr, "%s: too much lost sync\n",
					aep->dev_filepath);
				return -1;
			}

			aep->state = APP_FRAME_L;
			break;

		/* good path */

		case APP_VOLTAGE_L1:
			aep->voltage = v;
			aep->state++;
			break;

		case APP_VOLTAGE_H1:
			aep->voltage |= v << 8;
			aep->state++;
			break;

		case APP_CURRENT_L1:
			aep->current = v;
			aep->state++;
			break;

		case APP_CURRENT_H1:
			aep->current |= v << 8;
			aep->state++;

			if (aep->invalid)
				break;

			if (aep->current > ADC_COUNT_CURRENT_CLIP_LIMIT)
				fprintf(stderr, "\n*** %d.0: %s Shunt diff voltage %fV above"
					" %fV limit: clipping ***     \n",
					aep->index, aep->ch[0].channel_name,
					(double)aep->current / ADC_COUNTS_PER_VOLT_CH2,
					(double)ADC_COUNT_CURRENT_CLIP_LIMIT /
					ADC_COUNTS_PER_VOLT_CH2);

			return process_sample(aep, 0);
		case APP_VOLTAGE_L2:
			aep->voltage = v;
			aep->state++;
			break;

		case APP_VOLTAGE_H2:
			aep->voltage |= v << 8;
			aep->state++;
			break;

		case APP_CURRENT_L2:
			aep->current = v;
			aep->state++;
			break;

		case APP_CURRENT_H2:
			aep->current |= v << 8;
			aep->state++;

			if (aep->invalid)
				break;

			if (aep->current > ADC_COUNT_CURRENT_CLIP_LIMIT)
				fprintf(stderr, "\n*** %d.1: %s Shunt diff voltage %fV above"
					" %fV limit: clipping ***     \n",
					aep->index, aep->ch[1].channel_name,
					(double)aep->current / ADC_COUNTS_PER_VOLT_CH2,
					(double)ADC_COUNT_CURRENT_CLIP_LIMIT /
					ADC_COUNTS_PER_VOLT_CH2);

			return process_sample(aep, 1);

		case APP_VOLTAGE_L3:
			aep->voltage = v;
			aep->state++;
			break;

		case APP_VOLTAGE_H3:
			aep->voltage |= v << 8;
			aep->state++;
			break;

		case APP_CURRENT_L3:
			aep->current = v;
			aep->state++;
			break;

		case APP_CURRENT_H3:
			aep->current |= v << 8;
			aep->state = APP_FRAME_L;

			if (aep->invalid) {
				fprintf(stderr, "#%d ", aep->index);
				break;
			}

			if (aep->current > ADC_COUNT_CURRENT_CLIP_LIMIT)
				fprintf(stderr, "\n*** %d.2: %s Shunt diff voltage %fV above"
					" %fV limit: clipping ***     \n",
					aep->index, aep->ch[2].channel_name,
					(double)aep->current / ADC_COUNTS_PER_VOLT_CH2,
					(double)ADC_COUNT_CURRENT_CLIP_LIMIT /
					ADC_COUNTS_PER_VOLT_CH2);

			return process_sample(aep, 2);
		}
	}

	return 0;
}

