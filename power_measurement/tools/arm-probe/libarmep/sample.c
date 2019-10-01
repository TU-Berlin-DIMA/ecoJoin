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

int somebody_triggered = -1;
struct aep_channel *who_triggered = NULL;

static void track_limits_convert_to_current(struct aep_channel *ch,
							double *v1, double *v2)
{
	if ((*v1 - ch->voffset[0]) < ch->min[0])
		ch->min[0] = *v1 - ch->voffset[0];

	if ((*v1 - ch->voffset[0]) > ch->max[0])
		ch->max[0] = *v1 - ch->voffset[0];

	if ((*v2 - ch->voffset[1]) < ch->min[1])
		ch->min[1] = *v2 - ch->voffset[1];

	if ((*v2 - ch->voffset[1]) > ch->max[1])
		ch->max[1] = *v2 - ch->voffset[1];

	if (!ch->aep->aep_context->auto_zero) {
		if (*v1 < 0) {
			if (ch->samples_seen > IGNORE_NEG_PRIOR_TO_SAMPLES &&
						  -(*v1) > ch->vnoise[0] / 2) {
				if (ch->aep->aep_context->verbose)
					fprintf(stderr, "*** reducing voltage "
						"baseline by %fV to %fV\n",
						*v1, ch->voffset[0] - *v1);
				ch->voffset[0] -= *v1 *
					NEGATIVE_ADJUST_SCALING_FACTOR;
			}
			*v1 = 0;
		}
		if (*v2 < 0) {
			if (ch->samples_seen > IGNORE_NEG_PRIOR_TO_SAMPLES &&
						  -(*v2) > ch->vnoise[1] / 2) {
				if (ch->aep->aep_context->verbose)
					fprintf(stderr, "*** reducing shunt "
						"voltage baseline by "
						"%fV to %fV\n", *v2,
						ch->voffset[1] - *v2);
				ch->voffset[1] -= *v2 *
					NEGATIVE_ADJUST_SCALING_FACTOR;
			}
			*v2 = 0;
		}
	}

	*v2 = correct(ch->aep->aep_context->no_correction, *v1, *v2, ch->map_table) / ch->rshunt;
}

int process_sample(struct aep *aep, int ch_index)
{
	struct aep_channel *ch = &aep->ch[ch_index], *ich;
	double v1, v2, v1a, v2a, vo0 = 0, vo1 = 0;
	int m, i, from_pretrig = 0, n, hit;
	struct aepd_interface *aepd_interface;
	struct aepd_interface_result *aepd_interface_result;

	if (!ch->flag_was_configured)
		goto done;

	if (!aep->aep_context->no_correction) {
		vo0 = ch->voffset[0];
		vo1 = ch->voffset[1];
	}

	aepd_interface = aep->aep_context->aepd_interface;
	if (aepd_interface && aepd_interface->auto_zero) {
		fprintf(stderr, "setting inline autozero\n");
		aepd_interface->auto_zero = 0;
		aep->aep_context->auto_zero = 2;

		aep->aep_context->no_correction = 1;
		aep->aep_context->mv_min = 0;
		aep->aep_context->mw_min = 0;
		aep->aep_context->average_len = 5000;
		aep->aep_context->ms_capture = aep->aep_context->average_len / 10;
		aep->aep_context->decimate = aep->aep_context->average_len;
		aep->aep_context->done_az_channels = 0;
	}

	if (ch->pretrig_ring) {
		ch->pretrig_ring[ch->head].v = aep->voltage;
		ch->pretrig_ring[ch->head].i = aep->current;
		ch->head++;
		if (ch->head == ch->ring_samples)
			ch->head = 0;
		if (ch->head == ch->tail)
			ch->tail++;
		if (ch->tail == ch->ring_samples)
			ch->tail = 0;

		if (ch->pretrigger_samples_taken < ch->ring_samples)
			ch->pretrigger_samples_taken++;
	}

	if ((ch->triggered && aep->voltage < aep->aep_context->mv_min && ch->samples > 1000 && !ch->trigger_slave) ||
		(ch->trigger_slave && !ch->trigger_slave->triggered)) {

		ch->trigger_filter++;
		if (ch->trigger_filter < aep->aep_context->end_trigger_filter_us / 100)
			goto unripe;

		if (aep->aep_context->verbose)
			fprintf(stderr, "trigger session completed "
				"%dmV vs %dmV\n", aep->voltage, aep->aep_context->mv_min);
		avg_mean_us_flush(&ch->avg_mean_voltage);
		avg_mean_us_flush(&ch->avg_mean_current);
		// ch->ignore = 0;
		ch->triggered = 0;
		ch->trigger_filter = 0;
		ch->pretrigger_samples_taken = 0;
		ch->head = 0;
		ch->tail = 0;
		if (aep->aep_context->require_off) {
			if (aep->aep_context->verbose)
				fprintf(stderr, "marking capture for "
					"device %d done\n", aep->index);
			aep->aep_context->awaiting_capture &= ~(1 << (aep->index * 3 + ch_index));
		}
		return 0;
	} else
		if (ch->triggered)
			ch->trigger_filter = 0;
unripe:

	ch->samples_seen++;

	if (ch->ignore)
		return 0;

	if (!ch->triggered) {
		avg_mean_us_add(&ch->avg_mean_voltage, aep->voltage);
		avg_mean_us_add(&ch->avg_mean_current, aep->current);

		v1 = (avg_mean_us(&ch->avg_mean_voltage) /
						ADC_COUNTS_PER_VOLT_CH1) + vo0;
		v2 = (avg_mean_us(&ch->avg_mean_current) /
						ADC_COUNTS_PER_VOLT_CH2) + vo1;
		if ((!(ch->samples & 0x1ff)) && aep->aep_context->show_raw)
			fprintf(stderr, "%fmV raw shunt;  corr=%fmV\n", v2,
				correct(aep->aep_context->no_correction, v1, v2, ch->map_table));


		track_limits_convert_to_current(ch, &v1, &v2);

		if ((((aep->voltage > aep->aep_context->mv_min + TRIG_HYSTERESIS_MV) || !aep->aep_context->mv_min) &&
		      (((v1 * v2) > (aep->aep_context->mw_min_plus_hyst) / 1000.0) || !aep->aep_context->mw_min)) ||
			(somebody_triggered != -1 && ch->samples_seen >= somebody_triggered)) {

			ch->trigger_filter++;
			if (ch->trigger_filter > aep->aep_context->trigger_filter_us / 100 || somebody_triggered != -1) {

				if (somebody_triggered != -1)
					ch->trigger_slave = who_triggered;
				else {
					who_triggered = ch;
					somebody_triggered = ch->samples_seen;
				}
				ch->triggered = 1;
				ch->simple_avg[0] = 0;
				ch->simple_avg[1] = 0;
				ch->simple_avg[2] += 0;
				ch->avg_count = 0;
				ch->samples = 0;
				ch->trigger_filter = 0;
				if (aep->aep_context->verbose)
					fprintf(stderr, "%s: triggered "
						"%.2fV %.5fW pretrig %d "
						"(%dms)\n",
						ch->channel_name, v1,
						v1 * v2,
						ch->ring_samples,
						ch->ring_samples / 10);
			}
		} else
			ch->trigger_filter = 0;

		aep->rx_count++;
	}

	if (!ch->samples && aep->aep_context->ms_holdoff)
		fprintf(stderr, "%s: post-trigger holdoff for"
				     " %dms\n", ch->channel_name, aep->aep_context->ms_holdoff);

	if (ch->triggered) {
		ch->samples++;

		if (ch->samples / 10 < aep->aep_context->ms_holdoff)
			return 0;
		if (ch->samples - 1 == aep->aep_context->ms_holdoff * 10 && aep->aep_context->verbose) 
			fprintf(stderr, "%s: Starting capture\n", ch->channel_name);
	}

	if (ch->pretrig_ring) {
		from_pretrig = 1;
		avg_mean_us_add(&ch->avg_mean_voltage,
						ch->pretrig_ring[ch->tail].v);
		avg_mean_us_add(&ch->avg_mean_current,
						ch->pretrig_ring[ch->tail].i);
		ch->tail++;
		if (ch->tail == ch->ring_samples)
			ch->tail = 0;
	} else {
		avg_mean_us_add(&ch->avg_mean_voltage, aep->voltage);
		avg_mean_us_add(&ch->avg_mean_current, aep->current);
	}
	v1 = (avg_mean_us(&ch->avg_mean_voltage) /
						ADC_COUNTS_PER_VOLT_CH1) + vo0;
	v2 = (avg_mean_us(&ch->avg_mean_current) /
						ADC_COUNTS_PER_VOLT_CH2) + vo1;

	if ((!(ch->samples & 0x1ff)) && aep->aep_context->show_raw)
		fprintf(stderr, "%fmV raw shunt;  corr=%fmV\n", v2,
		      correct(aep->aep_context->no_correction, v1, v2, ch->map_table));

	track_limits_convert_to_current(ch, &v1, &v2);

	if (aep->aep_context->auto_zero && v1 > REJECT_AUTOZERO_VOLTAGE) {
		fprintf(stderr, "**** Autozero seeing > %fV... "
			"probe should not be in powered target for "
			"autozero! ****\n", REJECT_AUTOZERO_VOLTAGE);
		return -2;
	}

	if (aep->aep_context->auto_zero) {
		ch->simple_avg[0] += v1;
		ch->simple_avg[1] += v2;
		ch->simple_avg[2] += v1 * v2;
		ch->avg_count++;
	}

	ch->decimation_counter++;
	if (ch->decimation_counter != aep->aep_context->decimate)
		goto done;
	ch->decimation_counter = 0;

	if (aep->aep_context->auto_zero) {
		if (v1 > IGNORE_AUTOZERO_VOLTAGE || v1 < -IGNORE_AUTOZERO_VOLTAGE)
			fprintf(stderr, "   Voltage Autozero skipped as %fV "
				"unfeasibly high\n", v1);
		else {
			ch->voffset[0] = -v1;
			ch->vnoise[0] = ch->max[0] - ch->min[0];
		}
		ch->voffset[1] = -avg_mean_us(&ch->avg_mean_current) /
						ADC_COUNTS_PER_VOLT_CH2;
		ch->vnoise[1] = ch->max[1] - ch->min[1];
		ch->samples = 0;

		fprintf(stderr, "Autozero for %s done (voltage %f)  \n", ch->channel_name, ch->voffset[0]);

		if (configure(aep->aep_context, aep, aep->dev_filepath, aep->aep_context->config_filepath, ch) < 0)
			fprintf(stderr,
				"Failed to write autozero info to config\n");
		aep->aep_context->done_az_channels++;
		fprintf(stderr, "done %d autozero channels of %d\n", aep->aep_context->done_az_channels, aep->aep_context->count_channel_names);
		if (aep->aep_context->done_az_channels == aep->aep_context->count_channel_names) {

			if (aep->aep_context->auto_zero == 2) { /* as part of larger collection */
				aep->aep_context->auto_zero = 0;
				aep->aep_context->no_correction = 0;
				aep->aep_context->mv_min = 0;
				aep->aep_context->mw_min = 0;
				aep->aep_context->average_len = 1;
				aep->aep_context->ms_capture = 0;
				aep->aep_context->decimate = 1;
			} else
				goto done;
		}
	}
	
	/* not everyone else may be ready, save the data first in per-channel sync ring buffers */

	if (v1 < 0)
		v1 = 0;
	if (v2 < 0)
		v2 = 0;
	ch->out_ring[0][ch->out_head] = v1;
	ch->out_ring[1][ch->out_head++] = v2;
	if (ch->out_head == MAX_RESULT_QUEUE_DEPTH)
		ch->out_head = 0;
	if (ch->out_head == ch->out_tail)
		fprintf(stderr, "\n***** %s output ring overflow...%d %d\n",
			ch->channel_name, ch->out_head, ch->out_tail);

	/* the other capture channels have data for this sample? */

	hit = 0;
	for (m = 0; m <= aep->aep_context->highest; m++) {
		if (!aep->aep_context->aeps[m].fd)
			continue;
		for (i = 0; i < CHANNELS_PER_PROBE; i++) {
			if (!aep->aep_context->aeps[m].ch[i].flag_was_configured)
				continue;
			if (aep->aep_context->aeps[m].ch[i].out_head ==
					aep->aep_context->aeps[m].ch[i].out_tail)
				goto done;
			hit++;
		}
	}

	/* if so, output it all together */

	aepd_interface = aep->aep_context->aepd_interface;
	aepd_interface_result = &aepd_interface->aepd_interface_result[aepd_interface->head];

	aepd_interface_result->triggered = ch->triggered;
	aepd_interface_result->chans = hit;
	aepd_interface_result->samtime = ((double)ch->samples_seen - (double)ch->ring_samples) / 10000.0;

	n = 0;

	for (m = 0; m <= aep->aep_context->highest; m++) {
		if (!aep->aep_context->aeps[m].fd)
			continue;
		for (i = 0; i < CHANNELS_PER_PROBE; i++) {

			ich = &aep->aep_context->aeps[m].ch[i];
			if (!ich->flag_was_configured)
				continue;

			v1a = ich->out_ring[0][ich->out_tail];
			v2a = ich->out_ring[1][ich->out_tail];
			ich->out_tail++;
			if (ich->out_tail == MAX_RESULT_QUEUE_DEPTH)
				ich->out_tail = 0; 
			if (ch->samples >= ch->ring_samples && !from_pretrig) {
				ich->simple_avg[0] += v1a;
				ich->simple_avg[1] += v2a;
				ich->simple_avg[2] += v1a * v2a;
				ich->avg_count++;
			}
			if (!ich->requested)
				continue;
	
			aepd_interface_result->buf[n++] = v1a;
			aepd_interface_result->buf[n++] = v2a;
		}
	}

	if (aepd_interface->head == (sizeof(aepd_interface->aepd_interface_result) / sizeof(aepd_interface->aepd_interface_result[0])) - 1)
		aepd_interface->head = 0;
	else
		aepd_interface->head++;

	if (sem_post(aep->aep_context->aepd_interface->semaphore) < 0)
		fprintf(stderr, "failed to set semaphore\n");

done:

	if (aep->aep_context->auto_zero != 2 && aep->aep_context->ms_capture && (ch->samples / 10 - aep->aep_context->ms_holdoff) > aep->aep_context->ms_capture) {
		fprintf(stderr, "%s: %dms capture complete  ",
				    ch->channel_name, aep->aep_context->ms_capture);
		ch->ignore = 1;
		if (!aep->aep_context->require_off) {
			aep->aep_context->awaiting_capture &= ~(1 << (aep->index * 3 + ch_index));
			fprintf(stderr, "\n");
		} else
			fprintf(stderr, "(waiting for OFF)\n");
	}

	return 0;
}



