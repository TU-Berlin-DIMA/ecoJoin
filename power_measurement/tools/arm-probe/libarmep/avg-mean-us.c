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

#include <stdlib.h>
#include <malloc.h>
#include "libarmep.h"
                                                                                
int avg_mean_us_init(struct avg_mean_us *amu, int width)
{
	amu->ring = malloc(width * sizeof(unsigned short));
	if (amu->ring == NULL)
		return 1;
	amu->width = width;
	amu->count = 0;
	amu->next = 0;
	amu->sum = 0;

	return 0;
}
void avg_mean_us_add(struct avg_mean_us *amu, unsigned short s)
{
	if (amu->count == amu->width)
		amu->sum -= amu->ring[amu->next];
	else
		amu->count++;

	amu->ring[amu->next++] = s;
	if (amu->next == amu->width)
		amu->next = 0;
	amu->sum += s;
}

double avg_mean_us(struct avg_mean_us *amu)
{
	return (double)amu->sum / (double)amu->count;
}

void avg_mean_us_free(struct avg_mean_us *amu)
{
	free(amu->ring);
}

void avg_mean_us_flush(struct avg_mean_us *amu)
{
	amu->count = 0;
	amu->sum = 0;
	amu->next = 0;
}
