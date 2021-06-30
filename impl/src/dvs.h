#ifndef DVF_H
#define DVF_H
#include "master.h"

void set_freq(frequency_mode_e mode, unsigned id_cpu, unsigned id_gpu);

void set_min_freq(frequency_mode_e mode);
void set_max_freq(frequency_mode_e mode);

#endif /* DVF_H */
