#ifndef DVF_H
#define DVF_H

enum freq_device{GPU, CPU, BOTH}; 
void set_min_freq(freq_device);

void set_max_freq(freq_device);

#endif  /* DVF_H */
