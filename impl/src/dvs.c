#include <cpufreq.h>

char my_governer[] = {"userspace"};

long available_freqs[] ={102000, 204000, 307200, 403200, 518400, 
	614400, 710400, 825600, 921600, 1036800, 1132800, 1224000, 1326000, 1428000};

#define min_freq 102000
#define max_freq 1428000

void set_max_freq(){
                cpufreq_modify_policy_governor(0, my_governer);

		cpufreq_modify_policy_min(0, max_freq);
                cpufreq_modify_policy_max(0, max_freq);
		cpufreq_set_frequency(0, max_freq);
}

void set_min_freq(){
                cpufreq_modify_policy_governor(0, my_governer);

		cpufreq_modify_policy_min(0, min_freq);
                cpufreq_modify_policy_max(0, min_freq);

		cpufreq_set_frequency(0, min_freq);
}
