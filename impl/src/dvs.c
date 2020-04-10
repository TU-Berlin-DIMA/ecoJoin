#include <cpufreq.h>
#include <string>
#include <iostream>
#include <fstream>
#include "dvs.h"


void set_max_gpu_freq();
void set_min_gpu_freq();
void set_max_cpu_freq();
void set_min_cpu_freq();

char my_governer[] = {"userspace"};

long available_cpu_freqs[] ={102000, 204000, 307200, 403200, 518400, 
	614400, 710400, 825600, 921600, 1036800, 1132800, 1224000, 1326000, 1428000};

//long available_gpu_freqs[] ={114750000, 216750000, 318750000, 420750000, 522750000, 624750000, 675750000, 828750000, 905250000, 1032750000, 1198500000, 1236750000, 1338750000, 1377000000};

#define min_cpu_freq 102000
#define max_cpu_freq 1428000
#define min_gpu_freq 76800000
#define max_gpu_freq 921600000

void set_max_freq(freq_device dev){
        if (dev == CPU){
                set_max_cpu_freq();
        } else if (dev == GPU){
                set_max_gpu_freq();
        } else {
                set_max_cpu_freq();
                set_max_gpu_freq();
        }
}

void set_min_freq(freq_device dev){
	if (dev == CPU){
		set_min_cpu_freq();
	} else if (dev == GPU){
		set_min_gpu_freq();
	} else {
		set_min_cpu_freq();
		set_min_gpu_freq();
	}
}

void set_max_cpu_freq(){
         cpufreq_modify_policy_governor(0, my_governer);

	cpufreq_modify_policy_min(0, max_cpu_freq);
        cpufreq_modify_policy_max(0, max_cpu_freq);
	cpufreq_set_frequency(0, max_cpu_freq);
}

void set_min_cpu_freq(){
        cpufreq_modify_policy_governor(0, my_governer);

	cpufreq_modify_policy_min(0, min_cpu_freq);
        cpufreq_modify_policy_max(0, min_cpu_freq);

	cpufreq_set_frequency(0, min_cpu_freq);
}

void set_max_gpu_freq(){
	std::ofstream myfile;
	myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq");
	myfile << std::to_string(max_gpu_freq);
	myfile.close();
	myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq");
	myfile << std::to_string(max_gpu_freq);
	myfile.close();
}

void set_min_gpu_freq(){
	std::ofstream myfile;
	myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq");
	myfile << std::to_string(min_gpu_freq);
	myfile.close();
	myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq");
	myfile << std::to_string(min_gpu_freq);
	myfile.close();
}
