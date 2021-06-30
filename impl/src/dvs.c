#include "dvs.h"
#include <cpufreq.h>
#include <fstream>
#include <iostream>
#include <string>

void set_gpu_freq_to_max();
void set_gpu_freq_to_min();
void set_cpu_freq_to_max();
void set_cpu_freq_to_min();

void set_cpu_freq(unsigned id);
void set_gpu_freq(unsigned id);

char my_governer[] = { "userspace" };

long available_cpu_freqs[] = { 102000, 204000, 307200, 403200, 518400,
    614400, 710400, 825600, 921600, 1036800, 1132800, 1224000, 1326000, 1428000 };

long available_gpu_freqs[] = { 76800000, 153600000, 230400000, 307200000, 384000000, 460800000, 537600000, 614400000, 691200000, 768000000, 844800000, 921600000 };

#define min_cpu_freq 102000
#define max_cpu_freq 1428000
#define min_gpu_freq 76800000
#define max_gpu_freq 921600000

void set_freq(frequency_mode_e mode, unsigned id_cpu, unsigned id_gpu)
{
    if (mode == cpu) {
        set_cpu_freq(id_cpu);
    } else if (mode == gpu) {
        set_gpu_freq(id_gpu);
    } else if (mode == both) {
        set_cpu_freq(id_cpu);
        set_gpu_freq(id_gpu);
    } else {
        std::cout << "no mode is given\n";
    }
}

void set_cpu_freq(unsigned id)
{
    cpufreq_modify_policy_governor(0, my_governer);

    cpufreq_modify_policy_min(0, available_cpu_freqs[id]);
    cpufreq_modify_policy_max(0, available_cpu_freqs[id]);
    cpufreq_set_frequency(0, available_cpu_freqs[id]);
}

void set_gpu_freq(unsigned id)
{
    std::ofstream myfile;
    myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq");
    myfile << std::to_string(available_gpu_freqs[id]);
    myfile.close();
    myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq");
    myfile << std::to_string(available_gpu_freqs[id]);
    myfile.close();
}

void set_freq_to_max(frequency_mode_e dev)
{
    if (dev == cpu) {
        set_cpu_freq_to_max();
    } else if (dev == gpu) {
        set_gpu_freq_to_max();
    } else {
        set_cpu_freq_to_max();
        set_gpu_freq_to_max();
    }
}

void set_freq_to_min(frequency_mode_e dev)
{
    if (dev == cpu) {
        set_cpu_freq_to_min();
    } else if (dev == gpu) {
        set_gpu_freq_to_min();
    } else {
        set_cpu_freq_to_min();
        set_gpu_freq_to_min();
    }
}

void set_cpu_freq_to_max()
{
    cpufreq_modify_policy_governor(0, my_governer);

    cpufreq_modify_policy_min(0, max_cpu_freq);
    cpufreq_modify_policy_max(0, max_cpu_freq);
    cpufreq_set_frequency(0, max_cpu_freq);
}

void set_cpu_freq_to_min()
{
    cpufreq_modify_policy_governor(0, my_governer);

    cpufreq_modify_policy_min(0, min_cpu_freq);
    cpufreq_modify_policy_max(0, min_cpu_freq);

    cpufreq_set_frequency(0, min_cpu_freq);
}

void set_gpu_freq_to_max()
{
    std::ofstream myfile;
    myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq");
    myfile << std::to_string(max_gpu_freq);
    myfile.close();
    myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq");
    myfile << std::to_string(max_gpu_freq);
    myfile.close();
}

void set_gpu_freq_to_min()
{
    std::ofstream myfile;
    myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/max_freq");
    myfile << std::to_string(min_gpu_freq);
    myfile.close();
    myfile.open("/sys/devices/gpu.0/devfreq/57000000.gpu/min_freq");
    myfile << std::to_string(min_gpu_freq);
    myfile.close();
}
