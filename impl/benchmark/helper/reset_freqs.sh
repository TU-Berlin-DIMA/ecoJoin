sudo echo "schedutil" >  /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
sudo echo 1428000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
sudo echo 102000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
sudo echo 76800000 > /sys/devices/gpu.0/devfreq/57000000.gpu/min_freq
sudo echo 921600000 > /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq
