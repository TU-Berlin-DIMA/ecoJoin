for FREQ in $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies)
do
	echo "new Freq" $FREQ 
	for CPU in 0 1 2 3 
	do
		echo $FREQ > /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_max_freq
		echo $FREQ > /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_min_freq
		#cat /sys/devices/system/cpu/cpu$CPU/cpufreq/cpuinfo_cur_freq
	done
	~/stress-ng/stress-ng -c4
	read -p "press enter to continue"
done
echo  "Set to default"
/usr/sbin/nvpmodel -m 0
