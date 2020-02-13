echo "gpu load, pwr main, pwr gpu, pwr cpu, time" > sys_monitor.csv

while :
do
	cat /sys/devices/57000000.gpu/load | tr -d '\n\r' >> sys_monitor.csv
	echo -n ", " >> sys_monitor.csv
	cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power0_input | tr -d '\n\r' >> sys_monitor.csv
	echo -n ", " >> sys_monitor.csv
	cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power1_input | tr -d '\n\r' >> sys_monitor.csv
	echo -n ", " >> sys_monitor.csv
	cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power2_input | tr -d '\n\r' >> sys_monitor.csv
	echo -n ", " >> sys_monitor.csv
	date +%H:%M:%S:%N | tr -d '\n\r' >> sys_monitor.csv
	echo "" >> sys_monitor.csv
	sleep 0.1
done
