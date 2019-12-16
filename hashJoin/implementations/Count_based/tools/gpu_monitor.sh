echo "gpu load, pwr main, pwr gpu, pwr cpu, time" > monitor.csv

while :
do
	cat /sys/devices/57000000.gpu/load | tr -d '\n\r' >> monitor.csv
	echo -n ", " >> monitor.csv
	cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power0_input | tr -d '\n\r' >> monitor.csv
	echo -n ", " >> monitor.csv
	cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power1_input | tr -d '\n\r' >> monitor.csv
	echo -n ", " >> monitor.csv
	cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power2_input | tr -d '\n\r' >> monitor.csv
	echo -n ", " >> monitor.csv
	date +%H:%M:%S:%N | tr -d '\n\r' >> monitor.csv
	echo "" >> monitor.csv
	sleep 0.1
done
