num=1
file=1024000
etpw=1000
for ((i=1; i<7; i++)); do
	echo ../onsimpletest/bin/hellsJoin_cpu test_data/list1_${file}.csv test_data/list_${file}.csv ${etpw} ${num}
	time ../onsimpletest/bin/hellsJoin_cpu test_data/list1_${file}.csv test_data/list_${file}.csv ${etpw} ${num}
	let num=num*10
done
