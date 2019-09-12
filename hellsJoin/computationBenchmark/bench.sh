num=1
file=1024000
for ((i=1; i<7; i++)); do
	echo ./computationBenchmark test_data/list1_${file}.csv test_data/list_${file}.csv ${num}
	time ./computationBenchmark test_data/list1_${file}.csv test_data/list_${file}.csv ${num}
	let num=num*10
done
