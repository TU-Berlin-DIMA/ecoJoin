num=1000
for ((i=1; i<12; i++)); do
	echo ./bin/hellsJoin_batch data/list1_${num}.csv data/list_${num}.csv ${num} ${num} 100
	#time ./bin/hellsJoin_batch data/list1_${num}.csv data/list_${num}.csv ${num} ${num} 100
	time ./bin/hellsJoin_file data/list1_${num}.csv data/list_${num}.csv ${num} ${num} 100

	# prof
	#sudo /usr/local/cuda-10.0/bin/nvprof --export-profile export_${num}.prof ./hellsJoin_file list1_${num}.csv list_${num}.csv ${num} ${num}
	let num=num*2
done
