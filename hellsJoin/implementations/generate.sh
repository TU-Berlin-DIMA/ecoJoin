num=1000
for ((i=1; i<12; i++)); do
	echo $num
	python generate.py list1_${num}.csv ${num}
	python generate.py list_${num}.csv ${num}
	let num=num*2
done
