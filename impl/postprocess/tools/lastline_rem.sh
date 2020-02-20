#!/bin/bash
for f in $1/*
do
	echo $f
	if [[ "$f" = *"exp"* ]]; then
		echo $f
		sed '$ d' $f > temp.txt ; mv temp.txt $f
	fi
done
