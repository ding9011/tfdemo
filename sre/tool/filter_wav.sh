#!/bin/bash

set -e

if [ $# -ne 1 ]; then
	echo "Usage: $0 <wavDir:string>"
	exit 1;
fi

filterDir=$1

SampleRate=16000
Duration=1.8

for x in `find ${filterDir} -type f | grep -E "*.wav$" | sort -u -k1`
do
	sample_rate=`soxi ${x} | grep "Sample Rate" | awk -F' ' '{print $4}'`
	duration=`soxi ${x} | grep "Duration" | awk -F":" '{print $4}' | awk -F" " '{print $1}'`
	if [ ${SampleRate} -ne ${sample_rate} ] || [ `echo ${duration} - ${Duration} | bc -q | grep -E "^-"` ];then
		echo "remove ${x}, with sample rate: ${sample_rate}, duration: ${duration}"
		rm ${x} > /dev/null
	fi
done

find ${filterDir} -type d -empty | xargs -i rm -rf {}
