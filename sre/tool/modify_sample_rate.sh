#!/bin/bash

set -e

if [ $# -ne 2 ]; then
	echo "Usage: $0 <wavdir:string> <target_sample_rate:string>"
	exit 1;
fi


KALDIROOT=~/kaldi

#VOICEFILECONVERT=${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe

VOICEFILEDIR=$1
SAMPLERATE=$2

for x in `find ${VOICEFILEDIR} -type f | grep -E "*.wav$"`;
do
	filepath=`dirname ${x}`
	this_sample_rate=`soxi ${x} | grep "Sample Rate" | grep -o "[0-9]*"`
	if [ ${this_sample_rate}'x' != ${SAMPLERATE}'x' ];then
		echo ${x}
		echo "sample rate:"${this_sample_rate}
		$(sox ${x} -r ${SAMPLERATE} tmp.wav)
		$(rm ${x})
		$(mv tmp.wav ${x})
	fi
done


