#!/bin/bash

set -e
set -x

if [ $# -ne 5 ]; then
	echo "Notice: need source kaldi path.sh first.\n"
	echo "Usage: $0 <utt2spk:string> <plda:string> <dvector.ark:string> <mean.vec:string> <trials:(yes|no)>"
	exit 1;
fi

utt2spk=$1
plda=$2
dvector=$3
mean=$4
trials_symbol=$5

trials=tmp_trials
if [[ ${trials_symbol}'x' == 'yesx' ]];then
	[ -f ${trials} ] && rm ${trials}
	for i in `awk '{print $1}' "${utt2spk}"`
	do
		for j in `awk '{print $1}' "${utt2spk}"`
		do
			i1=`echo ${i}| awk -F"_" '{print $1}'`
			j1=`echo ${j}| awk -F"_" '{print $1}'`
			if [[ "${j1}"x == "${i1}"x ]];then
				echo "${i} ${j} target" >> ${trials}
			else
				echo "${i} ${j} nontarget" >> ${trials}
			fi
		done
	done
fi

echo "finish trials"

ivector-plda-scoring --normalize-length=true \
    --simple-length-normalization=false \
    "ivector-copy-plda --smoothing=0.0 ${plda} - |" \
    "ark:ivector-subtract-global-mean ${mean} ark:${dvector} ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length ark:${dvector} ark:- | ivector-subtract-global-mean ${mean} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '${trials}' | cut -d\  --fields=1,2 |" plda_scores || exit 1;

eer=`compute-eer <(python local/prepare_for_eer.py ${trials} plda_scores) 2> plda.log`
echo "plda eer: ${eer}" >> plda_eer.txt
echo plda eer: ${eer}
