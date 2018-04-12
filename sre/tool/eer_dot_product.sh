#!/bin/bash

set -e
set -x

if [ $# -ne 4 ]; then
	echo "Notice: need source kaldi path.sh first.\n"
	echo "Usage: $0 <utt2spk:string> <dvector.ark:string> <local_dir:string> <trials:(yes|no)>"
	exit 1;
fi

utt2spk=$1
dvector=$2
local_dir=$3
trials_symbol=$4

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

cat ${trials} | awk '{print $1, $2}' | \
	ivector-compute-dot-products - "ark:ivector-normalize-length ark:${dvector} ark:- |" \
	"ark:ivector-normalize-length ark:${dvector} ark:- |" \
	dot_product_score

eer=`compute-eer <(python ${local_dir}/prepare_for_eer.py ${trials} dot_product_score) 2> dot_product.log`
echo "dot product eer: ${eer}" >> eer_dot_product.txt
echo dot product eer: ${eer}
