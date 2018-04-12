#!/bin/bash

set -e
set -x

if [ $# -ne 5 ]; then
	echo "Notice: need source kaldi path.sh first.\n"
	echo "Usage: $0 <local_dir:string> <dvector.ark:string> <utt2spk:string> <model_path:string> <trials:(yes|no)>"
	exit 1;
fi

local_dir=$1
dvector=$2
utt2spk=$3
model_path=$4
trials_symbol=$5
transform_mat=${model_path}/transform.mat
plda=${model_path}/plda
mean_vec=${model_path}/mean.vec


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

echo "" > eer.txt

cat ${trials} | awk '{print $1, $2}' | \
	ivector-compute-dot-products - "ark:ivector-normalize-length ark:${dvector} ark:- |" \
	"ark:ivector-normalize-length ark:${dvector} ark:- |" \
	dot_products_score
eer=`compute-eer <(python ${local_dir}/prepare_for_eer.py ${trials} dot_products_score) 2> dot_product.log`
echo "dot products eer: ${eer}" >> eer.txt
echo dot products eer: ${eer}

cat ${trials} | awk '{print $1, $2}' | \
	ivector-compute-dot-products - \
	"ark:ivector-normalize-length ark:${dvector} ark:- | ivector-transform ${transform_mat} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	"ark:ivector-normalize-length ark:${dvector} ark:- | ivector-transform ${transform_mat} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	LDA_score
eer=`compute-eer <(python ${local_dir}/prepare_for_eer.py ${trials} LDA_score) 2> LDA.log`
echo "LDA eer: ${eer}" >> eer.txt
echo LDA eer: ${eer}

ivector-plda-scoring --normalize-length=true \
	"ivector-copy-plda --smoothing=0.0 ${plda} - |" \
	"ark:ivector-subtract-global-mean ${mean_vec} ark:${dvector} ark:- | transform-vec ${transform_mat} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	"ark:ivector-subtract-global-mean ${mean_vec} ark:${dvector} ark:- | transform-vec ${transform_mat} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	"cat '${trials}' | cut -d\  --fields=1,2 |" plda_score
eer=`compute-eer <(python ${local_dir}/prepare_for_eer.py ${trials} plda_score) 2> plda.log`
echo "plda eer: ${eer}" >> eer.txt
echo plda eer: ${eer}
