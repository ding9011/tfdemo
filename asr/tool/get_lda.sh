#!/bin/bash

set -e
set -x

if [ $# -ne 4 ]; then
	echo "Notice: need source kaldi path.sh first and abspath.\n"
	echo "Usage: $0 <train_dvector_ark:string> <train_utt2spk:string> <dim:int> <out_mat:string>"
	exit 1;
fi

train_dvector_ark=$1
train_utt2spk=$2
dim=$3
out_mat=$4

ivector-compute-lda --dim=${dim} --total-covariance-factor=0.0 "ark:ivector-normalize-length ark:${train_dvector_ark} ark:- |" ark:${train_utt2spk} ${out_mat}
