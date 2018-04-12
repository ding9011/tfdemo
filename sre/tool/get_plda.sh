#!/bin/bash

set -e
set -x

if [ $# -ne 3 ]; then
	echo "Notice: need source kaldi path.sh first and abspath.\n"
	echo "Usage: $0 <train_dvector_ark:string> <train_spk2utt:string> <out_plda:string>"
	exit 1;
fi

train_dvector_ark=$1
train_spk2utt=$2
out_plda=$3

ivector-compute-plda ark:${train_spk2utt} "ark:ivector-subtract-global-mean ark:${train_dvector_ark} ark:- | transform-vec transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" ${out_plda}
