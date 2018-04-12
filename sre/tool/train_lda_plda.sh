#!/bin/bash

set -e
set -x

if [ $# -ne 4 ]; then
	echo "Notice: need source kaldi path.sh first and abspath.\n"
	echo "Usage: $0 <train_dvector_ark:string> <train_utt2spk:string> <train_spk2utt:string> <dim:int>"
	exit 1;
fi

train_dvector_ark=$1
train_utt2spk=$2
train_spk2utt=$3
dim=$4

ivector-mean ark:${train_dvector_ark} mean.vec

ivector-compute-lda --dim=${dim} --total-covariance-factor=0.0 "ark:ivector-subtract-global-mean ark:${train_dvector_ark} ark:- |" ark:${train_utt2spk} transform.mat

ivector-compute-plda ark:${train_spk2utt} "ark:ivector-subtract-global-mean ark:${train_dvector_ark} ark:- | transform-vec transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" plda
