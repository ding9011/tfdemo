#!/bin/bash

set -e
set -x

if [ $# -ne 3 ]; then
	echo "Notice: need source kaldi path.sh first and abspath.\n"
	echo "Usage: $0 <dvector_ark:string> <lda_mat:string> <out_dvector:string>"
	exit 1;
fi

dvector_ark=$1
lda_mat=$2
out_dvector=$3

ivector-normalize-length ark:${dvector_ark} ark:- | ivector-transform ${lda_mat} ark:- ark:- | ivector-normalize-length ark:- ark:${out_dvector}
