#!/bin/bash

set -e
set -x

if [ $# -ne 5 ]; then
	echo "Notice: need source kaldi path.sh first and abspath.\n"
	echo "Usage: $0 <test_dvector_ark:string> <transform_mat:string> <out_ark:string>"
	exit 1;
fi

test_dvector_ark=$1
transform_mat=$2
out_ark=$3


ivector-normalize-length ark:${test_dvector_ark} ark:- | ivector-transform ${transform_mat} ark:- ark:- | ivector-normalize-length ark:- ark:${out_ark}
