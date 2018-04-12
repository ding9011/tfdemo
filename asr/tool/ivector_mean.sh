#!/bin/bash

set -e
set -x

if [ $# -ne 2 ]; then
	echo "Notice: need source kaldi path.sh first and abspath.\n"
	echo "Usage: $0 <dvector_ark:string> <out_mean:string>"
	exit 1;
fi

dvector_ark=$1
out_mean=$2

ivector-normalize-length ark:${dvector_ark} ark:- |ivector-mean ark:- ${out_mean}
