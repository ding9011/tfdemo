#!/bin/bash

set -x
set -e

datadir=$1
outdir=$2
filetype=$3

if [ $# -ne 3 ]; then
	echo "Usage: $0 <datadir:stirng> <outdir:string> <filetype:string>"
	exit 1;
fi

[ -d ${outdir} ] && rm -rf ${outdir} > /dev/null
mkdir -p ${outdir}


for x in `find ${datadir} -type f | grep -E "*.${filetype}$" | sort -u -k1`;
do
	filename=`basename ${x}`
	wavfilename=${filename%%.*}.wav
	speaker=${filename%%-*}
	[ -d ${outdir}/${speaker} ] || mkdir -p ${outdir}/${speaker} || exit 1;
	$(sox ${x} -t wav ${outdir}/${speaker}/${wavfilename})
done
