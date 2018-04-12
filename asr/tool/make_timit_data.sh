#!/bin/bash

set -x
set -e

datadir=$1
outdir=$2

if [ $# -ne 2 ]; then
	echo "Usage: $0 <datadir:stirng> <outdir:string>"
	exit 1;
fi

[ -d ${outdir} ] && rm -rf ${outdir} > /dev/null
mkdir -p ${outdir}


for x in `find ${datadir} -type f | grep -E "*.wav$" | sort -u -k1`;
do
	filename=`basename ${x}`
	filepath=`dirname ${x}`
	speaker=`echo ${filepath##*/}`
	[ -d ${outdir}/${speaker} ] || mkdir -p ${outdir}/${speaker} || exit 1;
	$(sox ${x} -t wav ${outdir}/${speaker}/${filename})
done
