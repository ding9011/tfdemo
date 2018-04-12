#!/bin/bash

set -e
set -x

if [ $# -ne 5 ]; then
	echo "Usage: $0 <string:indir> <string:outdir> <string:filetype> <int:number of utt> <move_symbol:string(yes|no)>"
	exit 1;
fi

indir=$1
outdir=$2
filetype=$3
utt=$4
mv_symbol=$5

mkdir -p ${outdir}

for x in `find ${indir} -type f | grep -E "*.${filetype}$" | sort -u -k1`;
do
	c_spkid=${x#*/}
	c_spkid=${c_spkid%%/*}
	num_utt=`ls ${outdir}/${c_spkid} | wc -l`
	if [ -d ${outdir}/${c_spkid} ] && [ ${num_utt} -ge ${utt} ];then
		continue
	else
		mkdir -p ${outdir}/${c_spkid} || exit 1;
		if [[ 'x'${mv_symbol} == 'xyes' ]]; then
			mv ${x} ${outdir}/${c_spkid} || exit 1;
		else
			cp ${x} ${outdir}/${c_spkid} || exit 1;
		fi
	fi
done
