#!/bin/bash

src_dir=$1
tgt_dir=$2

tgt_dir_full=data/wsj_full_ts/$tgt_dir
mkdir -p data/wsj_full_ts/$tgt_dir
mkdir -p data/wsj_mix_$tgt_dir

for f in `ls ${src_dir}`; do
    filename=$(basename ${f} .wav)
    srcwav=${src_dir}/${filename}.wav
    for ch in $(seq 1 6); do
	tgtname=`echo $filename | cut -d "_" -f1-3 | sed 's/SPH_//g' | sed 's/\./_/g'`
        tgtwav=${tgt_dir_full}/${tgtname}.CH${ch}.wav
        sox ${srcwav} ${tgtwav} remix ${ch}
    done
    tgtname=`echo $filename | cut -d "_" -f1-3 | sed 's/SPH_//g' | sed 's/\./_/g'`
    tgtwav=${tgt_dir_full}/${tgtname}.CLEAN.wav
    sox ${srcwav} ${tgtwav} remix 9
    tgtwav=${tgt_dir_full}/${tgtname}.INT.wav
    sox ${srcwav} ${tgtwav} remix 10
    tgtwav=${tgt_dir_full}/${tgtname}.NOISE.wav
    sox ${srcwav} ${tgtwav} remix 8
    tgtwav=${tgt_dir_full}/${tgtname}.S1.wav
    sox ${srcwav} ${tgtwav} remix 7
done
