#!/bin/bash

src_dir=$1
tgt_dir=$2

for f in `ls ${src_dir}`; do
    filename=$(basename ${f} .wav)
    srcwav=${src_dir}/${filename}.wav
    #for ch in $(seq 1 6); do
    #    tgtname=`echo $filename | cut -d "_" -f1-3 | sed 's/SPH_//g' | sed 's/\./_/g'`
    #    tgtwav=${tgt_dir}/${tgtname}.CH${ch}.wav
    #    sox ${srcwav} ${tgtwav} remix ${ch}
    #done
    tgtname=`echo $filename | cut -d "_" -f1-3 | sed 's/SPH_//g' | sed 's/\./_/g'`
    tgtwav=${tgt_dir}/${tgtname}.CLEAN.wav
    sox ${srcwav} ${tgtwav} remix 9
done
