#!/bin/bash

src_full=$1
tgt_dir=$2

mkdir -p wav/${tgt_dir}
mkdir -p log
rm -rf log/${tgt_dir}.sh
rm -rf log/${tgt_dir}.scp

cat ${src_full} | while read LINE; do
    wav_cmd=`echo $LINE | cut -d " " -f2-5`
    src_file=`echo $LINE | cut -d " " -f5`
    echo $src_file
    filename=$(basename ${src_file} .wv1)
    echo "$wav_cmd wav/$tgt_dir/$filename.wav" >> log/${tgt_dir}.sh
    echo "$PWD/wav/$tgt_dir/$filename.wav" >> log/${tgt_dir}.scp
done

chmod 755 log/${tgt_dir}.sh
log/${tgt_dir}.sh
