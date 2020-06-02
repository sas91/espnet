#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false

. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <jsons> <dict> <out-dir>";
    exit 1;
fi

jsons=$1
dic=$2
dir=$3

tasks="\
air-1 air-2 \
air-3 air-4"

for task in ${tasks}; do
    mkdir -p ${dir}/${task}
    python local/filterjson.py -f ${task} ${jsons} > ${dir}/${task}/data.1.json
    score_sclite.sh --wer ${wer} --nlsyms ${nlsyms} ${dir}/${task} ${dic} 1> /dev/null 2> /dev/null
done

echo "Scoring for different angles"
for task in ${tasks}; do
    echo "${task}:"
    grep -e Avg -e SPKR -m 2 ${dir}/${task}/result.wrd.txt
done | sed -e 's/\s\s\+/\t/g' | tee ${dir}/result.txt
