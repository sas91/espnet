#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=3
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false
use_bf_test=true
# configuration path
preprocess_conf=conf/preprocess_enh.json

# network architecture
num_spkrs=1
# frontend related
use_beamformer=True
bftype=mask
blayers=2
bunits=771
bprojs=514
loss=mse

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adam
epochs=30
patience=3
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_set=tr
#train_dev=dt
#recog_set="et_multich"
train_set=wsj_si_tr_s
train_dev=wsj_dt05
recog_set="wsj_et05_HI wsj_et05_LI wsj_et05_HN wsj_et05_LN"
recog_set="$(for setname in ${recog_set}; do echo -n "${setname}_multich "; done)"

train_set=${train_set}_multich
train_dev=${train_dev}_multich

expname=enhance_${train_set}_${backend}_${loss}_bs${batchsize}_ng${ngpu}_bt${bftype}
expdir=exp_enhance/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Training"

	${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        enhance_train.py \
        --num-spkrs ${num_spkrs} \
        --use-beamformer ${use_beamformer} \
        --blayers ${blayers} \
        --bunits ${bunits} \
        --bftype ${bftype} \
        --bprojs ${bprojs} \
        --ngpu ${ngpu} \
        --loss ${loss} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --maxlen-in ${maxlen_in} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json data/${train_set}/data.json \
        --valid-json data/${train_dev}/data.json \
        --preprocess-conf ${preprocess_conf} \
        --batch-size ${batchsize} \
        --sortagrad ${sortagrad} \
        --epochs ${epochs} \
        --patience ${patience}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Enhance speech"
    nj=10
    ngpu=0

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
    	if [ ${use_bf_test} == true ]; then
    	    enhdir=${enhdir}_bf
    	fi
        mkdir -p ${enhdir}/outdir
        splitjson.py --parts ${nj} data/${rtask}/data.json

        ${decode_cmd} JOB=1:${nj} ${enhdir}/log/enhance.JOB.log \
            enhance_test.py \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --model ${expdir}/results/${recog_model}  \
                --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
                --enh-wspecifier ark,scp:${enhdir}/outdir/enhance.JOB,${enhdir}/outdir/enhance.JOB.scp \
                --enh-filetype "sound" \
                --image-dir ${enhdir}/images \
                --use-bf-test ${use_bf_test} \
                --num-images 20 \
                --fs 16000
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    # Reduce all scp files from each jobs to one
    for rtask in ${recog_set}; do
        enhdir=${expdir}/enhance_${rtask}
    	if [ ${use_bf_test} == true ]; then
    	    enhdir=${enhdir}_bf
    	fi
        for i in $(seq 1 ${nj}); do
            cat ${enhdir}/outdir/enhance.${i}.scp
        done > ${enhdir}/enhance.scp
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Evaluate enhanced speech"
    nj=10

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
    	if [ ${use_bf_test} == true ]; then
    	    enhdir=${enhdir}_bf
    	fi
	sort ${enhdir}/enhance.scp > data/${rtask}/estimated.scp
	basedir=${enhdir}/se_scores/enhan_SSDR_v4/
        mkdir -p ${basedir}
        eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/clean.scp data/${rtask}/estimated.scp ${basedir}
	basedir=${enhdir}/se_scores/enhan_ds_SSDR_v4/
        mkdir -p ${basedir}
        eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/s1.scp data/${rtask}/estimated.scp ${basedir}
	basedir=${enhdir}/se_scores/enhan/
        mkdir -p ${basedir}
        eval_source_separation.sh --cmd "${decode_cmd}" --nj ${nj} --bss-eval-images true data/${rtask}/clean.scp data/${rtask}/estimated.scp ${basedir}
	basedir=${enhdir}/se_scores/enhan_ds/
        mkdir -p ${basedir}
        eval_source_separation.sh --cmd "${decode_cmd}" --nj ${nj} --bss-eval-images true data/${rtask}/s1.scp data/${rtask}/estimated.scp ${basedir}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
    	if [ ${use_bf_test} == true ]; then
    	    enhdir=${enhdir}_bf
    	fi
        basedir=${enhdir}/se_scores/noisy_ch1_SSDR_v4/
        mkdir -p ${basedir}
	eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/clean.scp data/${rtask}/noisy_ch1.scp ${basedir}
        basedir=${enhdir}/se_scores/noisy_ch1_ds_SSDR_v4/
        mkdir -p ${basedir}
	eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/s1.scp data/${rtask}/noisy_ch1.scp ${basedir}
        basedir=${enhdir}/se_scores/noisy_ch1/
        mkdir -p ${basedir}
	eval_source_separation.sh --cmd "${decode_cmd}" --nj ${nj} --bss-eval-images true data/${rtask}/clean.scp data/${rtask}/noisy_ch1.scp ${basedir}
        basedir=${enhdir}/se_scores/noisy_ch1_ds/
        mkdir -p ${basedir}
	eval_source_separation.sh --cmd "${decode_cmd}" --nj ${nj} --bss-eval-images true data/${rtask}/s1.scp data/${rtask}/noisy_ch1.scp ${basedir}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
