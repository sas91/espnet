#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=1
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false
# configuration path
preprocess_conf=conf/preprocess_doa.json

# network architecture
num_spkrs=1
# frontend related
llayers=2
lunits=771
lprojs=514
loss_type=ce
resolution=10
arch_type=2

# minibatch related
batchsize=30
maxlen_in=600  # if input length  > maxlen_in, batchsize is automatically reduced

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

expname=doa_${train_set}_${backend}_${loss_type}_arch${arch_type}_bs${batchsize}_ng${ngpu}_res${resolution}
expdir=exp_doa/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Training"

	${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
	doa_train.py \
        --num-spkrs ${num_spkrs} \
	--arch-type ${arch_type} \
        --resolution ${resolution} \
        --llayers ${llayers} \
        --lunits ${lunits} \
        --lprojs ${lprojs} \
        --ngpu ${ngpu} \
        --loss-type ${loss_type} \
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
    echo "stage 2: Classify"
    nj=10
    ngpu=0

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/doa_${rtask}
        mkdir -p ${enhdir}/outdir
        splitjson.py --parts ${nj} data/${rtask}/data.json

        ${decode_cmd} JOB=1:${nj} ${enhdir}/log/doa.JOB.log \
		doa_test.py \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --model ${expdir}/results/${recog_model}  \
                --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
