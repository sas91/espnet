#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=5        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# configuration path
preprocess_conf=conf/preprocess.json

# network architecture
num_spkrs=1
# frontend related
use_beamformer=True
ref_channel=-1
use_wpe_test=False
bftype=mask_asr
blayers=2
bunits=771
bprojs=514
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.2

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=8
maxlen_in=600  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=15
patience=3

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
lm_batchsize=600    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_patience=3
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# data
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B
si_tr_s_mix_sim_data=/export/c04/asubraman/simuData_WSJ_Aswin/outWavs
channels="1 2 3 4 5 6"

# cmvn
stats_file=""
apply_uttmvn=true

# exp tag
tag="" # tag for managing experiments.
cmvn_tag=""
fe_model=""
#fe_model_freeze=true
fe_model_freeze=false
#asr_model=""
#fe_model=exp_enhance/enhance_wsj_si_tr_s_multich_pytorch_mse_bs100_ng1_btmask/results/model.loss.best 
asr_model=exp/si_tr_s_wsj0_wsj1_pytorch_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs60_mli520_mlo130_ng1_lsmunigram0.05/results/model.loss.best
pt_loss=mse_psm

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=wsj_si_tr_s
train_dev=wsj_dt05
train_test="wsj_et05_HI wsj_et05_LI wsj_et05_HN wsj_et05_LN"
#recog_set="wsj_et05_LN"
recog_set="wsj_et05_HI wsj_et05_LI wsj_et05_HN wsj_et05_LN"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/sox_extract_wsj0.sh ${si_tr_s_mix_sim_data}/wsj_si_tr_s/ wsj_si_tr_s
    local/sox_extract_wsj0.sh ${si_tr_s_mix_sim_data}/wsj_dt05/ wsj_dt05
    local/sox_extract_wsj0.sh ${si_tr_s_mix_sim_data}/wsj_et05_HI/ wsj_et05_HI
    local/sox_extract_wsj0.sh ${si_tr_s_mix_sim_data}/wsj_et05_LI/ wsj_et05_LI
    local/sox_extract_wsj0.sh ${si_tr_s_mix_sim_data}/wsj_et05_HN/ wsj_et05_HN
    local/sox_extract_wsj0.sh ${si_tr_s_mix_sim_data}/wsj_et05_LN/ wsj_et05_LN
    local/wsj0_data_prep_new.sh ${wsj0}
    local/wsj0_data_prep.sh ${wsj0} ${wsj1}
fi

feat_tr_dir=data/${train_set}; #${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=data/${train_dev}; #${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    #echo "stage 1: Feature Generation"
    #fbankdir=fbank
    ## Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    ## <data/tr/utt2spk utils/utt2spk_to_spk2utt.pl >data/tr/spk2utt
    #for x in tr; do
    #    steps/make_fbank.sh --cmd "$train_cmd" --nj 64 --write_utt2num_frames true \
    #        data/${x} exp/make_fbank/${x} ${fbankdir}
    #done
    #
    #compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    echo "stage 1: Dump wav files into a HDF5 file"
    
    for setname in ${train_set} ${train_dev} ${train_test}; do
        mkdir -p data/${setname}_multich
        <data/${setname}/utt2spk sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/utt2spk
        <data/${setname}/text_spk1 sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/text_spk1
        <data/${setname}/text_spk2 sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/text_spk2
        <data/${setname}/text_spk1 sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/text
        <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

        for ch in 1 2 3 4 5 6; do
            <data/${setname}/wav.scp grep "CH${ch}" | sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
        done
        mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp >data/${setname}_multich/wav.scp
        rm -f data/${setname}_multich/wav_ch*.scp
    done

    for setname in ${train_set} ${train_dev} ${recog_set}; do
        dump_pcm.sh --cmd "$train_cmd" --nj 64 --filetype "sound.hdf5" data/${setname}_multich
    	cat data/${setname}/ch1/wav.scp | sed 's/_CH1//g' | sed 's/CH1/CLEAN/' > data/${setname}_multich/clean.scp
    	cat data/${setname}/ch1/wav.scp | sed 's/_CH1//g' | sed 's/CH1/S1/' > data/${setname}_multich/s1.scp
    	cat data/${setname}/ch1/wav.scp | sed 's/_CH1//g' > data/${setname}_multich/noisy_ch1.scp
    done
    local/get_spk_angle.sh
    #for setname in train_si284; do
    #    <data/${setname}/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}/spk2utt
    #done
    #dump_pcm.sh --cmd "$train_cmd" --nj 64 --filetype "sound.hdf5" --format flac data/tr_single
    #dump_pcm.sh --cmd "$train_cmd" --nj 64 --filetype "sound.hdf5" --format flac data/cv_single
    #dump_pcm.sh --cmd "$train_cmd" --nj 64 --filetype "sound.hdf5" data/train_si284
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: X-Vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    nj=8
    #for name in si_tr_s si_dt_05 si_et_05; do
    #    utils/copy_data_dir.sh data/${name} data/${name}_mfcc
    #    steps/make_mfcc.sh \
    #        --write-utt2num-frames true \
    #        --mfcc-config conf/mfcc.conf \
    #        --nj ${nj} --cmd "$train_cmd" \
    #        data/${name}_mfcc exp/make_mfcc ${mfccdir}
    #    utils/fix_data_dir.sh data/${name}_mfcc
    #    sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
    #        data/${name}_mfcc exp/make_vad ${vaddir}
    #    utils/fix_data_dir.sh data/${name}_mfcc
    #done
    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ];then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    #for name in si_tr_s si_dt_05 si_et_05; do
    #    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
    #        ${nnet_dir} data/${name}_mfcc \
    #        ${nnet_dir}/xvectors_${name}
    #done
    local/add_spk_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_tr_s/spk_xvector.scp --utt2spk data/wsj_si_tr_s_multich/utt2spk --out data/wsj_si_tr_s_multich/xvec_spk.scp
    local/add_spk_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_dt_05/spk_xvector.scp --utt2spk data/wsj_dt05_multich/utt2spk --out data/wsj_dt05_multich/xvec_spk.scp
    local/add_spk_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/spk_xvector.scp --utt2spk data/wsj_et05_HI_multich/utt2spk --out data/wsj_et05_HI_multich/xvec_spk.scp
    local/add_spk_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/spk_xvector.scp --utt2spk data/wsj_et05_LI_multich/utt2spk --out data/wsj_et05_LI_multich/xvec_spk.scp
    local/add_spk_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/spk_xvector.scp --utt2spk data/wsj_et05_HN_multich/utt2spk --out data/wsj_et05_HN_multich/xvec_spk.scp
    local/add_spk_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/spk_xvector.scp --utt2spk data/wsj_et05_LN_multich/utt2spk --out data/wsj_et05_LN_multich/xvec_spk.scp
    local/add_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_tr_s/xvector.scp --utt2spk data/wsj_si_tr_s_multich/utt2spk --out data/wsj_si_tr_s_multich/xvec.scp
    local/add_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_dt_05/xvector.scp --utt2spk data/wsj_dt05_multich/utt2spk --out data/wsj_dt05_multich/xvec.scp
    local/add_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/xvector.scp --utt2spk data/wsj_et05_HI_multich/utt2spk --out data/wsj_et05_HI_multich/xvec.scp
    local/add_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/xvector.scp --utt2spk data/wsj_et05_LI_multich/utt2spk --out data/wsj_et05_LI_multich/xvec.scp
    local/add_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/xvector.scp --utt2spk data/wsj_et05_HN_multich/utt2spk --out data/wsj_et05_HN_multich/xvec.scp
    local/add_xvec.py --xvector-scp exp/xvector_nnet_1a/xvectors_si_et_05/xvector.scp --utt2spk data/wsj_et05_LN_multich/utt2spk --out data/wsj_et05_LN_multich/xvec.scp
    local/add_ss.py --feats-scp data/si_tr_s_wsj0_wsj1/feats.scp --utt2spk data/wsj_si_tr_s_multich/utt2spk --out data/wsj_si_tr_s_multich/ss_wav.scp --out2 data/wsj_si_tr_s_multich/ss_wav_2.scp --out3 data/wsj_si_tr_s_multich/ss_wav_3.scp 
    local/add_ss.py --feats-scp data/si_dt_05/feats.scp --utt2spk data/wsj_dt05_multich/utt2spk --out data/wsj_dt05_multich/ss_wav.scp --out2 data/wsj_dt05_multich/ss_wav_2.scp --out3 data/wsj_dt05_multich/ss_wav_3.scp
    local/add_ss.py --feats-scp data/si_et_05/feats.scp --utt2spk data/wsj_et05_HI_multich/utt2spk --out data/wsj_et05_HI_multich/ss_wav.scp --out2 data/wsj_et05_HI_multich/ss_wav_2.scp --out3 data/wsj_et05_HI_multich/ss_wav_3.scp
    local/add_ss.py --feats-scp data/si_et_05/feats.scp --utt2spk data/wsj_et05_LI_multich/utt2spk --out data/wsj_et05_LI_multich/ss_wav.scp --out2 data/wsj_et05_LI_multich/ss_wav_2.scp --out3 data/wsj_et05_LI_multich/ss_wav_3.scp
    local/add_ss.py --feats-scp data/si_et_05/feats.scp --utt2spk data/wsj_et05_HN_multich/utt2spk --out data/wsj_et05_HN_multich/ss_wav.scp --out2 data/wsj_et05_HN_multich/ss_wav_2.scp --out3 data/wsj_et05_HN_multich/ss_wav_3.scp
    local/add_ss.py --feats-scp data/si_et_05/feats.scp --utt2spk data/wsj_et05_LN_multich/utt2spk --out data/wsj_et05_LN_multich/ss_wav.scp --out2 data/wsj_et05_LN_multich/ss_wav_2.scp --out3 data/wsj_et05_LN_multich/ss_wav_3.scp
fi


train_set=${train_set}_multich
train_dev=${train_dev}_multich
recog_set="$(for setname in ${recog_set}; do echo -n "${setname}_multich "; done)"
dict=data/lang_1char/si_tr_s_wsj0_wsj1_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 3: Dictionary and Json Data Preparation"
    #mkdir -p data/lang_1char/

    #echo "make a non-linguistic symbol list"
    #cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    #cat ${nlsyms}

    #echo "make a dictionary"
    #echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    #text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    #| sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    for setname in ${train_set} ${train_dev} ${recog_set}; do
    	dump_pcm.sh --nj 64 --filetype "sound.hdf5" --f_in "clean.scp" --f_out "feats_clean.scp" data/${setname}/ data/${setname}/log_clean data/${setname}/data_clean
    done

    echo "make json files"

    for setname in ${train_set} ${train_dev} ${recog_set}; do
        local/data2json.sh --cmd "${train_cmd}" --nj 64 --angle1 1 --angle2 1 --xvector 1 --xvector_spk 1 --clean 1 --ss_wav 1 --ss_wav_2 1 --ss_wav_3 1\
            --category "multichannel" \
            --preprocess-conf ${preprocess_conf} --filetype sound.hdf5 \
            --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
            --out data/${setname}/data.json data/${setname} ${dict}
    done
    #setname=train_si284
    #local/data2json.sh --cmd "${train_cmd}" --nj 64 \
    #    --category "singlechannel" \
    #    --preprocess-conf ${preprocess_conf} --filetype sound.hdf5 \
    #    --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
    #    --out data/${setname}/data.json data/${setname} ${dict}

    #train_set=tr_spatialized_anechoic_multich_si284
    #mkdir -p data/${train_set}
    #concatjson.py data/tr_multich/data.json data/train_si284/data.json > data/${train_set}/data.json
fi
    
# It takes about one day. If you just want to do end-to-end ASR without LM,
# YOU CAN SKIP this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: LM Preparation"
    
    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        #mkdir -p ${lmdatadir}
        #cut -f 2- -d" " data/${wsj_train_set}/text > ${lmdatadir}/train_trans.txt
        #zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
        #        | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        #cut -f 2- -d" " data/${wsj_train_dev}/text > ${lmdatadir}/valid.txt
        #cut -f 2- -d" " data/${wsj_train_test}/text > ${lmdatadir}/test.txt
        #cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        #text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${wsj_train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --patience ${lm_patience} \
        --maxlen ${lm_maxlen} \
        --dict ${lmdict}
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_ng${ngpu}_bt${bftype}_rc${ref_channel}
    if [ "${lsm_type}" != "" ]; then
        expname=${expname}_lsm${lsm_type}${lsm_weight}
    fi
    if [ "${asr_model}" != "" ]; then
        expname=${expname}_apt
    fi
    if [ "${fe_model}" != "" ]; then
        expname=${expname}_fpt${pt_loss}_ffe${fe_model_freeze}
    fi
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
if [ -n "${cmvn_tag}" ]; then
    expname=${expname}_${cmvn_tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Network Training"
    train_opts=""
    if [ -n "${stats_file}" ]; then
        train_opts=${train_opts}" --stats-file ${stats_file}"
    fi
    if ${apply_uttmvn}; then
        train_opts=${train_opts}" --apply-uttmvn true"
    fi

	${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --num-spkrs ${num_spkrs} \
        --use-frontend True \
        --use-beamformer ${use_beamformer} \
        --ref-channel ${ref_channel} \
        --blayers ${blayers} \
        --bunits ${bunits} \
        --bftype ${bftype} \
        --bprojs ${bprojs} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json data/${train_set}/data.json \
        --valid-json data/${train_dev}/data.json \
        --preprocess-conf ${preprocess_conf} \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --sampling-probability ${samp_prob} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --sortagrad ${sortagrad} \
        --epochs ${epochs} \
        --patience ${patience} \
        --fe-model ${fe_model} \
        --fe-model-freeze ${fe_model_freeze} \
        --asr-model ${asr_model} \
        ${train_opts}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding"
    nj=20

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}_twpe${use_wpe_test}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        if [ ${lm_weight} == 0 ]; then
            recog_opts=""
        fi
        feat_recog_dir=data/${rtask}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --backend ${backend} \
            --use-wpe-test ${use_wpe_test} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            ${recog_opts} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} --num_spkrs 1 ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Enhance speech"
    nj=10
    ngpu=0

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
        mkdir -p ${enhdir}/outdir
        splitjson.py --parts ${nj} data/${rtask}/data.json

        ${decode_cmd} JOB=1:${nj} ${enhdir}/log/enhance.JOB.log \
            asr_enhance.py \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --model ${expdir}/results/${recog_model}  \
            --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
            --enh-wspecifier ark,scp:${enhdir}/outdir/enhance.JOB,${enhdir}/outdir/enhance.JOB.scp \
            --enh-filetype "sound" \
            --image-dir ${enhdir}/images \
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
        for i in $(seq 1 ${nj}); do
            cat ${enhdir}/outdir/enhance.${i}.scp
        done > ${enhdir}/enhance.scp
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Evaluate enhanced speech"
    nj=10

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
	sort ${enhdir}/enhance.scp > data/${rtask}/estimated.scp
	#basedir=${enhdir}/se_scores/enhan_SSDR_v4/
        #mkdir -p ${basedir}
        #eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/clean.scp data/${rtask}/estimated.scp ${basedir}
	#basedir=${enhdir}/se_scores/enhan_ds_SSDR_v4/
        #mkdir -p ${basedir}
        #eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/s1.scp data/${rtask}/estimated.scp ${basedir}
	#basedir=${enhdir}/se_scores/enhan/
        #mkdir -p ${basedir}
        #eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR STOI ESTOI" --nj ${nj} --bss-eval-images true data/${rtask}/clean.scp data/${rtask}/estimated.scp ${basedir}
	#basedir=${enhdir}/se_scores/enhan_ds/
        #mkdir -p ${basedir}
        #eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR STOI ESTOI" --nj ${nj} --bss-eval-images true data/${rtask}/s1.scp data/${rtask}/estimated.scp ${basedir}
	basedir=${enhdir}/se_scores/enhan/
        mkdir -p ${basedir}
        eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR PESQ" --nj ${nj} data/${rtask}/clean.scp data/${rtask}/estimated.scp ${basedir}
	basedir=${enhdir}/se_scores/enhan_ds/
        mkdir -p ${basedir}
        eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR PESQ" --nj ${nj} data/${rtask}/s1.scp data/${rtask}/estimated.scp ${basedir}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    for rtask in ${recog_set}; do
    (
        enhdir=${expdir}/enhance_${rtask}
        #basedir=${enhdir}/se_scores/noisy_ch1_SSDR_v4/
        #mkdir -p ${basedir}
	#eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/clean.scp data/${rtask}/noisy_ch1.scp ${basedir}
        #basedir=${enhdir}/se_scores/noisy_ch1_ds_SSDR_v4/
        #mkdir -p ${basedir}
	#eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR" --nj ${nj} --bss-eval-images false --bss_eval_version "v4" data/${rtask}/s1.scp data/${rtask}/noisy_ch1.scp ${basedir}
        #basedir=${enhdir}/se_scores/noisy_ch1/
        #mkdir -p ${basedir}
	#eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR STOI ESTOI" --nj ${nj} --bss-eval-images true data/${rtask}/clean.scp data/${rtask}/noisy_ch1.scp ${basedir}
        #basedir=${enhdir}/se_scores/noisy_ch1_ds/
        #mkdir -p ${basedir}
	#eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR STOI ESTOI" --nj ${nj} --bss-eval-images true data/${rtask}/s1.scp data/${rtask}/noisy_ch1.scp ${basedir}
        basedir=${enhdir}/se_scores/noisy_ch1/
        mkdir -p ${basedir}
	eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR PESQ" --nj ${nj} data/${rtask}/clean.scp data/${rtask}/noisy_ch1.scp ${basedir}
        basedir=${enhdir}/se_scores/noisy_ch1_ds/
        mkdir -p ${basedir}
	eval_source_separation.sh --cmd "${decode_cmd}" --evaltypes "SDR PESQ" --nj ${nj} data/${rtask}/s1.scp data/${rtask}/noisy_ch1.scp ${basedir}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        decode_dir=decode_wsj_et05_LN_multich_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}_twpe${use_wpe_test}
	local/score_for_angle.sh --wer true --nlsyms ${nlsyms} \
		"${expdir}/${decode_dir}/data.json" \
		${dict} ${expdir}/${decode_dir}/decode_summary_LI
    	for rtask in ${recog_set}; do
        	decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}_twpe${use_wpe_test}
		score_sclite.sh --wer true --nlsyms ${nlsyms} --num_spkrs 1 ${expdir}/${decode_dir} ${dict}
		grep -e Avg -e SPKR -m 2 ${expdir}/${decode_dir}/result.wrd.txt
	done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    	for rtask in ${recog_set}; do
        	decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}_twpe${use_wpe_test}
        	enhdir=${expdir}/enhance_${rtask}
		echo ${rtask}
		cat ${enhdir}/se_scores/enhan_ds_SSDR_v4/mean_SDR
		cat ${enhdir}/se_scores/noisy_ch1_SSDR_v4/mean_SDR
		cat ${enhdir}/se_scores/enhan_ds/mean_STOI
		cat ${enhdir}/se_scores/noisy_ch1_ds/mean_STOI
	done
fi
