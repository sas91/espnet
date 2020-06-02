#!/bin/bash

# Copyright 2018 Xuankai Chang (Johns Hopkins University & Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section
cmd=run.pl
nj=4
data=data
local=`pwd`/local

# End configuration section

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -le 3 ]; then
  echo "Usage: $0 [options] <wav-dir> <src-dir> <wsj0-dir> <channels> ";
  echo "e.g.: $0 spatialize_wsj0_wav/ spatialize_wsj0_wav/script wsj0_dir \"1 2\" ";
  echo "Options: "
  echo "--cmd (run.pl | queue.pl | slurm.pl)"
  echo "--nj num of jobs"
  echo "--data output dir, default is ./data/"
  exit 1;
fi

wavdir=$1
srcdir=$2
wsj0dir=$3
channels=$4
wavdir_isolated=$5


# check if the wav dir exists.
for f in $wavdir/tr $wavdir/cv $wavdir/tt; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

# check if the script file exists.
for f in $srcdir/wsj0-2mix_tr.flist $srcdir/wsj0-2mix_cv.flist $srcdir/wsj0-2mix_tt.flist; do
  if [ ! -f $f ]; then
    echo "Could not find $f.";
    exit 1;
  fi
done

for x in tr cv tt; do
  mkdir -p $data/$x
  cat $srcdir/wsj0-2mix_${x}.flist | sed 's/\.wav//' | \
    awk -v dir=$wavdir/$x '{printf("%s %s/mix/%s.wav\n", $1, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > $data/$x/wav.list

  for i in $channels; do
    tgtdir=$data/$x/ch$i
    mkdir -p $tgtdir
    awk -v ch=$i -v tgtwavdir=$wavdir_isolated/$x/mix '{printf("%s_CH%s %s/%s_CH%s.wav\n", $1, ch, tgtwavdir, $1, ch)}' $data/$x/wav.list | sort > $tgtdir/wav.scp
    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' $tgtdir/wav.scp | sort > $tgtdir/utt2spk
    utt2spk_to_spk2utt.pl $tgtdir/utt2spk > $tgtdir/spk2utt
  done
done


# transcriptions
rm -r tmp/ 2>/dev/null
mkdir -p tmp
cd tmp
rm -r links/ 2>/dev/null
mkdir links
ln -s ${wsj0dir}/??-{?,??}.? links

# Do some basic checks that we have what we expected.
if [ ! -d links/11-13.1 ]; then
  echo "WSJ0 directory may be in a noncompatible form."
  exit 1;
fi

for disk in 11-1.1 11-2.1 11-3.1; do
  for spk in `ls links/${disk}/wsj0/si_tr_s`; do
    ls links/${disk}/wsj0/si_tr_s/$spk | grep wv1 | \
      awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_tr_s/%s/%s\n", pwd, disk, spk, $1)}'
  done
done | sort > si_tr_s.flist

disk=11-14.1;
for spk in `ls links/${disk}/wsj0/si_et_05`; do
  ls links/${disk}/wsj0/si_et_05/$spk | grep wv1 | \
    awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_et_05/%s/%s\n", pwd, disk, spk, $1)}'
done | sort > si_et_05.flist

disk=11-6.1;
for spk in `ls links/${disk}/wsj0/si_dt_05`; do
  ls links/${disk}/wsj0/si_dt_05/$spk | grep wv1 | \
    awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_dt_05/%s/%s\n", pwd, disk, spk, $1)}'
done | sort > si_dt_05.flist

# Finding the transcript files:
for x in `ls links/`; do find -L links/$x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for f in si_tr_s si_et_05 si_dt_05; do
  $local/flist2scp.pl ${f}.flist | sort > ${f}.scp
  cat ${f}.scp | awk '{print $1}' | $local/find_transcripts.pl dot_files.flist > ${f}.trans1

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  noiseword="<NOISE>"
  cat ${f}.trans1 | $local/normalize_transcript.pl $noiseword | sort > ${f}.txt || exit 1;
done

# change to the original path
cd ..

for i in $channels; do
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt $data/tr/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/tr/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt $data/tr/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/tr/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt $data/cv/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/cv/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt $data/cv/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/cv/ch${i}/text_spk2
  awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt $data/tt/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/tt/ch${i}/text_spk1
  awk '(ARGIND<=2) {txt[$1]=$0} (ARGIND==3) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_dt_05.txt tmp/si_et_05.txt $data/tt/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/tt/ch${i}/text_spk2
done

for x in tr cv tt; do
  cat data/$x/ch*/wav.scp | sort > data/$x/wav.scp
  cat data/$x/ch*/utt2spk | sort > data/$x/utt2spk
  cat data/$x/ch*/text_spk1 | sort > data/$x/text_spk1
  cat data/$x/ch*/text_spk2 | sort > data/$x/text_spk2
done

rm -r tmp/
