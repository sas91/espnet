#!/bin/bash

data=./data
local=`pwd`/local
wsj0dir=/data2/v_aswin/corpora/WSJ0
rm -r ${data}/{tr,dt,et} 2>/dev/null
if [ -f path.sh ]; then . ./path.sh; fi

for x in tr dt et; do
  mkdir -p ${data}/$x/
  find -L ${data}/wsj_ts/wav/${x}/ -iname '*.wav' | sed 's!.*/!!' | sed 's/\.[^.]*$//' | sed 's/\.[^.]*$//' | sort | uniq > ${data}/$x/wav.list
  for ch in 1 2 3 4 5 6; do
    mkdir -p ${data}/$x/ch${ch}
    awk -v ch=$ch -v dir=$PWD/${data}/wsj_ts/wav/${x} '{printf("%s_CH%s %s/%s.CH%s.wav\n", $1, ch, dir, $1, ch)}' $data/$x/wav.list | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > ${data}/$x/ch${ch}/wav.scp
    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/$x/ch${ch}/wav.scp | sort > ${data}/$x/ch${ch}/utt2spk
    utt2spk_to_spk2utt.pl ${data}/$x/ch${ch}/utt2spk > ${data}/$x/ch${ch}/spk2utt
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

for i in 1 2 3 4 5 6; do
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt $data/tr/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/tr/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt $data/tr/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/tr/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_dt_05.txt $data/dt/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/dt/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/si_dt_05.txt $data/dt/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/dt/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_et_05.txt $data/et/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/et/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/si_et_05.txt $data/et/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/et/ch${i}/text_spk2
done

for x in tr dt et; do
  cat data/$x/ch*/wav.scp | sort > data/$x/wav.scp
  cat data/$x/ch*/utt2spk | sort > data/$x/utt2spk
  cat data/$x/ch*/text_spk1 | sort > data/$x/text_spk1
  cat data/$x/ch*/text_spk2 | sort > data/$x/text_spk2
  utt2spk_to_spk2utt.pl ${data}/$x/utt2spk > ${data}/$x/spk2utt
done
