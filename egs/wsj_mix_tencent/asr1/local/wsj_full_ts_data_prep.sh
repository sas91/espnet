#!/bin/bash

data=./data
local=`pwd`/local
rm -r ${data}/{wsj_mix_*} 2>/dev/null
if [ -f path.sh ]; then . ./path.sh; fi

for x in  tr_si284 dev93 eval92_HI eval92_LI eval92_HN eval92_LN; do
  mkdir -p ${data}/wsj_mix_${x}/
  find -L ${data}/wsj_full_ts/${x}/ -iname '*.wav' | sed 's!.*/!!' | sed 's/\.[^.]*$//' | sed 's/\.[^.]*$//' | sort | uniq > ${data}/wsj_mix_${x}/wav.list
  for ch in 1 2 3 4 5 6; do
    mkdir -p ${data}/wsj_mix_${x}/ch${ch}
    awk -v ch=$ch -v dir=$PWD/${data}/wsj_full_ts/${x} '{printf("%s_CH%s %s/%s.CH%s.wav\n", $1, ch, dir, $1, ch)}' $data/wsj_mix_$x/wav.list | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > ${data}/wsj_mix_${x}/ch${ch}/wav.scp
    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/wsj_mix_${x}/ch${ch}/wav.scp | sort > ${data}/wsj_mix_${x}/ch${ch}/utt2spk
    utt2spk_to_spk2utt.pl ${data}/wsj_mix_${x}/ch${ch}/utt2spk > ${data}/wsj_mix_${x}/ch${ch}/spk2utt
  done
done

# transcriptions
rm -r tmp/ 2>/dev/null
mkdir -p tmp
cd tmp
rm -r links/ 2>/dev/null
mkdir links
ln -s $* links

# Do some basic checks that we have what we expected.
if [ ! -d links/11-13.1 -o ! -d links/13-34.1 -o ! -d links/11-2.1 ]; then
  echo "wsj_data_prep.sh: Spot check of command line arguments failed"
  echo "Command line arguments must be absolute pathnames to WSJ directories"
  echo "with names like 11-13.1."
  echo "Note: if you have old-style WSJ distribution,"
  echo "local/cstr_wsj_data_prep.sh may work instead, see run.sh for example."
  exit 1;
fi

# This version for SI-84

cat links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl $* | sort | \
 grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_si84.flist

nl=`cat train_si84.flist | wc -l`
[ "$nl" -eq 7138 ] || echo "Warning: expected 7138 lines in train_si84.flist, got $nl"

# This version for SI-284
cat links/13-34.1/wsj1/doc/indices/si_tr_s.ndx \
 links/11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx | \
 $local/ndx2flist.pl  $* | sort | \
 grep -v -i 11-2.1/wsj0/si_tr_s/401 > tr_si284.flist

nl=`cat tr_si284.flist | wc -l`
[ "$nl" -eq 37416 ] || echo "Warning: expected 37416 lines in train_si284.flist, got $nl"

cat links/11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx | \
  $local/ndx2flist.pl $* |  awk '{printf("%s.wv1\n", $1)}' | \
  sort > eval92.flist

# Dev-set for Nov'93 (503 utts)
cat links/13-34.1/wsj1/doc/indices/h1_p0.ndx | \
  $local/ndx2flist.pl $* | sort > dev93.flist

# Finding the transcript files:
for x in $*; do find -L $x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for f in tr_si284 dev93 eval92; do
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
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/tr_si284.txt $data/wsj_mix_tr_si284/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_tr_si284/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/tr_si284.txt $data/wsj_mix_tr_si284/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_tr_si284/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/dev93.txt $data/wsj_mix_dev93/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_dev93/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/dev93.txt $data/wsj_mix_dev93/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_dev93/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_HI/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_HI/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_HI/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_HI/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_LI/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_LI/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_LI/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_LI/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_HN/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_HN/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_HN/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_HN/ch${i}/text_spk2
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_LN/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_LN/ch${i}/text_spk1
  awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[4]; text=txt[utt2]; print($1, text)}' tmp/eval92.txt $data/wsj_mix_eval92_LN/ch${i}/wav.scp | awk '{$2=""; print $0}' > $data/wsj_mix_eval92_LN/ch${i}/text_spk2
done

for x in tr_si284 dev93 eval92_HI eval92_LI eval92_HN eval92_LN; do
  cat data/wsj_mix_${x}/ch*/wav.scp | sort > data/wsj_mix_${x}/wav.scp
  cat data/wsj_mix_${x}/ch*/utt2spk | sort > data/wsj_mix_${x}/utt2spk
  cat data/wsj_mix_${x}/ch*/text_spk1 | sort > data/wsj_mix_${x}/text_spk1
  cat data/wsj_mix_${x}/ch*/text_spk2 | sort > data/wsj_mix_${x}/text_spk2
  utt2spk_to_spk2utt.pl ${data}/wsj_mix_${x}/utt2spk > ${data}/wsj_mix_${x}/spk2utt
done
