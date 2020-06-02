#!/bin/bash

data=./data
local=`pwd`/local
wsj0dir=$1
wsj1dir=$2
rm -r ${data}/{si_tr_s,si_tr_s_wsj0_wsj1,si_dt_05,si_et_05} 2>/dev/null
if [ -f path.sh ]; then . ./path.sh; fi

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

mkdir -p data/si_tr_s
mkdir -p data/si_tr_s_wsj0_wsj1
mkdir -p data/si_dt_05
mkdir -p data/si_et_05
# transcriptions
rm -r tmp/ 2>/dev/null
mkdir -p tmp
cd tmp
rm -r links/ 2>/dev/null
mkdir links
ln -s ${wsj0dir}/??-{?,??}.? ${wsj1dir}/??-{?,??}.? links

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

cat links/13-34.1/wsj1/doc/indices/si_tr_s.ndx | \
 $local/ndx2flist.pl ${wsj0dir}/??-{?,??}.? ${wsj1dir}/??-{?,??}.? | sort | \
 grep -v -i 11-2.1/wsj0/si_tr_s/401 > train_wsj1.flist
cat si_tr_s.flist train_wsj1.flist  | sort > si_tr_s_wsj0_wsj1.flist

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
for f in si_tr_s si_et_05 si_dt_05 si_tr_s_wsj0_wsj1; do

  $local/flist2scp.pl ${f}.flist | sort > ${f}.scp
  cat ${f}.scp | awk '{print $1}' | $local/find_transcripts.pl dot_files.flist > ${f}.trans1
  cat ${f}.scp | awk '{print $1}' | perl -ane 'chop; m:^...:; print "$_ $&\n";' > ../data/$f/utt2spk
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${f}.scp > ../data/${f}/wav.scp

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  noiseword="<NOISE>"
  cat ${f}.trans1 | $local/normalize_transcript.pl $noiseword | sort > ../data/${f}/text || exit 1;
done

# change to the original path
cd ..

for x in si_tr_s si_dt_05 si_et_05 si_tr_s_wsj0_wsj1; do
  utt2spk_to_spk2utt.pl ${data}/$x/utt2spk > ${data}/$x/spk2utt
done
