#!/bin/bash

. path.sh
. cmd.sh

#for x in tr_si284 dev93 eval92_HI eval92_LI eval92_HN eval92_LN; do
for x in wsj_si_tr_s wsj_dt05 wsj_et05_HI wsj_et05_LI wsj_et05_HN wsj_et05_LN; do
  rm -rf angle_${x}
  src_dir=/export/c04/asubraman/simuData_WSJ_Aswin/outWavs/${x}
  for f in `ls ${src_dir}`; do
    filename=$(basename ${f} .wav)
    tgtname=`echo $filename | cut -d "_" -f1-3 | sed 's/SPH_//g' | sed 's/\./_/g'`
    tgtinfo=`echo $filename | cut -d "." -f9-10 | sed 's/\./_/g'`
    angle1=`echo $filename | sed 's/spk1-/$/g' | cut -d '$' -f2 | cut -d '.' -f1`
    echo "${tgtname}_${tgtinfo} $angle1" >> angle1_${x}
    angle2=`echo $filename | sed 's/spk2-/$/g' | cut -d '$' -f2 | cut -d '.' -f1`
    echo "${tgtname}_${tgtinfo} $angle2" >> angle2_${x}
  done
  #cat angle1_${x} | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > data/wsj_mix_${x}_multich/angle1
  #cat angle2_${x} | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > data/wsj_mix_${x}_multich/angle2
  cat angle1_${x} | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > data/${x}_multich/angle1
  cat angle2_${x} | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > data/${x}_multich/angle2
done
