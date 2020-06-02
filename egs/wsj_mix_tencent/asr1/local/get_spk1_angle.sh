#!/bin/bash

. path.sh
. cmd.sh

for x in tr dt et; do
  rm -rf angle_${x}
  src_dir=/data1/v_aswin/simuData_WSJ_Aswin/outWavs/wsj_${x}
  for f in `ls ${src_dir}`; do
    filename=$(basename ${f} .wav)
    tgtname=`echo $filename | cut -d "_" -f1-3 | sed 's/SPH_//g' | sed 's/\./_/g'`
    angle1=`echo $filename | sed 's/spk1-/$/g' | cut -d '$' -f2 | cut -d '.' -f1`
    echo "$tgtname $angle1" >> angle1_${x}
    angle2=`echo $filename | sed 's/spk2-/$/g' | cut -d '$' -f2 | cut -d '.' -f1`
    echo "$tgtname $angle2" >> angle2_${x}
  done
  cat angle1_${x} | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > data/${x}_multich/angle1
  cat angle2_${x} | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[2],1,3); print(spk"_"$0)}' | sort > data/${x}_multich/angle2
done
