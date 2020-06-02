#!/bin/bash

. path.sh
. cmd.sh

for x in tr dt et; do
  wsj_spatialized=/data1/v_aswin/simuData_WSJ_Aswin/outWavs/wsj_${x}
  outputdir=data/wsj_ts/wav/${x}
  mkdir -p ${outputdir}
  $train_cmd log/log.isolate_wav_${x} local/sox_extract.sh ${wsj_spatialized} ${outputdir}
done
