#!/bin/bash

#PJM -L rscgrp=share-short
#PJM -L gpu=2
#PJM -g ge43
#PJM -L jobenv=singularity
#PJM -j

module load gcc/8.3.1
module load cuda/11.8
module load singularity/3.9.5

cd /work/ge43/e43003/work_clap/CLAP-SpeechJP

export PYTHONPATH=`pwd`/src/laion_clap:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

models=(
# paths to trained models' directories
)

for model in ${models[@]}; do
    echo "begin extraction with" $model
    singularity exec --bind `pwd` --nv ../env \
        python jvs_extractor.py \
            -m $model \
            -j /path/to/jvs \
            -e "40"
done