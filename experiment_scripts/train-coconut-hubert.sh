#!/bin/bash

#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -g ge43
#PJM -L jobenv=singularity
#PJM -j

module load gcc/8.3.1
module load cuda/11.8
module load singularity/3.9.5

cd .

export PYTHONPATH=`pwd`/src/laion_clap:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DISABLE_ADDMM_CUDA_LT=1
for i in 0.0 0.5 1.0
do
    echo "Begin with ${i} * feature prediction loss"
    singularity exec --bind `pwd` --nv ../env \
    python src/laion_clap/training/main.py \
        --save-frequency 5 \
        --save-top-performance 3 \
        --top-k-checkpoint-select-metric="coco-Nut-test/text_to_audio_mAP@10" \
        --save-most-recent \
        --dataset-type="webdataset" \
        --datasetpath="full/path/to/parent/directory/of/dataset" \
        --precision="fp32" \
        --batch-size=48 \
        --lr=5e-6 \
        --wd=0.0 \
        --epochs=90 \
        --workers=8 \
        --use-bn-sync \
        --amodel HuBERT \
        --tmodel roberta \
        --warmup 3200 \
        --datasetnames "Coco-Nut_CLAP" \
        --datasetinfos "train" \
        --logs 'logs_coconut' \
        --seed 3407 \
        --gather-with-grad \
        --optimizer "adam" \
        --data-filling "pad" \
        --data-truncating "rand_trunc" \
        --feature-loss-ratio $i \
        # --separated-feature-predictor # False 
        # --clap-mlploss
    
    singularity exec --bind `pwd` --nv ../env \
    python src/laion_clap/training/main.py \
        --save-frequency 5 \
        --save-top-performance 3 \
        --top-k-checkpoint-select-metric="coco-Nut-test/text_to_audio_mAP@10" \
        --save-most-recent \
        --dataset-type="webdataset" \
        --datasetpath="full/path/to/parent/directory/of/dataset" \
        --precision="fp32" \
        --batch-size=48 \
        --lr=5e-6 \
        --wd=0.0 \
        --epochs=90 \
        --workers=8 \
        --use-bn-sync \
        --amodel HuBERT \
        --tmodel roberta \
        --warmup 3200 \
        --datasetnames "Coco-Nut_CLAP" \
        --datasetinfos "train" \
        --logs 'logs_coconut' \
        --seed 3407 \
        --gather-with-grad \
        --optimizer "adam" \
        --data-filling "pad" \
        --data-truncating "rand_trunc" \
        --feature-loss-ratio $i \
        --separated-feature-predictor
        # --clap-mlploss
done