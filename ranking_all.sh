#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -cwd
#$ -o job.log

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10
module load cuda/12.0/12.0.0
module load cudnn/8.9/8.9.2

source path/to/activate

export PYTHONPATH="/path/to/src/laion_clap:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python ranking_all.py \
    --ckpt_path=/path/to/ckpt \
    --test_path=/path/to/Coco-Nut/test.csv \
    --wav_dir=/path/to/Coco-Nut/Wave \
    --save_rank=ranking_all.csv