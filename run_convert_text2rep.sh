#!/bin/bash

#$ -l s_gpu=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -M iida.h.ac@m.titech.ac.jp

dinput=$1
qinput=$2
OUTDIR=$3
PRETRAIN_MODEL=$4
RETRIEVAL_PATH=$5
BATCH_SIZE=$6

. /etc/profile.d/modules.sh
module load gcc/8.3.0
module load python/3.9.2
module load cuda/11.0.194
module load openjdk/1.8.0.242
source ~/env/rerank_sts/bin/activate
module load intel

echo $dinput
echo $qinput
echo $OUTDIR
echo $PRETRAIN_MODEL
echo $BATCH_SIZE


python convert_text2rep.py \
 -d ${dinput} \
 -q ${qinput} \
 -o ${OUTDIR} \
 -p ${PRETRAIN_MODEL} \
 -rp ${RETRIEVAL_PATH} \
 -m /home/gaia_data/iida.h/msmarco/passage/trained_model/fast-text_msmarco-finetune.bin \
 --batch_size ${BATCH_SIZE}
