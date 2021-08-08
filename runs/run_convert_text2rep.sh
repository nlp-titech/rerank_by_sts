#!/bin/bash

dinput=$1
qinput=$2
OUTDIR=$3
PRETRAIN_MODEL=$4
RETRIEVAL_PATH=$5
BATCH_SIZE=$6

echo $dinput
echo $qinput
echo $OUTDIR
echo $PRETRAIN_MODEL
echo $BATCH_SIZE


python ../src/convert_text2rep.py \
 -d ${dinput} \
 -q ${qinput} \
 -o ${OUTDIR} \
 -p ${PRETRAIN_MODEL} \
 -rp ${RETRIEVAL_PATH} \
 --batch_size ${BATCH_SIZE}
