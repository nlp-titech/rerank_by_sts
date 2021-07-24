#!/bin/bash

TYPE=$1
OUTDIR=$2
PRETRAIN_MODEL=$3
BATCH_SIZE=$4

if [ $TYPE = "robust04" ]; then
  DINDIR="/PATH/TO/ROBUST04/DOC_DIR"
  QINDIR="/PATH/TO/ROBUST04/QUERY_DIR"
  retrieval_path="/PATH/TO/ROBUST04_RUN"
elif [ $TYPE = "msmarco_passage_dev" ]; then
  DINDIR="/PATH/TO/MSMARCOPASSAGE/DOC_DIR"
  QINDIR="/PATH/TO/MSMARCOPASSAGE/QUERY_DIR"
  retrieval_path="/PATH/TO/MSMARCOPASSAGE_DEV_RUN"
elif [ $TYPE = "msmarco_passage_test" ]; then
  DINDIR="/PATH/TO/MSMARCOPASSAGE/DOC_DIR"
  QINDIR="/PATH/TO/MSMARCOPASSAGE/QUERY_DIR"
  retrieval_path="/PATH/TO/MSMARCOPASSAGE_TEST_RUN"
elif [ $TYPE = "msmarco_document_test" ]; then
  DINDIR="/PATH/TO/MSMARCODOC/DOC_DIR"
  QINDIR="/PATH/TO/MSMARCODOC/QUERY_DIR"
  retrieval_path="/PATH/TO/MSMARCODOC_TEST_RUN"
fi

echo $DINDIR
for qfilepath in `\find  ${QINDIR} -maxdepth 1 -name "*.tsv" -type f` ; do
  bash run_convert_text2rep.sh $DINDIR $qfilepath $OUTDIR $PRETRAIN_MODEL $retrieval_path $BATCH_SIZE
done
