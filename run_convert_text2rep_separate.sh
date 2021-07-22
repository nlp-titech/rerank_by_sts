#!/bin/bash

TYPE=$1
OUTDIR=$2
PRETRAIN_MODEL=$3
BATCH_SIZE=$4

if [ $TYPE = "robust04" ]; then
  DINDIR="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/doc_jsonl/all_in_one"
  QINDIR="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/query/"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/result/run.robust04.100.trec"
elif [ $TYPE = "msmarco_passage_dev" ]; then
  DINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/collection_jsonl/"
  QINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/dev_split_queries/"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.dev.small.bm25-tuned.trec"
elif [ $TYPE = "msmarco_passage_test" ]; then
  DINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/collection_jsonl/"
  QINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/test_query/"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.test2019.bm25-tuned.trec"
elif [ $TYPE = "msmarco_document_dev" ]; then
  DINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document//msmarco-docs-json-all-in-contents/"
  QINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/dev_split_queries/"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/run.msmarco-doc.dev.bm25-tuned.trec"
elif [ $TYPE = "msmarco_document_test" ]; then
  DINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document//msmarco-docs-json-all-in-contents/"
  QINDIR="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/test_query/"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/run.msmarco-doc.test2019.bm25-tuned.trec"
fi

echo $DINDIR
for qfilepath in `\find  ${QINDIR} -maxdepth 1 -name "*.tsv" -type f` ; do
  qsub -g tga-nlp-titech run_convert_text2rep.sh $DINDIR $qfilepath $OUTDIR $PRETRAIN_MODEL $retrieval_path $BATCH_SIZE
done
