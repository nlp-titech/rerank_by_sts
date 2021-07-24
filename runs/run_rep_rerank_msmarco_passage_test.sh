#!/bin/bash

ROOTDIR="/PATH/TO/ROOT/"
FUNC=$1
POOLER=$2
PRETRAIN_MODEL=$3
USE_IDF=$4
OUTDIR=${ROOTDIR}/${PRETRAIN_MODEL}/result_test/${FUNC}/${POOLER}/${USE_IDF}
EMBED_DIR=${ROOTDIR}/${PRETRAIN_MODEL}
STATS_DIR=${ROOTDIR}/${PRETRAIN_MODEL}

echo $OUTDIR
echo $EMBED_DIR

DOC_PATH="/PATH/TO/DOC"
QUERY_PATH="/PATH/TO/QUERY"
RUN_RETRIEVER_PATH="/PATH/TO/RUN_RETRIEVER"
QREL_PATH="/PATH/TO/QREL"


python rep_reranker.py \
 -d $DOC_PATH \
 -q $QUERY_PATH \
 -o ${OUTDIR} \
 -ed ${EMBED_DIR} \
 -f ${FUNC} \
 -p ${POOLER} \
 -sd ${STATS_DIR} \
 -rp $RUN_RETRIEVER_PATH\
 -pm ${PRETRAIN_MODEL} \
 --use_idf ${USE_IDF} \
 --top_k 1000


python ../utils/convert_score2msmarco_format.py -r ${OUTDIR}/rerank_score.json -o ${OUTDIR}/rerank_score_msmarco.tsv


python -m pyserini.eval.convert_msmarco_run_to_trec_run \
 --input ${OUTDIR}/rerank_score_msmarco.tsv  --output ${OUTDIR}/rerank_score_msmarco.trec


../anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100 -m map -m P.30 -m ndcg_cut.10 \
 $QREL_PATH ${OUTDIR}/rerank_score_msmarco.trec > ${OUTDIR}/rerank_trec_eval.txt
