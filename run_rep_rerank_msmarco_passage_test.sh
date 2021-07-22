#!/bin/bash

#$ -l s_core=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -M iida.h.ac@m.titech.ac.jp

ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage/"
FUNC=$1
POOLER=$2
PRETRAIN_MODEL=$3
USE_IDF=$4
OUTDIR=${ROOTDIR}/${PRETRAIN_MODEL}/result_test/${FUNC}/${POOLER}/${USE_IDF}
EMBED_DIR=${ROOTDIR}/${PRETRAIN_MODEL}
STATS_DIR=${ROOTDIR}/${PRETRAIN_MODEL}

echo $OUTDIR
echo $EMBED_DIR


. /etc/profile.d/modules.sh
module load gcc/8.3.0
module load python/3.9.2
module load cuda/11.0.194
source ~/.bash_profile
source ~/env/rerank_sts/bin/activate
module load intel


python rep_reranker.py \
 -d /gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/collection_jsonl/ \
 -q /gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/msmarco-test2019-queries.tsv \
 -o ${OUTDIR} \
 -ed ${EMBED_DIR} \
 -f ${FUNC} \
 -p ${POOLER} \
 -sd ${STATS_DIR} \
 -rp /gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.test2019.bm25-tuned.trec \
 -pm ${PRETRAIN_MODEL} \
 --use_idf ${USE_IDF} \
 --top_k 1000


python convert_score2msmarco_format.py -r ${OUTDIR}/rerank_score.json -o ${OUTDIR}/rerank_score_msmarco.tsv


python -m pyserini.eval.convert_msmarco_run_to_trec_run \
 --input ${OUTDIR}/rerank_score_msmarco.tsv  --output ${OUTDIR}/rerank_score_msmarco.trec


${HOME}/work/pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100 -m map -m P.30 -m ndcg_cut.10 \
 /gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/2019qrels-pass.txt \
 ${OUTDIR}/rerank_score_msmarco.trec > ${OUTDIR}/rerank_trec_eval.txt
