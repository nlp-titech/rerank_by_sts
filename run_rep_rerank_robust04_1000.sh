#!/bin/bash

#$ -l s_core=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -M iida.h.ac@m.titech.ac.jp

ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/robust04_1000"
FUNC=$1
POOLER=$2
PRETRAIN_MODEL=$3
USE_IDF=$4
OUTDIR=${ROOTDIR}/${PRETRAIN_MODEL}/result/${FUNC}/${POOLER}/${USE_IDF}
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
 -d /gs/hs0/tga-nlp-titech/iida.h//TREC/robust04/doc_jsonl/all_in_one/ \
 -q /gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/04.testset.tsv \
 -o ${OUTDIR} \
 -ed ${EMBED_DIR} \
 -f ${FUNC} \
 -p ${POOLER} \
 -sd ${STATS_DIR} \
 -rp /gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/result/run.robust04.trec \
 -pm ${PRETRAIN_MODEL} \
 --use_idf ${USE_IDF} \
 --bm25_k1 0.9 \
 --bm25_b 0.4


python convert_score2msmarco_format.py -r ${OUTDIR}/rerank_score.json -o ${OUTDIR}/rerank_score_msmarco.tsv


python -m pyserini.eval.convert_msmarco_run_to_trec_run \
 --input ${OUTDIR}/rerank_score_msmarco.tsv  --output ${OUTDIR}/rerank_score_msmarco.trec


${HOME}/work/pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100 -m map -m P.30 -m ndcg_cut.10  \
 /gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/qrels.robust2004.txt \
 ${OUTDIR}/rerank_score_msmarco.trec > ${OUTDIR}/rerank_trec_eval.txt

${HOME}/work/pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m P.20 -m ndcg_cut.20  \
 /gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/qrels.robust2004.txt \
 ${OUTDIR}/rerank_score_msmarco.trec >> ${OUTDIR}/rerank_trec_eval.txt
