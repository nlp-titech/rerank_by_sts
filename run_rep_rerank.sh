OUTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage/sbert/result_dev/max_soft_cos/token/"
ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage/sbert/"

python rep_reranker.py \
 -d /gs/hs0/tga-nlp-titech//iida.h/msmarco/passage/collection_and_queries/collection_jsonl \
 -q /gs/hs0/tga-nlp-titech//iida.h/msmarco/passage/collection_and_queries/queries.dev.small.tsv \
 -o ${OUTDIR} \
 -ed ${ROOTDIR} \
 -sd ${ROOTDIR} \
 -f max_soft_cos \
 -p token \
 -rp /gs/hs0/tga-nlp-titech//iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.dev.small.2000.optbm25.wo-sw.score.tsv \
 -pm sbert \
 --use_idf 1 \

python convert_score2msmarco_format.py -r ${OUTDIR}/rerank_score.json -o ${OUTDIR}/rerank_score_msmarco.tsv

python -m pyserini.eval.convert_msmarco_run_to_trec_run \
 --input ${OUTDIR}/rerank_score_msmarco.tsv  --output ${OUTDIR}/rerank_score_msmarco.trec

${HOME}/work/pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100 -m map -m P.30 -m ndcg_cut.10 \
 /gs/hs0/tga-nlp-titech//iida.h/msmarco/passage/collection_and_queries/qrels.dev.small.tsv \
 ${OUTDIR}/rerank_score_msmarco.trec > ${OUTDIR}/rerank_trec_eval.txt
