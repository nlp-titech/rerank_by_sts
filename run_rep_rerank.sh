OUTDIR="./test/msmarco-passage/nli_mpnet/compress"

python bertrep_reranker.py \
 -d /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/collection_jsonl/docs \
 -q /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/queries.dev.small.tsv \
 -o ${OUTDIR} \
 -ed ./test/msmarco-passage/nli_mpnet/compress \
 -sd ./test/msmarco-passage/ \
 -f coef_ave \
 -p ave \
 -rp /home/gaia_data/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.dev.small.2000.optbm25.wo-sw.score.tsv \
 -pm sbert \
 --use_idf 1 \

python convert_score2msmarco_format.py -r ${OUTDIR}/rerank_score.json -o ${OUTDIR}/rerank_score_msmarco.tsv

python -m pyserini.eval.convert_msmarco_run_to_trec_run \
 --input ${OUTDIR}/rerank_score_msmarco.tsv  --output ${OUTDIR}/rerank_score_msmarco.trec

${HOME}/work/pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100 -m map -m P.30 -m ndcg_cut.10 \
 /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/qrels.dev.small.tsv \
 ${OUTDIR}/rerank_score_msmarco.trec > ${OUTDIR}/rerank_trec_eval.txt