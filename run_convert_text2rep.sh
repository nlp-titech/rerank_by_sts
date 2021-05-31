OUTDIR="./test/msmarco-doc/nli_mpnet"


python convert_text2rep.py \
 -d /home/gaia_data/iida.h/msmarco/document/msmarco-docs-jsonl \
 -q /home/gaia_data/iida.h/msmarco/document/msmarco-test2019-queries.tsv \
 -o ${OUTDIR} \
 -p sbert \
 --batch_size 64

python convert_score2msmarco_format.py -r ${OUTDIR}/rerank_score.json -o ${OUTDIR}/rerank_score_msmarco.tsv

python -m pyserini.eval.convert_msmarco_run_to_trec_run \
 --input ${OUTDIR}/rerank_score_msmarco.tsv  --output ${OUTDIR}/rerank_score_msmarco.trec

${HOME}/work/pyserini/tools/eval/trec_eval.9.0.4/trec_eval -c -m recall.100 -m map -m P.30 -m ndcg_cut.10 -m mrr \
 /home/gaia_data/iida.h/msmarco/document/2019qrels-docs.txt \
 ${OUTDIR}/rerank_score_msmarco.trec > ${OUTDIR}/rerank_trec_eval.txt