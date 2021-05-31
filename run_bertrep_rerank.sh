OUTDIR="./test/robust04"

python bertrep_reranker.py \
 -d /home/gaia_data/iida.h//TREC/robust04/doc_jsonl/all_in_one/all_a.jsonl \
 -q /home/gaia_data/iida.h/TREC/robust04/04.testset.tsv \
 -o ${OUTDIR} \
 -ed ./test/robust04 \
 -f coef_ave \
 -p ave \
 -rp /home/gaia_data/iida.h/TREC/robust04/result/run.robust04.trec \
 -pm sbert \
 --use_idf \