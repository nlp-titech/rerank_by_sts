OUTDIR="./test/msmarco-passage/fast_text"


python convert_text2rep.py \
 -d /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/collection_jsonl/docs \
 -q /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/queries.dev.small.tsv \
 -rp /home/gaia_data/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.dev.small.2000.optbm25.wo-sw.score.tsv \
 -o ${OUTDIR} \
 -p fast_text \
 -m /home/gaia_data/iida.h/msmarco/passage/trained_model/fast-text_msmarco-finetune.bin
 # --batch_size 128