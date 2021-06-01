OUTDIR="./test/msmarco-doc/nli_mpnet"


python convert_text2rep.py \
 -d /home/gaia_data/iida.h/msmarco/document/msmarco-docs-jsonl \
 -q /home/gaia_data/iida.h/msmarco/document/msmarco-test2019-queries.tsv \
 -o ${OUTDIR} \
 -p sbert \
 --batch_size 64