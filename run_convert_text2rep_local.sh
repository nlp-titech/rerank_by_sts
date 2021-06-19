OUTDIR="./test/msmarco_document/sbert/"
# OUTDIR="./test/robust04/fast_text"
#  -d /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/collection_jsonl/docs \
# -q /home/gaia_data/iida.h/msmarco/passage/collection_and_queries/msmarco-test2019-queries.tsv \
#  -rp /home/gaia_data/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.test2019.bm25-tuned.trec \



python convert_text2rep.py \
 -d /home/gaia_data/iida.h/msmarco/document/msmarco-docs-json-all-in-contents/ \
 -q /home/gaia_data/iida.h/msmarco/document/msmarco-test2019-queries.tsv \
 -rp /home/gaia_data/iida.h/msmarco/document/run.msmarco-doc.test2019.bm25-tuned.trec \
 -o ${OUTDIR} \
 -p sbert \
 -m /home/gaia_data/iida.h/msmarco/passage/trained_model/crawl-300d-2M-subword_msmarco_passage_finetune_analzer.bin
 # --batch_size 128