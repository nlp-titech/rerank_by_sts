# About
This repository is for our PACLIC 35 paper, [Incorporating Semantic Textual Similarity and Lexical Matching for Information Retrieval]() 

# How to use
## Prepareing Data
### Document Files
We adopt `Anserini's jsonl files` jsonl format as document format. The format is like following. You need "id" field and "contents" field. You can make this style document files about MSMARCO passage and document by following [this instruction](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md)
```
{"id": "0", "contents": "The presence of..."}
```
For robust 04, we prepare `utils/parse_trec_doc_to_json.py` as the converter.

### Prepareing Query
For MSMARCO passage and document, the format of queries' file is officially distributed tsv. To fit Robust04 to the format we use `utils/parse_trec_query2tsv.py`


## calculate stats
It is necessary to calcurate document length and document frequency. We prepare a script `utils/make_stats.py` for the calculation.

Please output the result on `${EXPERIMENT_ROOT_DIR}/${TASK_NAME}/${PRETRAIN_MODEL_NAME}/stats`

Pretrain model is definied in `src/load_model.py`. Please see the code. If you would like to add your own model, just add name and path to the model if it is readable on [transformers](https://github.com/huggingface/transformers).

## encode token vectors
You can encode tokens to vectors by using `run_convert_text2rep.sh`. 

```
$ cd runs/
$ bash run_convert_text2rep.sh $DOCUMENT_PATH $QUERY_PATH $OUTPUT_DIR $PRETRAIN_MODEL RETRIEVAL_PATH BATCH_SIZE
```

The file of `RETRIEVAL_PATH` includes the result of bm25 top-n with trec format. You can also make this file by [`anserini`](https://github.com/castorini/anserini)

Actually `DOCUMENT_PATH`, `QUERY_PATH` and `RETRIEVA_PATH` is fixed with target task. You can use `runs/run_convert_text2rep_dataset.sh` as an utility tool


## run_experiment
Please execute `run_rep_rerank_<task_name>.sh` for each target task. First, please rewrite `ROOT_DIR` to `${EXPERIMENT_ROOT_DIR}` where you set the path as rewrite path in the bash file, and execute it like following.
```
$ cd runs/
$ bash run_rep_rerank_<task_name>.sh $FUNC $POOLER $PRETRAIN_MODEL $USE_IDF 
```

To execute experiments with all parameter, you can use `run_rep_rerank_all.sh`.

```
$ cd runs/
$ bash run_rep_rerank_all.sh <task_name>
```

You need use the same `<task_name>` as the `run_rep_rerank_<task_name>.sh`