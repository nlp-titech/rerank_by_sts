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
$ bash run_convert_text2rep.sh $DOCUMENT_PATH $QUERY_PATH $OUTPUT_DIR $PRETRAIN_MODEL $RETRIEVAL_PATH $BATCH_SIZE
```

The file of `RETRIEVAL_PATH` includes the result of bm25 top-n with trec format. You can also make this file by [`anserini`](https://github.com/castorini/anserini)

Actually `DOCUMENT_PATH`, `QUERY_PATH` and `RETRIEVA_PATH` is fixed with target task. You can use `runs/run_convert_text2rep_dataset.sh` as an utility tool


## run_experiment
Please execute `run_rep_rerank_<task_name>.sh` for each target task. Please set `DOC_PATH`, `QUERY_PATH`, `RUN_RETRIEVER_PATH`, `QREL_PATH`, `ROOT_DIR` in `run_rep_rerank_<task_name>.sh`. You nees to set `ROOT_DIR` to the same path as `${EXPERIMENT_ROOT_DIR}/<task_name>` where you set the path as rewrite path in the bash file, and execute it like following.

```
$ cd runs/
$ bash run_rep_rerank_<task_name>.sh $FUNC $POOLER $PRETRAIN_MODEL $USE_IDF 
```

To execute experiments with all parameter, you can use `run_rep_rerank_all.sh`.

```
$ cd runs/
$ bash run_rep_rerank_all.sh <task_name>
```

You need to use the same `<task_name>` as the `run_rep_rerank_<task_name>.sh`


## evaluation
After running experiment, execute `gather_result.sh` to output the final evaluation. 
Firstly, please set `INDIR` to the same path as `${EXPERIMENT_ROOT_DIR}/<task_name>` in `gather_result.sh`. Then execute the script like the following

```
$ cd runs/
$ bash gather_result.sh <task_name>
```

The script outputs the result of each pretrain_model at `INDIR/$PRETRAIN_MODEL/all_result.csv`.
For msmarco, the result of all parameters is output at `INDIR/<dev/test> _result.csv`.
For robust04, the result of all parameters is output at `INDIR/result.csv`.
To execute evaluation on msmaro passage dev, please run `gathre_result_mrr.sh`.

## significance test
The code for significance test is `run_statistical_significance.sh`
Fill in the blank path following the previous setting and execute it.

```
$ cd runs/
$ bash run_statistical_significance.sh <task_name>
```

To execute significance test on msmaro passage dev, please run `run_stats_significance_msmarco_passage_dev.sh`.

