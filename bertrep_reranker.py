import argparse
import json
from pathlib import Path

import numpy as np

import core_rep_rerank
from load_bert_model import load_tokenizer
from load_data import load_doc, load_query, load_retreival_result, load_stats
from file_path_setteing import DOC, QUERY, DF, DOC_LEN, RERANK_SCORE


def main(args):
    doc_path = Path(args.doc_path)
    query_path = Path(args.query_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    embed_dir = Path(args.embed_dir)
    q_embed_dir = embed_dir / QUERY
    d_embed_dir = embed_dir / DOC
    df_path = embed_dir / DF
    doc_len_path = embed_dir / DOC_LEN
    func_mode = args.func_mode
    pooler = args.pooler
    use_idf = args.use_idf
    retrieval_result_path = Path(args.first_rank_path)

    tokenizer = load_tokenizer(args.pretrain_model)

    queries = load_query(query_path)
    docs = load_doc(doc_path)
    df, doc_len = load_stats(df_path, doc_len_path)

    if retrieval_result_path.suffix == ".trec":
        trec = True
    else:
        trec = False

    retrieval_rank, retrieval_score = load_retreival_result(retrieval_result_path, trec)

    idf = dict()
    doc_num = len(docs)

    for k, v in df.items():
        idf[k] = np.log(doc_num / (v + 1))

    idf = core_rep_rerank.min_dict(idf)
    doc_len_ave = np.mean(doc_len)

    reranker = core_rep_rerank.reranker_factory(
        func_mode,
        pooler,
        idf,
        use_idf,
        tokenizer,
        retrieval_rank,
        retrieval_score,
        q_embed_dir,
        d_embed_dir,
        doc_len_ave,
        window=5,
        bm25_k1=0.82,
        bm25_b=0.68,
    )

    scores = reranker.rerank(queries, docs)

    output_path = output_dir / RERANK_SCORE
    with output_path.open(mode="w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="doc_path")
    parser.add_argument("-q", dest="query_path")
    parser.add_argument("-o", dest="output_dir")
    parser.add_argument("-ed", dest="embed_dir")
    parser.add_argument("-f", dest="func_mode")
    parser.add_argument("-p", dest="pooler")
    parser.add_argument("-pm", dest="pretrain_model")
    parser.add_argument("-rp", dest="first_rank_path")
    parser.add_argument("--use_idf", action="store_true")

    args = parser.parse_args()

    main(args)
