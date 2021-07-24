import argparse
import json
from pathlib import Path

import numpy as np

import core_rep_bert_rerank
import core_rep_w2v_rerank
from load_model import load_tokenizer, BERT_BASE_MODEL, W2V_BASE_MODEL
from load_data import load_doc, load_query, load_retreival_result, load_stats
from file_path_setteing import DOC, QUERY, DF, DOC_LEN, RERANK_SCORE, STATS


def main(args):
    doc_path = Path(args.doc_path)
    query_path = Path(args.query_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    embed_dir = Path(args.embed_dir)
    stats_dir = Path(args.stats_dir)
    q_embed_dir = embed_dir / QUERY
    d_embed_dir = embed_dir / DOC
    df_path = stats_dir / STATS / DF
    doc_len_path = stats_dir / STATS / DOC_LEN
    func_mode = args.func_mode
    pooler = args.pooler
    use_idf = bool(args.use_idf)
    print(use_idf)
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
    print("doc_num: ", doc_num)

    if func_mode in {core_rep_bert_rerank.NWT, core_rep_bert_rerank.APPROX_NWT}:
        for k, v in df.items():
            idf[k] = np.log((doc_num - v + 0.5) / (v + 0.5))
            if idf[k] < 0.0 or np.isnan(idf[k]):
                idf[k] = 0.0
    else:
        for k, v in df.items():
            idf[k] = np.log(doc_num / v)

    idf = core_rep_bert_rerank.OOV1_dict(idf)
    doc_len_ave = np.mean(doc_len)

    if args.pretrain_model in BERT_BASE_MODEL:
        reranker = core_rep_bert_rerank.reranker_factory(
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
            top_k=args.top_k,
            window=args.window,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )
    elif args.pretrain_model in W2V_BASE_MODEL:
        reranker = core_rep_w2v_rerank.reranker_factory(
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
            top_k=args.top_k,
            window=args.window,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )
    else:
        raise ValueError(f"{args.pretrain_model} mode doesn't exist")

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
    parser.add_argument("-sd", dest="stats_dir")
    parser.add_argument("-f", dest="func_mode")
    parser.add_argument("-p", dest="pooler")
    parser.add_argument("-pm", dest="pretrain_model")
    parser.add_argument("-rp", dest="first_rank_path")
    parser.add_argument("--use_idf", type=int)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--bm25_k1", type=float, default=0.82)
    parser.add_argument("--bm25_b", type=float, default=0.68)

    args = parser.parse_args()

    main(args)
