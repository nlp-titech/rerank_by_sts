from collections import defaultdict
import json

from tqdm import tqdm


def load_query(query_path):
    print("read query")
    queries = dict()
    with query_path.open(mode="r") as f:
        for line in tqdm(f):
            qid, query = line.strip().split("\t")
            queries[qid] = query

    return queries


def load_doc(input_doc_path):
    def read_doc(input_file, docs):
        with input_file.open(mode="r") as f:
            for i, line in tqdm(enumerate(f)):
                jline = json.loads(line)
                text = jline["contents"]
                did = jline["id"]
                docs[did] = text

    print("read doc")
    docs = dict()
    if input_doc_path.is_dir():
        input_doc_files = sorted(input_doc_path.glob("*.json"))
        for input_doc_file in input_doc_files:
            read_doc(input_doc_file, docs)
    else:
        read_doc(input_doc_path, docs)

    return docs


def load_retreival_result(retrieval_result_path, trec):
    retrieval_rank = defaultdict(list)
    retrieval_score = defaultdict(list)

    print("read first result")
    with retrieval_result_path.open(mode="r") as f:
        for line in f:
            chank = line.strip().split()
            if trec:
                qid = chank[0]
                did = chank[2]
                score = float(chank[4])
                retrieval_rank[qid].append(did)
                retrieval_score[qid].append(score)

            else:
                qid = chank[0]
                did = chank[1]
                retrieval_rank[qid].append(did)
                if len(chank) == 4:
                    retrieval_score[qid].append(float(chank[-1]))

    return retrieval_rank, retrieval_score


def load_stats(df_path, doc_len_path):
    with df_path.open() as f:
        df = json.load(f)

    with doc_len_path.open() as f:
        doc_len = json.load(f)

    return df, doc_len
