from collections import defaultdict

import numpy as np
from tqdm import tqdm
from convert_text2rep import TRUNCATE_LENGTH


DENSE = "dense"
COEF_AVE = "coef_ave"
COEF_MAX = "coef_max"
SOFT_BM25 = "soft_bm25"
MAX_SOFT_DOT = "max_soft_dot"
MAX_SOFT_COS = "max_soft_cos"
T2T_DOT_MAX = "t2t_dot"
T2T_COS_MAX = "t2t_cos"
NWT = "nwt"

CLS = "cls"
AVE = "ave"
MAX = "max"
TOKEN = "token"
LOCAL_AVE = "local_ave"


class min_dict(dict):
    def __init__(self, data):
        self.data = data
        self.min_v = min(data.values())

    def __str__(self):
        return str(self.data)

    def __getitem__(self, k):
        if k in self.data:
            return self.data[k]
        else:
            return self.min_v


def reranker_factory(
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
    top_k=1000,
    window=5,
    bm25_k1=0.82,
    bm25_b=0.68,
):
    if func_mode == DENSE:
        if pooler == CLS:
            reranker = DENSE_CLS_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == AVE:
            reranker = DENSE_AVE_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window=5,
            )
        elif pooler == MAX:
            reranker = DENSE_MAX_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window=5,
            )
        else:
            raise ValueError(f"{func_mode}-{pooler} doesnlt exist")

    elif func_mode == COEF_AVE:
        # propose1
        if pooler == CLS:
            reranker = CLS_COEF_GLOBAL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == AVE:
            reranker = AVE_COEF_GLOBAL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == MAX:
            reranker = MAX_COEF_GLOBAL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == TOKEN:
            reranker = TOKEN_COEF_POOL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                AVE,
            )
        elif pooler == LOCAL_AVE:
            reranker = LOCAL_AVE_COEF_POOL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                AVE,
            )

    elif func_mode == COEF_MAX:
        # propose1
        if pooler == CLS:
            reranker = CLS_COEF_GLOBAL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == AVE:
            reranker = AVE_COEF_GLOBAL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == MAX:
            reranker = MAX_COEF_GLOBAL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == TOKEN:
            reranker = TOKEN_COEF_POOL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                MAX,
            )
        elif pooler == LOCAL_AVE:
            reranker = LOCAL_AVE_COEF_POOL_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                MAX,
            )

    elif func_mode == SOFT_BM25:
        # propose2
        if pooler == LOCAL_AVE:
            reranker = LOCAL_AVE_SOFT_BM25_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                bm25_k1,
                bm25_b,
            )
        elif pooler == CLS:
            reranker = CLS_SOFT_BM25_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                bm25_k1,
                bm25_b,
            )
        elif pooler == AVE:
            reranker = AVE_SOFT_BM25_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                bm25_k1,
                bm25_b,
            )
        elif pooler == MAX:
            reranker = MAX_SOFT_BM25_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                bm25_k1,
                bm25_b,
            )
        elif pooler == TOKEN:
            reranker = TOKEN_SOFT_BM25_RERANKER(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
                bm25_k1,
                bm25_b,
            )

    elif func_mode == MAX_SOFT_DOT:
        # coil
        if pooler == LOCAL_AVE:
            reranker = LOCAL_AVE_MAX_SOFT_TF_DOT(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == TOKEN:
            reranker = TOKEN_MAX_SOFT_TF_DOT(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )

    elif func_mode == MAX_SOFT_COS:
        # coil-cos
        if pooler == LOCAL_AVE:
            reranker = LOCAL_AVE_MAX_SOFT_TF_COS(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )
        elif pooler == TOKEN:
            reranker = TOKEN_MAX_SOFT_TF_COS(
                idf,
                use_idf,
                tokenizer,
                retrieval_rank,
                retrieval_score,
                q_embed_dir,
                d_embed_dir,
                doc_len_ave,
                top_k,
                window,
            )

    elif func_mode == T2T_DOT_MAX:
        # colbert
        reranker = T2T_DOT_RERANKER(
            idf,
            use_idf,
            tokenizer,
            retrieval_rank,
            retrieval_score,
            q_embed_dir,
            d_embed_dir,
            doc_len_ave,
            top_k,
            window,
        )

    elif func_mode == T2T_COS_MAX:
        # colbert-cos
        reranker = T2T_COS_RERANKER(
            idf,
            use_idf,
            tokenizer,
            retrieval_rank,
            retrieval_score,
            q_embed_dir,
            d_embed_dir,
            doc_len_ave,
            top_k,
            window,
        )

    elif func_mode == NWT:
        # NWT
        reranker = NWT_RERANKER(
            idf,
            use_idf,
            tokenizer,
            retrieval_rank,
            retrieval_score,
            q_embed_dir,
            d_embed_dir,
            doc_len_ave,
            top_k,
            window,
        )

    return reranker


class BERT_REP_RERANKER:
    def __init__(
        self,
        idf,
        use_idf,
        tokenizer,
        retrieval_rank,
        retrieval_score,
        q_embed_dir,
        d_embed_dir,
        doc_len_ave,
        top_k,
        window,
    ):
        self.ret_rank = retrieval_rank
        self.ret_score = retrieval_score
        self.q_embed_dir = q_embed_dir
        self.d_embed_dir = d_embed_dir
        self.idf = idf
        self.use_idf = use_idf
        self.top_k = top_k
        self.doc_len_ave = doc_len_ave
        self.tokenizer = tokenizer
        self.window = window

    def rerank(self, queries, docs):
        all_scores = dict()
        for i, (qid, query) in tqdm(enumerate(queries.items()), total=len(queries)):
            all_scores[str(qid)] = dict()
            doc_ids = self.ret_rank[qid]

            t_query = self.tokenizer(query, truncation=True, max_length=TRUNCATE_LENGTH)
            t_query_id = [str(s) for s in t_query["input_ids"][1:-1]]
            q_embed_path = self.q_embed_dir / f"{qid}.npz"
            # q_embed = zarr.convenience.load(str(q_embed_path))[:]
            q_embed = np.load(q_embed_path)["arr_0"]
            q_rep = self.q_rep_pooler(q_embed, t_query_id)
            d_embeds = np.load(self.d_embed_dir / f"{qid}.npz")
            for j, did in enumerate(doc_ids[: self.top_k]):
                doc = docs[did]
                t_doc = self.tokenizer(doc, truncation=True, max_length=TRUNCATE_LENGTH)
                t_doc_id = [str(s) for s in t_doc["input_ids"][1:-1]]
                t_att_mask = t_doc["attention_mask"][1:-1]
                doc_embed = d_embeds[did]
                d_rep = self.d_rep_pooler(doc_embed, t_doc_id)
                doc_score = self.score_func(q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, j)
                all_scores[qid][did] = float(doc_score)

        return all_scores

    def q_rep_pooler(self, q_embed, t_query_id):
        raise NotImplementedError

    def d_rep_pooler(self, d_embed, t_doc_id):
        raise NotImplementedError

    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        raise NotImplementedError


class DENSE_RERANKER(BERT_REP_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        score = np.dot(q_rep, d_rep)
        return score


class DENSE_CLS_RERANKER(DENSE_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = q_embed[0]
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        d_rep = d_embed[0]
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class DENSE_AVE_RERANKER(DENSE_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        if self.use_idf:
            q_weight = np.array([self.idf[t] for t in t_query_id])
            q_rep = np.average(q_embed[1:-1], axis=0, weights=q_weight)
        else:
            q_rep = np.average(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        if self.use_idf:
            d_weight = np.array([self.idf[t] for t in t_doc_id])
            try:
                d_rep = np.average(d_embed[1:-1], axis=0, weights=d_weight)
            except ValueError:
                print(d_weight.shape, d_embed.shape)
                d_rep = np.average(d_embed[1:-1], axis=0)
        else:
            d_rep = np.average(d_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class DENSE_MAX_RERANKER(DENSE_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        if self.use_idf:
            q_weight = np.array([self.idf[t] for t in t_query_id])
            q_rep = np.max(q_embed[1:-1] * q_weight[:, np.newaxis], axis=0)
        else:
            q_rep = np.max(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        if self.use_idf:
            d_weight = np.array([self.idf[t] for t in t_doc_id])
            try:
                d_rep = np.max(d_embed[1:-1] * d_weight[:, np.newaxis], axis=0)
            except ValueError:
                print(d_weight.shape, d_embed.shape)
                d_rep = np.max(d_embed[1:-1], axis=0)
        else:
            d_rep = np.max(d_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class COEF_GLOBAL_RERANKER(BERT_REP_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        coef = np.dot(q_rep, d_rep)
        return (1 + coef) * self.ret_score[qid][drank]

    def q_rep_pooler(self, q_rep, i):
        raise NotImplementedError

    def d_rep_pooler(self, d_rep, i):
        raise NotImplementedError


class CLS_COEF_GLOBAL_RERANKER(COEF_GLOBAL_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = q_embed[0]
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        d_rep = d_embed[0]
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class AVE_COEF_GLOBAL_RERANKER(COEF_GLOBAL_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        if self.use_idf:
            q_weight = np.array([self.idf[t] for t in t_query_id])
            try:
                q_rep = np.average(q_embed[1:-1], axis=0, weights=q_weight)
            except ValueError:
                print(q_weight.shape, q_embed[1:-1].shape)
                print(t_query_id)
                raise ValueError()
        else:
            q_rep = np.average(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        if self.use_idf:
            d_weight = np.array([self.idf[t] for t in t_doc_id])
            try:
                d_rep = np.average(d_embed[1:-1], axis=0, weights=d_weight)
            except ValueError:
                print(d_weight.shape, d_embed.shape)
                d_rep = np.average(d_embed[1:-1], axis=0)
        else:
            d_rep = np.average(d_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class MAX_COEF_GLOBAL_RERANKER(COEF_GLOBAL_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        if self.use_idf:
            q_weight = np.array([self.idf[t] for t in t_query_id])
            q_rep = np.max(q_embed[1:-1] * q_weight[:, np.newaxis], axis=0)
        else:
            q_rep = np.max(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        if self.use_idf:
            d_weight = np.array([self.idf[t] for t in t_doc_id])
            try:
                d_rep = np.max(d_embed[1:-1] * d_weight[:, np.newaxis], axis=00)
            except ValueError:
                print(d_weight.shape, d_embed.shape)
                d_rep = np.max(d_embed[1:-1], axis=0)
        else:
            d_rep = np.max(d_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class COEF_POOL_RERANKER(BERT_REP_RERANKER):
    def __init__(
        self,
        idf,
        use_idf,
        tokenizer,
        retrieval_rank,
        retrieval_score,
        q_embed_dir,
        d_embed_dir,
        doc_len_ave,
        top_k,
        window,
        coef_pooler_mode,
    ):
        super().__init__(
            idf,
            use_idf,
            tokenizer,
            retrieval_rank,
            retrieval_score,
            q_embed_dir,
            d_embed_dir,
            doc_len_ave,
            top_k,
            window,
        )
        self.coef_pooler_mode = coef_pooler_mode

    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        token_score = defaultdict(list)
        for i, tq in enumerate(t_query_id):
            indexes = [j for j, (td, am) in enumerate(zip(t_doc_id, t_att_mask)) if tq == td and am == 1]

            for k in indexes:
                local_q_rep = self.local_q_rep_pooler(q_rep, i)
                local_d_rep = self.local_d_rep_pooler(d_rep, k)
                score = np.dot(local_q_rep, local_d_rep)
                token_score[tq].append(score)

        coef = np.mean([self.coef_pooler(v) for v in token_score.values()])

        return (1 + coef) * self.ret_score[qid][drank]

    def local_q_rep_pooler(self, q_rep, i):
        raise NotImplementedError

    def local_d_rep_pooler(self, d_rep, i):
        raise NotImplementedError

    def coef_pooler(self, this_token_scores):
        if self.coef_pooler_mode == MAX:
            return np.max(this_token_scores)
        elif self.coef_pooler_mode == AVE:
            return np.average(this_token_scores)


class TOKEN_COEF_POOL_RERANKER(COEF_POOL_RERANKER):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep[i]

    def local_d_rep_pooler(self, d_rep, i):
        return d_rep[i]

    def q_rep_pooler(self, q_embed, t_queru_id):
        return q_embed[1:-1]

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class LOCAL_AVE_COEF_POOL_RERANKER(COEF_POOL_RERANKER):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep

    def local_d_rep_pooler(self, d_rep, i):
        d_start = i - self.window if i - self.window > 0 else 0
        d_end = i + self.window
        d_embed = np.mean(d_rep[d_start:d_end], axis=0)
        d_norm = np.linalg.norm(d_embed)
        if d_norm < 1e-08:
            d_embed = np.zeros(d_embed.shape[0])
        else:
            d_embed /= d_norm

        return d_rep[i]

    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = np.mean(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed


class LOCAL_SOFT_BM25_RERANKER(BERT_REP_RERANKER):
    def __init__(
        self,
        idf,
        use_idf,
        tokenizer,
        retrieval_rank,
        retrieval_score,
        q_embed_dir,
        d_embed_dir,
        doc_len_ave,
        top_k,
        window,
        bm25_k1,
        bm25_b,
    ):
        super().__init__(
            idf,
            use_idf,
            tokenizer,
            retrieval_rank,
            retrieval_score,
            q_embed_dir,
            d_embed_dir,
            doc_len_ave,
            top_k,
            window,
        )
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def calc_bm25_each_token(self, scores, doc_len):
        score = 0
        for t, stf in scores.items():
            score += (
                self.idf[t]
                * stf
                * (1 + self.bm25_k1)
                / (stf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / self.doc_len_ave))
            )
        return score

    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        soft_tf = defaultdict(float)
        doc_len = len(t_doc_id)
        for i, tq in enumerate(t_query_id):
            indexes = [j for j, (td, am) in enumerate(zip(t_doc_id, t_att_mask)) if tq == td and am == 1]

            for k in indexes:
                local_q_rep = self.local_q_rep_pooler(q_rep, i)
                local_d_rep = self.local_d_rep_pooler(d_rep, k)
                score = np.dot(local_q_rep, local_d_rep)
                soft_tf[tq] += np.maximum(score, 0.0)

        doc_score = self.calc_bm25_each_token(soft_tf, doc_len)
        return doc_score

    def local_q_rep_pooler(self, i):
        raise NotImplementedError

    def local_d_rep_pooler(self, i):
        raise NotImplementedError


class TOKEN_SOFT_BM25_RERANKER(LOCAL_SOFT_BM25_RERANKER):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep[i]

    def local_d_rep_pooler(self, d_rep, i):
        return d_rep[i]

    def q_rep_pooler(self, q_embed, t_query_id):
        return q_embed[1:-1]

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class LOCAL_AVE_SOFT_BM25_RERANKER(LOCAL_SOFT_BM25_RERANKER):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep

    def local_d_rep_pooler(self, d_rep, i):
        d_start = i - self.window if i - self.window > 0 else 0
        d_end = i + self.window
        d_embed = np.mean(d_rep[d_start:d_end], axis=0)
        d_norm = np.linalg.norm(d_embed)
        if d_norm < 1e-08:
            d_embed = np.zeros(d_embed.shape[0])
        else:
            d_embed /= d_norm

        return d_rep[i]

    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = np.mean(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed


class GLOBAL_SOFT_BM25_RERANKER(LOCAL_SOFT_BM25_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        soft_tf = defaultdict(float)
        doc_len = len(t_doc_id)
        global_score = np.dot(q_rep, d_rep)
        for i, tq in enumerate(t_query_id):
            indexes = [j for j, (td, am) in enumerate(zip(t_doc_id, t_att_mask)) if tq == td and am == 1]
            for k in indexes:
                soft_tf[tq] += global_score

        doc_score = self.calc_bm25_each_token(soft_tf, doc_len)
        return doc_score


class CLS_SOFT_BM25_RERANKER(GLOBAL_SOFT_BM25_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = q_embed[0]
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        d_rep = d_embed[0]
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class AVE_SOFT_BM25_RERANKER(GLOBAL_SOFT_BM25_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        if self.use_idf:
            q_weight = np.array([self.idf[t] for t in t_query_id])
            q_rep = np.average(q_embed[1:-1], axis=0, weights=q_weight)
        else:
            q_rep = np.average(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        if self.use_idf:
            d_weight = np.array([self.idf[t] for t in t_doc_id])
            try:
                d_rep = np.average(d_embed[1:-1], axis=0, weights=d_weight)
            except ValueError:
                print(d_weight.shape, d_embed.shape)
                d_rep = np.average(d_embed[1:-1], axis=0)
        else:
            d_rep = np.average(d_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class MAX_SOFT_BM25_RERANKER(GLOBAL_SOFT_BM25_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        if self.use_idf:
            q_weight = np.array([self.idf[t] for t in t_query_id])
            q_rep = np.max(q_embed[1:-1] * q_weight[:, np.newaxis], axis=0)
        else:
            q_rep = np.max(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        if self.use_idf:
            d_weight = np.array([self.idf[t] for t in t_doc_id])
            try:
                d_rep = np.max(d_embed[1:-1] * d_weight[:, np.newaxis], axis=0)
            except ValueError:
                print(d_weight.shape, d_embed.shape)
                d_rep = np.max(d_embed[1:-1], axis=0)
        else:
            d_rep = np.max(d_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class MAX_SOFT_TF(BERT_REP_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        soft_tf = defaultdict(list)
        for i, tq in enumerate(t_query_id):
            indexes = [j for j, (td, am) in enumerate(zip(t_doc_id, t_att_mask)) if tq == td and am == 1]

            for k in indexes:
                local_q_rep = self.local_q_rep_pooler(q_rep, i)
                local_d_rep = self.local_d_rep_pooler(d_rep, k)
                score = np.dot(local_q_rep, local_d_rep)
                soft_tf[tq].append(score)

        score = 0.0
        for v, tf_scores in soft_tf.items():
            if self.use_idf:
                score += np.maximum(np.max(tf_scores), 0.0) * self.idf[v]
            else:
                score += np.maximum(np.max(tf_scores), 0.0)

        return score


class TOKEN_MAX_SOFT_TF_COS(MAX_SOFT_TF):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep[i] / np.linalg.norm(q_rep[i])

    def local_d_rep_pooler(self, d_rep, i):
        return d_rep[i] / np.linalg.norm(d_rep[i])

    def q_rep_pooler(self, q_embed, t_query_id):
        return q_embed[1:-1]

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class TOKEN_MAX_SOFT_TF_DOT(MAX_SOFT_TF):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep[i]

    def local_d_rep_pooler(self, d_rep, i):
        return d_rep[i]

    def q_rep_pooler(self, q_embed, t_query_id):
        return q_embed[1:-1]

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class LOCAL_AVE_MAX_SOFT_TF_COS(MAX_SOFT_TF):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep

    def local_d_rep_pooler(self, d_rep, i):
        d_start = i - self.window if i - self.window > 0 else 0
        d_end = i + self.window
        local_d_rep = np.mean(d_rep[d_start:d_end], axis=0)
        d_norm = np.linalg.norm(local_d_rep)
        if d_norm < 1e-08:
            local_d_rep = np.zeros(local_d_rep.shape[0])
        else:
            local_d_rep /= d_norm

        return local_d_rep

    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = np.mean(q_embed[1:-1], axis=0)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class LOCAL_AVE_MAX_SOFT_TF_DOT(MAX_SOFT_TF):
    def local_q_rep_pooler(self, q_rep, i):
        return q_rep

    def local_d_rep_pooler(self, d_rep, i):
        d_start = i - self.window if i - self.window > 0 else 0
        d_end = i + self.window
        local_d_rep = np.mean(d_rep[d_start:d_end], axis=0)
        return local_d_rep

    def q_rep_pooler(self, q_embed, t_query_id):
        q_rep = np.mean(q_embed[1:-1], axis=0)
        return q_rep

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class T2T_RERANKER(BERT_REP_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep, d_rep.T)
        if self.use_idf:
            weight = np.array([self.idf[t] for t, at in zip(t_doc_id, t_att_mask) if at == 1])
            weight /= np.linalg.norm(weight)
        else:
            weight = 1 / len(t_doc_id) * np.ones(len(t_doc_id))
        max_score = np.max(sim_mat, axis=0)
        max_score = np.maximum(max_score, np.zeros(max_score.shape[0]))
        score = np.sum(max_score * weight)
        return score


class T2T_DOT_RERANKER(T2T_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        return q_embed[1:-1]

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1]


class T2T_COS_RERANKER(T2T_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        return q_embed[1:-1] / np.linalg.norm(q_embed[1:-1], axis=1)[:, np.newaxis]

    def d_rep_pooler(self, d_embed, t_doc_id):
        return d_embed[1:-1] / np.linalg.norm(d_embed[1:-1], axis=1)[:, np.newaxis]


class NWT_RERANKER(T2T_COS_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep, d_rep.T)
        argmax_sim_mat = np.argmax(sim_mat, axis=0)
        pow_index = np.array([self.idf[t_query_id[i]] for i in argmax_sim_mat])
        if self.use_idf:
            weight = np.array([self.idf[t] for t, at in zip(t_doc_id, t_att_mask) if at == 1])
            weight /= np.linalg.norm(weight)
        else:
            weight = 1 / len(t_doc_id) * np.ones(len(t_doc_id))
        max_score = np.max(sim_mat, axis=0)
        max_score = np.maximum(max_score, np.zeros(max_score.shape[0]))
        score = np.sum(np.log(np.power(max_score, pow_index) * weight))
        return score


"""
ボツ
class DENSE_QDIDFwAVE_RERANKER(DENSE_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        q_weight = np.array([self.idf[t] for t in t_query_id])
        q_rep = np.average(q_embed[1:-1], axis=0, weight=q_weight)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, doc_embed, t_doc_id):
        d_weight = np.array([self.idf[t] for t in t_doc_id])
        d_rep = np.average(doc_embed[1:-1], axis=0, weight=d_weight)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class DENSE_QIDFwAVE_RERANKER(DENSE_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        q_weight = np.array([self.idf[t] for t in t_query_id])
        q_rep = np.average(q_embed[1:-1], axis=0, weight=q_weight)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, doc_embed, t_doc_id):
        d_rep = np.mean(doc_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep


class DENSE_DIDFwAVE_RERANKER(DENSE_RERANKER):
    def q_rep_pooler(self, q_embed, t_query_id):
        q_weight = np.array([self.idf[t] for t in t_query_id])
        q_rep = np.average(q_embed[1:-1], axis=0, weight=q_weight)
        q_rep /= np.linalg.norm(q_rep)
        return q_rep

    def d_rep_pooler(self, doc_embed, t_doc_id):
        d_rep = np.mean(doc_embed[1:-1], axis=0)
        d_rep /= np.linalg.norm(d_rep)
        return d_rep

class T2T_DOTMAX_RERANKER(T2T_DOT_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep.T, d_rep.T)
        score = np.sum(np.max(sim_mat, axis=0))
        return score


class T2T_DOTMAXDIDF_RERANKER(T2T_DOT_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep.T, d_rep.T)
        weight = np.array([self.idf[t] for t in t_doc_id])
        max_score = np.max(sim_mat, axis=0) * weight
        score = np.sum(max_score)
        return score

class T2T_DOTMAXQIDF_RERANKER(T2T_DOT_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep.T, d_rep.T)
        argmax_sim_mat = np.argmax(sim_mat, axis=0)
        weight = np.array([self.idf[t_query_id[i]] for i in argmax_sim_mat])
        max_score = np.max(sim_mat, axis=0) * weight
        score = np.sum(max_score)
        return score

class T2T_COSMAX_RERANKER(T2T_COS_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep.T, d_rep.T)
        score = np.sum(np.max(sim_mat, axis=0))
        return score


class T2T_COSMAXDIDF_RERANKER(T2T_COS_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep.T, d_rep.T)
        weight = np.array([self.idf[t] for t, at in zip(t_doc_id, t_att_mask) if at == 1])
        weight /= np.linalg.norm(weight)
        max_score = np.max(sim_mat, axis=0) * weight
        score = np.sum(max_score)
        return score

class T2T_COSMAXQIDF_RERANKER(T2T_COS_RERANKER):
    def score_func(self, q_rep, d_rep, t_query_id, t_doc_id, t_att_mask, qid, drank):
        sim_mat = np.dot(q_rep.T, d_rep.T)
        argmax_sim_mat = np.argmax(sim_mat, axis=0)
        weight = np.array([self.idf[t_query_id[i]] for i in argmax_sim_mat])
        max_score = np.max(sim_mat, axis=0) * weight
        score = np.sum(max_score)
        return score
"""
