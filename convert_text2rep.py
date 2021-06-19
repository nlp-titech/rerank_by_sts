import argparse
from pathlib import Path
import torch
import numpy as np

from tqdm import tqdm

from load_model import load_model, load_tokenizer, SBERT, MPNET, FAST_TEXT, SBERT_GEN
from load_data import load_doc, load_query, load_retreival_result
from file_path_setteing import DOC, QUERY


TRUNCATE_LENGTH = 16384
MAX_LENGTH = 512
Q_MODE = "query"
D_MODE = "doc"
B_MODE = "both"


def encode_and_save_batch(output_dir, model, t_batch_doc, max_doc_length, att_masks, batch_doc_did, device):
    if max_doc_length > MAX_LENGTH:
        embeds = []
        for ti in range(0, max_doc_length, MAX_LENGTH):
            input_doc = dict()
            for k in t_batch_doc:
                input_doc[k] = t_batch_doc[k][:, ti : ti + MAX_LENGTH].to(device)
            with torch.no_grad():
                part_embeds = model(**input_doc)["last_hidden_state"]

            embeds.append(part_embeds)

        embeds = torch.cat(embeds, dim=1).cpu().numpy()

    else:
        for k in t_batch_doc:
            t_batch_doc[k] = t_batch_doc[k].to(device)

        with torch.no_grad():
            embeds = model(**t_batch_doc)["last_hidden_state"].cpu().numpy()

    for did, embed, t_att in zip(batch_doc_did, embeds, att_masks):
        out_file = output_dir / f"{did}.npz"
        # zarr.convenience.save(str(out_file), embed[t_att == 1, :])
        np.savez(out_file, embed[t_att == 1, :])


def encode_and_save_query_bert(output_dir, docs, batch_size, tokenizer, model, device):
    ids = list(docs.keys())
    for i in tqdm(range(0, len(ids), batch_size)):
        batch_doc_did = ids[i : i + batch_size]
        batch_doc = [docs[did] for did in batch_doc_did]
        t_batch_doc = tokenizer(
            batch_doc, truncation=True, padding=True, return_tensors="pt", max_length=TRUNCATE_LENGTH
        )
        att_masks = t_batch_doc["attention_mask"].numpy()
        max_doc_length = t_batch_doc["input_ids"].shape[-1]
        encode_and_save_batch(output_dir, model, t_batch_doc, max_doc_length, att_masks, batch_doc_did, device)


def encode_and_save_query_w2v(output_dir, docs, tokenizer, model):
    dids = list(docs.keys())
    for did in dids:
        doc = docs[did]
        t_doc = tokenizer(doc)
        embed = np.array([model[t] for t in t_doc])
        out_file = output_dir / f"{did}.npz"
        np.savez(out_file, embed)


def encode_batch(store_dict, model, t_batch_doc, max_doc_length, att_masks, batch_doc_did, device):
    if max_doc_length > MAX_LENGTH:
        embeds = []
        for ti in range(0, max_doc_length, MAX_LENGTH):
            input_doc = dict()
            for k in t_batch_doc:
                input_doc[k] = t_batch_doc[k][:, ti : ti + MAX_LENGTH].to(device)
            with torch.no_grad():
                part_embeds = model(**input_doc)["last_hidden_state"]

            embeds.append(part_embeds)

        embeds = torch.cat(embeds, dim=1).cpu().numpy()

    else:
        for k in t_batch_doc:
            t_batch_doc[k] = t_batch_doc[k].to(device)

        with torch.no_grad():
            embeds = model(**t_batch_doc)["last_hidden_state"].cpu().numpy()

    for did, embed, t_att in zip(batch_doc_did, embeds, att_masks):
        store_dict[did] = embed[t_att == 1, :]


def encode_and_save_doc_bert(output_dir, batch_size, tokenizer, qid, dids, docs, model, device):
    store_dict = dict()
    outfile = output_dir / f"{qid}.npz"
    for i in range(0, len(dids), batch_size):
        batch_doc_did = dids[i : i + batch_size]
        batch_doc = [docs[did] for did in batch_doc_did]
        t_batch_doc = tokenizer(
            batch_doc, truncation=True, padding=True, return_tensors="pt", max_length=TRUNCATE_LENGTH
        )
        att_masks = t_batch_doc["attention_mask"].numpy()
        max_doc_length = t_batch_doc["input_ids"].shape[-1]
        encode_batch(store_dict, model, t_batch_doc, max_doc_length, att_masks, batch_doc_did, device)

    np.savez_compressed(outfile, **store_dict)


def encode_and_save_doc_w2v(output_dir, tokenizer, qid, dids, docs, model):
    store_dict = dict()
    outfile = output_dir / f"{qid}.npz"
    for did in dids:
        t_doc = tokenizer(docs[did])
        store_dict[did] = np.array([model[t] for t in t_doc])

    np.savez_compressed(outfile, **store_dict)


def encode_and_save_retrieval(
    output_dir, queries, docs, retrieval_result, batch_size, tokenizer, model, pretrain_model, device=None
):
    for qid in tqdm(queries.keys()):
        dids = retrieval_result[qid]
        if pretrain_model in {SBERT, MPNET}:
            encode_and_save_doc_bert(output_dir, batch_size, tokenizer, qid, dids, docs, model, device)
        elif pretrain_model in {FAST_TEXT}:
            encode_and_save_doc_w2v(output_dir, tokenizer, qid, dids, docs, model)


def main(args):
    input_doc_path = Path(args.input_doc)
    output_dir = Path(args.output_dir)
    query_path = Path(args.query_path)
    model_path = args.model_path
    batch_size = args.batch_size
    pretrain_model = args.pretrain_model
    retrieval_result_path = Path(args.first_rank_path)

    if pretrain_model in {SBERT, MPNET, SBERT_GEN}:
        model = load_model(pretrain_model, model_path)
        tokenizer = load_tokenizer(pretrain_model)
        device = torch.device("cuda")
        model.to(device)
    else:
        model = load_model(pretrain_model, model_path)
        tokenizer = load_tokenizer(pretrain_model)
        device = None

    queries = load_query(query_path)

    if args.mode != Q_MODE:
        docs = load_doc(input_doc_path)

        if retrieval_result_path.suffix == ".trec":
            trec = True
        else:
            trec = False

        retrieval_result, retrieval_score = load_retreival_result(retrieval_result_path, trec)

    if args.mode in {Q_MODE, B_MODE}:
        q_output_dir = output_dir / QUERY
        q_output_dir.mkdir(exist_ok=True, parents=True)

        if pretrain_model in {SBERT, MPNET, SBERT_GEN}:
            encode_and_save_query_bert(q_output_dir, queries, batch_size, tokenizer, model, device)
        elif pretrain_model in {FAST_TEXT}:
            encode_and_save_query_w2v(q_output_dir, queries, tokenizer, model)

    if args.mode in {D_MODE, B_MODE}:
        d_output_dir = output_dir / DOC
        d_output_dir.mkdir(exist_ok=True, parents=True)
        encode_and_save_retrieval(
            d_output_dir, queries, docs, retrieval_result, batch_size, tokenizer, model, pretrain_model, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="input_doc")
    parser.add_argument("-q", dest="query_path")
    parser.add_argument("-o", dest="output_dir")
    parser.add_argument("-m", dest="model_path", default="")
    parser.add_argument("-p", dest="pretrain_model", default="")
    parser.add_argument("-rp", dest="first_rank_path")
    parser.add_argument("--mode", default="both")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    main(args)
