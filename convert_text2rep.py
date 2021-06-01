import argparse
from pathlib import Path
import torch
import numpy as np

from tqdm import tqdm

from load_bert_model import load_model, load_tokenizer
from load_data import load_doc, load_query
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
        out_file = output_dir / f"{did}.npy"
        # zarr.convenience.save(str(out_file), embed[t_att == 1, :])
        np.save(out_file, embed[t_att == 1, :])


def encode_and_save(output_dir, docs, batch_size, tokenizer, model, device):
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


def main(args):
    input_doc_path = Path(args.input_doc)
    output_dir = Path(args.output_dir)
    query_path = Path(args.query_path)
    model_path = args.model_path
    batch_size = args.batch_size
    pretrain_model = args.pretrain_model

    model = load_model(pretrain_model, model_path)
    tokenizer = load_tokenizer(pretrain_model)
    device = torch.device("cuda")
    model.to(device)

    if args.mode == Q_MODE:
        queries = load_query(query_path)

    elif args.mode == D_MODE:
        docs = load_doc(input_doc_path)

    elif args.mode == B_MODE:
        queries = load_query(query_path)
        docs = load_doc(input_doc_path)
    else:
        raise ValueError(f"{args.mode} doesn't exist")

    if args.mode in {Q_MODE, B_MODE}:
        q_output_dir = output_dir / QUERY
        q_output_dir.mkdir(exist_ok=True, parents=True)

        encode_and_save(q_output_dir, queries, batch_size, tokenizer, model, device)

    if args.mode in {D_MODE, B_MODE}:
        d_output_dir = output_dir / DOC
        d_output_dir.mkdir(exist_ok=True, parents=True)
        encode_and_save(d_output_dir, docs, batch_size, tokenizer, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="input_doc")
    parser.add_argument("-q", dest="query_path")
    parser.add_argument("-o", dest="output_dir")
    parser.add_argument("-m", dest="model_path", default="")
    parser.add_argument("-p", dest="pretrain_model", default="")
    parser.add_argument("--mode", default="both")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    main(args)
