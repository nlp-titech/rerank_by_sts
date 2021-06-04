import argparse
from collections import Counter
import json
from pathlib import Path

from tqdm import tqdm

from load_model import load_tokenizer, SBERT, MPNET, FAST_TEXT
from file_path_setteing import DF, DOC_LEN, STATS


def doc_stats(input_doc_path, tokenizer, pretrain_model):
    def doc_len_and_df_bert(input_file, df, doc_lens):
        with input_file.open(mode="r") as f:
            for i, line in tqdm(enumerate(f)):
                jline = json.loads(line)
                text = jline["contents"]
                t_doc = tokenizer(text)
                doc_lens.append(len(t_doc["input_ids"]))
                df.update(set(t_doc["input_ids"]))

    def doc_len_and_df_w2v(input_file, df, doc_lens):
        with input_file.open(mode="r") as f:
            for i, line in tqdm(enumerate(f)):
                jline = json.loads(line)
                text = jline["contents"]
                t_doc = tokenizer(text)
                doc_lens.append(len(t_doc))
                df.update(set(t_doc))

    doc_lens = list()
    df = Counter()

    if pretrain_model in {SBERT, MPNET}:
        take_stats = doc_len_and_df_bert
    else:
        take_stats = doc_len_and_df_w2v

    if input_doc_path.is_dir():
        input_doc_files = sorted(input_doc_path.glob("*.json"))
        for input_doc_file in input_doc_files:
            take_stats(input_doc_file, df, doc_lens)

    else:
        take_stats(input_doc_path, df, doc_lens)

    return df, doc_lens


def main(args):
    input_doc_path = Path(args.input_doc)
    output_dir = Path(args.output_dir)
    output_dir = output_dir / STATS
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_model = args.pretrain_model

    tokenizer = load_tokenizer(pretrain_model)
    df, doc_lens = doc_stats(input_doc_path, tokenizer, pretrain_model)

    df_path = output_dir / DF
    doc_lens_path = output_dir / DOC_LEN

    with df_path.open(mode="w") as f:
        json.dump(df, f)

    with doc_lens_path.open(mode="w") as f:
        json.dump(doc_lens, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="input_doc")
    parser.add_argument("-o", dest="output_dir")
    parser.add_argument("-p", dest="pretrain_model", default="")

    args = parser.parse_args()

    main(args)
