import argparse
import sys
from pathlib import Path

sys.path.append('../src/')

import pandas as pd
from load_model import BERT_BASE_MODELS, W2V_BASE_MODEL

EMBEDDINGS = set(BERT_BASE_MODELS) | W2V_BASE_MODEL


def main(args):
    dataset_root = Path(args.dataset_path)
    dataset_type = args.dataset_type
    dataset_result = []
    for e in EMBEDDINGS:
        if dataset_type:
            dataset_dir = dataset_root / Path(e) / Path(f"result_{dataset_type}")
        else:
            # for robust04
            dataset_dir = dataset_root / Path(e) / Path("result")
        data_pathes = dataset_dir.glob("**/all_result.csv")
        for dp in data_pathes:
            df = pd.read_csv(dp, index_col=0)
            columns = df.columns
            upper_columns = [e] * len(df.columns)
            idx = pd.MultiIndex.from_arrays([upper_columns, columns])
            df.columns = idx

            dataset_result.append(df)

    if dataset_type:
        outpath = dataset_root / f"{dataset_type}_result.csv"
    else:
        outpath = dataset_root / f"result.csv"

    out = pd.concat(list(dataset_result), axis=1)
    out.to_csv(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="dataset_path")
    parser.add_argument("-t", dest="dataset_type", default="")

    args = parser.parse_args()
    main(args)