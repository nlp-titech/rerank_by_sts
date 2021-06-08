import argparse
from pathlib import Path
import pandas as pd
from load_model import SBERT, MPNET, FAST_TEXT

EMBEDDINGS = [SBERT, MPNET, FAST_TEXT]


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
            df = pd.read_csv(dp)
            columns = df.columns
            upper_columns = [e] * len(df.columns)
            idx = pd.MultiIndex.from_arrays([upper_columns, columns])
            df.columns = idx

            dataset_result.append(df)

    outpath = dataset_root / f"{dataset_type}_result.csv"
    out = pd.concat(list(dataset_result), axis=1)
    out.to_csv(out, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumementParser()

    parser.add_argument("-d", dest="dataset_path")
    parser.add_argument("-t", dest="dataset_type", default="")

    args = parser.parse_args()
    main(args)