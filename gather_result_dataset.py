import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd
from load_model import SBERT, MPNET, FAST_TEXT

EMBEDDINGS = [SBERT, MPNET, FAST_TEXT]


def main(args):
    dataset_path = Path(args.dataset_path)
    dataset_result = defaultdict(list)
    for e in EMBEDDINGS:
        embed_path = dataset_path / Path(e)
        data_pathes = embed_path.glob("**/all_result.csv")
        for dp in data_pathes:
            dataset_name = dp.parts[-2]
            df = pd.read_csv(dp)
            columns = df.columns
            upper_columns = [e] * len(df.columns)
            idx = pd.MultiIndex.from_arrays([upper_columns, columns])
            df.columns = idx
                
            dataset_result[dataset_name].append(df)

    for k, v in dataset_result.items():
        if "_" in k:
            name = k.split("_")[-1]

        outpath = dataset_path / f"{name}_result.csv"
        out = pd.concat(list(v), axis=1)
        out.to_csv(out, outpath)


if __name__ == "__main__":
    parser = argparse.Argumement()

    parser.add_argument("-d", dest="dataset_path")

    args = parser.parse_args()
    main(args)