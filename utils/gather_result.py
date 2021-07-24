import argparse
from pathlib import Path
import pandas as pd


def main(args):
    indir = Path(args.indir)
    p_indir = indir.parts
    result_files = indir.glob("**/rerank_trec_eval.txt")
    all_params = []
    out_df = pd.DataFrame()
    for rf in result_files:
        params = rf.parts[len(p_indir): -1]
        result = dict()
        with rf.open() as f:
            for line in f:
                k, _, v = line.split()
                result[k] = v

        out_df = pd.concat((out_df, pd.Series(result)), axis=1)
        all_params.append("_".join(params))

    out_df.columns = all_params
    out_df.T.to_csv(args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest="indir")
    parser.add_argument("-o", dest="out_file")

    args = parser.parse_args()

    main(args)
