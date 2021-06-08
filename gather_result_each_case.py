import argparse
from pathlib import Path
import pandas as pd


def main(args):
    indir = Path(args.root_dir)
    p_indir = indir.parts
    result_files = indir.glob("**/rerank_trec_eval.txt")
    all_params = []
    out_df = pd.DataFrame()
    for rf in result_files:
        params = rf.parts[len(p_indir) :]
        result = dict()
        with rf.open() as f:
            for line in f:
                k, v = line.split()
                result[k] = v

        out_df = pd.concat((out_df, pd.Series(result)), axis=1)
        all_params.append(params)

    out_df.columns = all_params
    output_path = root_dir / "all_result.csv"
    out_df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", dest="root_dir")

    args = parser.parse_args()

    main(args)
