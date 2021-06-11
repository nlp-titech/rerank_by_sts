import argparse
from pathlib import Path
import pandas as pd


def main(args):
    root_dir = Path(args.root_dir)
    p_indir = root_dir.parts
    result_files = root_dir.glob("**/rerank_msmarco_eval.txt")
    all_params = []
    out_df = pd.DataFrame()
    for rf in result_files:
        params = rf.parts[len(p_indir) : -1]
        result = dict()
        with rf.open() as f:
            for line in f:
                if "MRR" in line:
                    score = line.split(":")[-1].stirp()
                    result["MRR@10"] = score

        out_df = pd.concat((out_df, pd.Series(result)), axis=1)
        all_params.append("_".join(params))

    out_df.columns = all_params
    out_df = out_df.T
    output_path = root_dir / "all_result_mrr.csv"
    out_df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", dest="root_dir")

    args = parser.parse_args()

    main(args)
