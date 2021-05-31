import argparse
import json


def main(args):

    with open(args.run_score) as f:
        scores = json.load(f)

    with open(args.output_path, "w") as g:
        for sqid, sdids in scores.items():
            hit_ids = [(sd, score) for sd, score in sorted(sdids.items(), key=lambda x: -x[1])][: args.top_k]
            for i, hi in enumerate(hit_ids):
                # line = "\t".join((sqid, hi[0], str(i + 1), str(hi[1])))
                line = "\t".join((sqid, hi[0], str(i + 1)))
                print(line, file=g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", dest="run_score")
    parser.add_argument("-o", dest="output_path")
    parser.add_argument("-k", dest="top_k", type=int, default=1000)

    args = parser.parse_args()

    main(args)