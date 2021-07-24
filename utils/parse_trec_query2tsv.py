import argparse
import json
import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from tqdm import tqdm


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output_path)

    id_pattern = re.compile("<num> Number: (?P<id>[0-9]*)")
    query_pattern = re.compile("<title>")
    desc_q_pattern = re.compile("<desc>")
    queries = []
    qid = ""
    query = ""
    q_flag = False
    with input_path.open() as f:
        for line in f:
            sline = line.strip()
            id_match = id_pattern.match(sline)
            if id_match:
                qid = id_match.group("id")

            desc_q_match = desc_q_pattern.match(sline)
            if qid and desc_q_match:
                queries.append((qid, query))
                qid = ""
                query = ""
                q_flag = False

            query_match = query_pattern.match(sline)
            if qid and (query_match or q_flag):
                q_flag = True
                query += sline.strip().replace("<title>", "")

    with output_path.open(mode="w") as f:
        for q in queries:
            print("\t".join(q), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest="input")
    parser.add_argument("-o", dest="output_path")

    args = parser.parse_args()

    main(args)