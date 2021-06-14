"""
The MIT License (MIT)

Copyright (c) 2017 Christophe Van Gysel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Demonstrates how statistical significance tests can be ran using pytrec_eval."""

import argparse
import os
import scipy.stats
import sys

import pytrec_eval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('qrel')
    parser.add_argument('run', nargs=2)

    # A bit too strict, as it does not allow for parametrized measures,
    # but sufficient for the example.
    parser.add_argument('--measure',
                        choices=pytrec_eval.supported_measures,
                        required=True)

    args = parser.parse_args()

    assert os.path.exists(args.qrel)
    assert all(map(os.path.exists, args.run))

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.run[0], 'r') as f_run:
        first_run = pytrec_eval.parse_run(f_run)

    with open(args.run[1], 'r') as f_run:
        second_run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {args.measure})

    first_results = evaluator.evaluate(first_run)
    second_results = evaluator.evaluate(second_run)

    query_ids = list(
        set(first_results.keys()) & set(second_results.keys()))

    first_scores = [
        first_results[query_id][args.measure] for query_id in query_ids]
    second_scores = [
        second_results[query_id][args.measure] for query_id in query_ids]

    print(scipy.stats.ttest_rel(first_scores, second_scores))

if __name__ == "__main__":
    sys.exit(main())
