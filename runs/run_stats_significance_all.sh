#!/bin/bash

bash run_stats_significance_msmarco_passage_dev.sh
bash run_stats_significance.sh msmarco_passage_test
bash run_stats_significance.sh msmarco_document_test
bash run_stats_significance.sh robust04