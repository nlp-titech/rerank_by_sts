#!/bin/bash

#$ -l s_core=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -M iida.h.ac@m.titech.ac.jp

. /etc/profile.d/modules.sh
module load gcc/8.3.0
module load python/3.9.2
module load cuda/11.0.194
source ~/.bash_profile
source ~/env/rerank_sts/bin/activate
module load intel


# bash run_stats_significance_msmarco_passage_dev.sh
# bash run_stats_significance.sh msmarco_passage_test
# bash run_stats_significance.sh msmarco_document_test
# bash run_stats_significance.sh robust04
bash run_stats_significance.sh robust04_1000
# bash run_stats_significance_rm3.sh msmarco_passage_dev
# bash run_stats_significance_rm3.sh msmarco_passage_test
# bash run_stats_significance_rm3.sh msmarco_document_test
# bash run_stats_significance_rm3.sh robust04
