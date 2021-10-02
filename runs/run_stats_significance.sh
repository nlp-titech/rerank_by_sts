TYPE=$1

if [ $TYPE = "robust04" ]; then
  qrel_path=
  retrieval_path=
  ROOTDIR=
  result="result"
  measure="ndcg_cut"
  rank="20"
  outfile="significant_test.txt"
elif [ $TYPE = "msmarco_passage_dev" ]; then
  qrel_path=
  retrieval_path=
  ROOTDIR=
  result="result_dev"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_dev.txt"
elif [ $TYPE = "msmarco_passage_test" ]; then
  qrel_path=
  retrieval_path=
  ROOTDIR=
  result="result_test"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_test.txt"
elif [ $TYPE = "msmarco_document_test" ]; then
  qrel_path=
  retrieval_path=
  ROOTDIR=
  result="result_test"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_test.txt"
fi


func_g0=("dense")
pooler_g0=("cls" "ave" "max")

func_g1=("coef_ave" "coef_max" "soft_bm25")
pooler_g1=("cls" "ave" "max" "token" "local_ave")

func_g2=("max_soft_dot" "max_soft_cos")
pooler_g2=("token" "local_ave")

func_g3=("t2t_dot" "t2t_cos" )

pretrain_model=("sbert_gen" "sbert" "mpnet" "fast_text")

use_idf=(0 1)

outpath=$ROOTDIR/$outfile

echo "significant test" > $outpath


for fg0 in ${func_g0[@]}
do
  for pl0 in ${pooler_g0[@]}
  do
    for pm in ${pretrain_model[@]}
    do
      for ui in ${use_idf[@]}
      do
        echo "$fg0-$pl0-$pm-$ui" >> $outpath
        echo "$fg0-$pl0-$pm-$ui"
        python ../utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg0/$pl0/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
      done
    done
  done
done



for fg1 in ${func_g1[@]}
do
  for pl1 in ${pooler_g1[@]}
  do
    for pm in ${pretrain_model[@]}
    do
      for ui in ${use_idf[@]}
      do
        echo "$fg1-$pl1-$pm-$ui" >> $outpath
        echo "$fg1-$pl1-$pm-$ui"
        python ../utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg1/$pl1/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
      done
    done
  done
done

for fg2 in ${func_g2[@]}
do
  for pl2 in ${pooler_g2[@]}
  do
    for pm in ${pretrain_model[@]}
    do
      for ui in ${use_idf[@]}
      do
        echo "$fg2-$pl2-$pm-$ui" >> $outpath
        echo "$fg2-$pl2-$pm-$ui"
        python ../utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg2/$pl2/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
      done
    done
  done
done

for fg3 in ${func_g3[@]}
do
  for pm in ${pretrain_model[@]}
  do
    for ui in ${use_idf[@]}
    do
      echo "$fg3-$pm-$ui" >> $outpath
      echo "$fg3-$pm-$ui"
      python ../utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg3/dummy/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
    done
  done
done
