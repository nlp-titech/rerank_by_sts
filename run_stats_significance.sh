TYPE=$1

if [ $TYPE = "robust04" ]; then
  qrel_path="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/qrels.robust2004.txt"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/result/run.robust04.100.trec"
  ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/robust04"
  result="result"
  measure="ndcg_cut"
  rank="20"
  outfile="significant_test.txt"
elif [ $TYPE = "robust04_1000" ]; then
  qrel_path="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/qrels.robust2004.txt"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/TREC/robust04/result/run.robust04.trec"
  ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/robust04_1000"
  result="result"
  measure="ndcg_cut"
  rank="20"
  outfile="significant_test.txt"
elif [ $TYPE = "msmarco_passage_dev" ]; then
  qrel_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/qrels.dev.small.tsv"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.dev.small.bm25-tuned.trec"
  ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage"
  result="result_dev"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_dev.txt"
elif [ $TYPE = "msmarco_passage_test" ]; then
  qrel_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/collection_and_queries/2019qrels-pass.txt"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/passage/experiment/bm25/run.msmarco-passage.test2019.bm25-tuned.trec"
  ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage"
  result="result_test"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_test.txt"
elif [ $TYPE = "msmarco_document_dev" ]; then
  qrel_path="/home/gaia_data/iida.h/msmarco/document/msmarco-docdev-qrels.tsv"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/run.msmarco-doc.dev.bm25-tuned.trec"
  ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_document"
  result="result_dev"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_dev.txt"
elif [ $TYPE = "msmarco_document_test" ]; then
  qrel_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/2019qrels-docs.txt"
  retrieval_path="/gs/hs0/tga-nlp-titech/iida.h/msmarco/document/run.msmarco-doc.test2019.bm25-tuned.trec"
  ROOTDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_document"
  result="result_test"
  measure="ndcg_cut"
  rank="10"
  outfile="significant_test_test.txt"
fi


func_g0=("dense")
# pooler_g0=("cls" "ave" "max")
pooler_g0=("ave")

func_g1=("coef_ave" "coef_max" "soft_bm25")
# pooler_g1=("cls" "ave" "max" "token" "local_ave")
# func_g1=("soft_bm25")
 pooler_g1=("token" "local_ave")

# func_g2=("max_soft_dot" "max_soft_cos")
func_g2=("max_soft_cos")
pooler_g2=("token" "local_ave")

# func_g3=("t2t_dot" "t2t_cos" "nwt")
func_g3=("t2t_dot" "t2t_cos" )
# func_g3=("nwt")

# pretrain_model=("sbert_gen" "sbert" "mpnet" "fast_text")
pretrain_model=("sbert_gen" "sbert")

use_idf=(0 1)
# use_idf=(0)

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
        python utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg0/$pl0/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
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
        python utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg1/$pl1/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
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
        python utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg2/$pl2/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
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
      python utils/statistical_significance.py $qrel_path $retrieval_path $ROOTDIR/$pm/$result/$fg3/dummy/$ui/rerank_score_msmarco.trec --measure $measure --rank $rank >> $outpath
    done
  done
done
