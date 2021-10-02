CORPUS=$1

func_g0=("dense")
pooler_g0=("cls" "ave" "max")

func_g1=("coef_ave" "coef_max" "soft_bm25")
pooler_g1=("cls" "ave" "max" "token" "local_ave")

func_g2=("max_soft_dot" "max_soft_cos")
pooler_g2=("token" "local_ave")

func_g3=("t2t_dot" "t2t_cos")

pretrain_model=("sbert_gen" "sbert" "mpnet" "fast_text")

use_idf=(0 1)


for fg0 in ${func_g0[@]}
do
  for pl0 in ${pooler_g0[@]}
  do
    for pm in ${pretrain_model[@]}
    do
      for ui in ${use_idf[@]}
      do
        bash run_rep_rerank_${CORPUS}.sh $fg0 $pl0 $pm $ui
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
        bash run_rep_rerank_${CORPUS}.sh $fg1 $pl1 $pm $ui
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
        bash run_rep_rerank_${CORPUS}.sh $fg2 $pl2 $pm $ui
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
      bash run_rep_rerank_${CORPUS}.sh $fg3 "dummy" $pm $ui
    done
  done
done
