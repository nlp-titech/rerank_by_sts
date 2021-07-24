TYPE=$1

pretrain_model=("sbert_gen" "sbert" "mpnet" "fast_text")

if [ $TYPE = "robust04" ]; then
  INDIR="/PATH/TO/DATASET?ROOT"
  RTYPE="result"
elif [ $TYPE = "msmarco_passage_dev" ]; then
  INDIR="/PATH/TO/DATASET?ROOT"
  RTYPE="result_dev"
  DTYPE="dev"
elif [ $TYPE = "msmarco_passage_test" ]; then
  INDIR="/PATH/TO/DATASET?ROOT"
  RTYPE="result_test"
  DTYPE="test"
elif [ $TYPE = "msmarco_document_test" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_document/"
  RTYPE="result_test"
  DTYPE="test"
fi


for pm in ${pretrain_model[@]}
do
    input_dir=${INDIR}/${pm}/${RTYPE}
    echo $input_dir
    python ../utils/gather_result_each_case.py -r ${input_dir}
done

if [ $TYPE = "robust04" ]; then
  python ../utils/gather_result_dataset.py -d ${INDIR}
else
  python ../utils/gather_result_dataset.py -d ${INDIR} -t ${DTYPE}
fi
