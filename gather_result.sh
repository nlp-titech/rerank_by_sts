TYPE=$1

# pretrain_model=("sbert" "mpnet" "fast_text")
pretrain_model=("sbert_gen" "sbert")
# pretrain_model=("sbert")

if [ $TYPE = "robust04" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/robust04/"
  RTYPE="result"
elif [ $TYPE = "robust04_1000" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/robust04_1000/"
  RTYPE="result"
elif [ $TYPE = "msmarco_passage_dev" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage/"
  RTYPE="result_dev"
  DTYPE="dev"
elif [ $TYPE = "msmarco_passage_test" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_passage/"
  RTYPE="result_test"
  DTYPE="test"
elif [ $TYPE = "msmarco_document_dev" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rearnk_by_sts/msmarco_document/"
  RTYPE="result_dev"
  DTYPE="dev"
elif [ $TYPE = "msmarco_document_test" ]; then
  INDIR="/gs/hs0/tga-nlp-titech/iida.h/rerank_by_sts/msmarco_document/"
  RTYPE="result_test"
  DTYPE="test"
fi


for pm in ${pretrain_model[@]}
do
    input_dir=${INDIR}/${pm}/${RTYPE}
    echo $input_dir
    python gather_result_each_case.py -r ${input_dir}
done

if [ $TYPE = "robust04" ] || [ $TYPE = "robust04_1000" ]; then
  python gather_result_dataset.py -d ${INDIR}
else
  python gather_result_dataset.py -d ${INDIR} -t ${DTYPE}
fi
