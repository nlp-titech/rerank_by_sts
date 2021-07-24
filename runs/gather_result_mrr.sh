TYPE=$1

pretrain_model=("sbert_gen" "sbert" "mpnet" "fast_text")

INDIR="/PATH/TO/MSMARCOPASSAGE/RESULTS/ROOT"
RTYPE="result_dev"
DTYPE="dev"


for pm in ${pretrain_model[@]}
do
    input_dir=${INDIR}/${pm}/${RTYPE}
    echo $input_dir
    python ../utils/gather_result_mrr_each_case.py -r ${input_dir}
done

python ../utils/gather_result_mrr_dataset.py -d ${INDIR} -t ${DTYPE}
