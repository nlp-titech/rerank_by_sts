import os
from transformers import AutoModel, AutoTokenizer
from pyserini.analysis import Analyzer, get_lucene_analyzer
from gensim.models import FastText

SBERT = "sbert"
SBERT_GEN = "sbert_gen"
MPNET = "mpnet"
FAST_TEXT = "fast_text"

BERT_BASE_MODELS = {
    "sbert": "sentence-transformers/nli-mpnet-base-v2",
    "sbert_gen": "sentence-transformers/paraphrase-mpnet-base-v2",
    "mpnet": "microsoft/mpnet-base",
    "sbert_msmarco_sts_best": "/home/iida.h/work/sentence-transformers/examples/training/paraphrases/output/training_paraphrases_microsoft-mpnet-base-2021-07-19_23-31-19/",
    "sbert_msmarco_1epoch": "//home/iida.h/work/sentence-transformers/examples/training/paraphrases/output/training_paraphrases_microsoft-mpnet-base-2021-07-19_23-31-19/732490",
    "coil": "/home/iida.h/work/rerank_sts/test/models/coil-hn-checkpoint",
}

# BERT_BASE_MODEL = {SBERT, MPNET, SBERT_GEN}
W2V_BASE_MODEL = {FAST_TEXT}


def load_model(pretrain_model_type, model_path=""):
    if pretrain_model_type in BERT_BASE_MODELS:
        model = AutoModel.from_pretrained(BERT_BASE_MODELS[pretrain_model_type])
    elif pretrain_model_type.lower() == FAST_TEXT:
        model = FastText.load(model_path)
        model = model.wv
    # elif os.path.exists(model_path) and re.match("sbert", pretrain_model_type):
    #     model = AutoModel.from_pretrained(model_path)
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return model


def load_tokenizer(pretrain_model_type):
    if pretrain_model_type in BERT_BASE_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODELS[pretrain_model_type])
    elif pretrain_model_type.lower() == FAST_TEXT:
        analyzer = Analyzer(get_lucene_analyzer())
        tokenizer = analyzer.analyze
    # elif os.path.exists(tokenizer_path) and re.match("sbert", pretrain_model_type):
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return tokenizer