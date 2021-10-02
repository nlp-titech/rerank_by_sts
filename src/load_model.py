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
}

W2V_BASE_MODEL = {FAST_TEXT}


def load_model(pretrain_model_type, model_path=""):
    if pretrain_model_type in BERT_BASE_MODELS:
        model = AutoModel.from_pretrained(BERT_BASE_MODELS[pretrain_model_type])
    elif pretrain_model_type.lower() == FAST_TEXT:
        model = FastText.load(model_path)
        model = model.wv
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return model


def load_tokenizer(pretrain_model_type):
    if pretrain_model_type in BERT_BASE_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODELS[pretrain_model_type])
    elif pretrain_model_type.lower() == FAST_TEXT:
        analyzer = Analyzer(get_lucene_analyzer())
        tokenizer = analyzer.analyze
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return tokenizer