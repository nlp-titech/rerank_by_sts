from transformers import AutoModel, AutoTokenizer
from pyserini.analysis import Analyzer, get_lucene_analyzer
from gensim.models import FastText

SBERT = "sbert"
MPNET = "mpnet"
FAST_TEXT = "fast_text"


def load_model(pretrain_model_type, model_path=""):
    if pretrain_model_type.lower() == SBERT:
        # model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
        model = AutoModel.from_pretrained("sentence-transformers/nli-mpnet-base-v2")
    elif pretrain_model_type.lower() == MPNET:
        model = AutoModel.from_pretrained("microsoft/mpnet-base")
    elif pretrain_model_type.lower() == FAST_TEXT:
        model = FastText.load(model_path)
        model = model.wv
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return model


def load_tokenizer(pretrain_model_type):
    if pretrain_model_type == SBERT:
        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/nli-mpnet-base-v2")
    elif pretrain_model_type.lower() == MPNET:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    elif pretrain_model_type.lower() == FAST_TEXT:
        analyzer = Analyzer(get_lucene_analyzer())
        tokenizer = analyzer.analyze
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return tokenizer