from transformers import AutoModel, AutoTokenizer

SBERT = "sbert"
MPNET = "mpnet"


def load_model(pretrain_model_type, model_path=""):
    if pretrain_model_type.lower() == SBERT:
        # model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
        model = AutoModel.from_pretrained("sentence-transformers/nli-mpnet-base-v2")
    elif pretrain_model_type.lower() == MPNET:
        model = AutoModel.from_pretrained("microsoft/mpnet-base")
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return model


def load_tokenizer(pretrain_model_type):
    if pretrain_model_type == SBERT:
        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L12-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/nli-mpnet-base-v2")
    elif pretrain_model_type.lower() == MPNET:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    else:
        raise ValueError(f"{pretrain_model_type} doesn't exeit")

    return tokenizer