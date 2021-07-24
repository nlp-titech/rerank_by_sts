import argparse
import json
from pathlib import Path
from collections import Counter
from functools import partial
import concurrent

from tqdm import tqdm
import numpy as np
from gensim.models import word2vec, KeyedVectors, fasttext, Word2Vec, FastText
from pyserini.analysis import Analyzer, get_lucene_analyzer


GLOVE = "glove"
W2V = "w2v"
FAST_TEXT = "fast_text"
FAST_TEXT_KV = "fast_text_kv"


def tokenize_process(text, st_tokenizer):
    t_doc = st_tokenizer([text.lower().split()])
    t_text = [token.text for sentence in t_doc.sentences for token in sentence.tokens]
    return t_text


def w2v_finetune(sentences_tokenized, model, model_path, epoch=3):
    # https://www.kaggle.com/rtatman/fine-tuning-word2vec
    d = model.vector_size
    model_2 = Word2Vec(vector_size=d, min_count=1)
    model_2.build_vocab(sentences_tokenized)
    total_examples = model_2.corpus_count
    model_2.build_vocab([list(model.key_to_index.keys())], update=True)
    print(len(model_2.wv))
    model_2.wv.vectors_lockf = np.ones(len(model_2.wv), dtype=np.float32)
    model_2.wv.intersect_word2vec_format(model_path, binary=True, lockf=1.0)
    model_2.train(sentences_tokenized, total_examples=total_examples, epochs=epoch)
    return model_2


def fasttext_finetune_ftkv(sentences_tokenized, model, model_path, epoch=3):
    # https://www.kaggle.com/rtatman/fine-tuning-word2vec
    d = model.vector_size
    model_2 = FastText(vector_size=d, min_count=1)
    model_2.build_vocab(sentences_tokenized)
    total_examples = model_2.corpus_count
    model_2.build_vocab([list(model.key_to_index.keys())], update=True)
    print(len(model_2.wv))
    model_2.wv.vectors_lockf = np.ones(len(model_2.wv), dtype=np.float32)
    model_2.wv.intersect_word2vec_format(model_path, lockf=1.0)
    model_2.train(sentences_tokenized, total_examples=total_examples, epochs=epoch)
    return model_2


def fasttext_finetune(sentences_tokenized, model, model_path, epoch=3):
    model.build_vocab(sentences_tokenized, update=True)
    total_examples = len(sentences_tokenized)
    print("start train")
    model.train(sentences_tokenized, total_examples=total_examples, epochs=epoch)
    return model


def doc_tokenizer(doc_file, tokenizer):
    tokenized_sentences = []
    with doc_file.open() as f:
        for line in tqdm(f):
            sentence = json.loads(line)["contents"]
            t_sentence = tokenizer(sentence)
            tokenized_sentences.append(t_sentence)

    return tokenized_sentences


def main(args):
    input_doc_path = Path(args.input)
    model_path = args.model_path
    output_model_path = args.output

    print("load model")
    if args.pretrain_model.lower() == GLOVE:
        model = KeyedVectors.load_word2vec_format(model_path, no_header=True)
        finetune = w2v_finetune
    elif args.pretrain_model.lower() == W2V:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        finetune = w2v_finetune
    elif args.pretrain_model.lower() == FAST_TEXT_KV:
        model = KeyedVectors.load_word2vec_format(model_path)
        finetune = fasttext_finetune_ftkv
    elif args.pretrain_model.lower() == FAST_TEXT:
        model = fasttext.load_facebook_model(model_path)
        model.min_count = 1
        finetune = fasttext_finetune
    else:
        raise ValueError(f"{args.pretrain_model} doesn't exist")

    analyzer = Analyzer(get_lucene_analyzer())
    tokenizer = analyzer.analyze
    tokenized_sentences = []
    if input_doc_path.is_dir():
        doc_files = sorted(input_doc_path.glob("*.json"))
        for doc_file in doc_files:
            print(doc_file)
            with doc_file.open() as f:
                for line in f:
                    sentence = json.loads(line)["contents"]
                    t_sentence = tokenizer(sentence)
                    tokenized_sentences.append(t_sentence)

    else:
        with input_doc_path.open() as f:
            for line in f:
                sentence = json.loads(line)["contents"]
                t_sentence = tokenizer(sentence)
                tokenized_sentences.append(t_sentence)

    finetuned_model = finetune(tokenized_sentences, model, model_path)
    if args.pretrain_model.lower() not in {FAST_TEXT, FAST_TEXT_KV}:
        finetuned_model.wv.save_word2vec_format(output_model_path)
    else:
        finetuned_model.save(output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest="input")
    parser.add_argument("-o", dest="output")
    parser.add_argument("-m", dest="model_path")
    parser.add_argument("-p", dest="pretrain_model", default="")

    args = parser.parse_args()

    main(args)
