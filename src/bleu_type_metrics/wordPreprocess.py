import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import logging
from logging import getLogger, DEBUG, NullHandler
from itertools import zip_longest

logger = getLogger(__name__)
getLogger(__package__).addHandler(NullHandler())
logger.setLevel(DEBUG)


def parserToken(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("-c_size", "--conversion_size", type=int, default=100,
                        help="the size of sentence to convert from its word to the embedding")
    parser.add_argument("-tokenize", "--tokenize_method",
                        type=str, choices=["sentencepiece", "tokenized", "mecab"], help="select tokenization methods. tokenized means sentences are tokenized by white space.")

    spm = parser.add_argument_group("spm", "sentencepiece option")
    spm.add_argument("-train", "--train_file", type=str,
                     help="path to the file which is trained model by SentencePiece (*.model on default)")
    return parser


def parserSettings(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help, parents=[
                                     parserToken(add_help=False)])
    parser.add_argument("-vector", "--vector_file", type=str,
                        help="the vector file made by Glove")
    parser.add_argument("-no_vector", default=False,
                        action="store_true", help="activate when tokenizing only")

    return parser


def MecabTokenizer(sents):
    if not "mecab_tokenizer" in globals():
        import MeCab
        global mecab_tokenizer
        mecab_tokenizer = MeCab.Tagger("-Owakati")
    return [mecab_tokenizer.parse(sent).split() for sent in tqdm(sents, ncols=0, leave=False)]


def sentencepiece(args, sents, logger=logger):
    import sentencepiece as spm
    assert args.train_file, "input train files"
    sp_model_path = args.train_file
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sp_model_path)
    return [sp_model.EncodeAsPieces(sent) for sent in tqdm(sents, ncols=0, leave=False)]


def tokenize(args, sents, logger=logger):
    if not args.tokenize_method:
        pass
    elif args.tokenize_method == "tokenized":
        sents = [sent.split(" ") for sent in sents]
    elif args.tokenize_method == "sentencepiece":
        sents = sentencepiece(args, sents, logger)
    elif args.tokenize_method == "mecab":
        sents = MecabTokenizer(sents)
    else:
        assert False
    return sents


def sentIterByBatch(sents, max_len, conversion_size, pad=0):
    for i in range(0, len(sents), conversion_size):
        yield np.array([ids[:-1] for ids in zip_longest(*(sents[i:i+conversion_size]+[[pad]*max_len]), fillvalue=pad)]).T


def gloveEmbedding(args, sents, leave=False, logger=logger):
    if not "glove_W" in globals():
        global glove_W, glove_word2id
        logger.info(f"loading vocab & embedding from {args.vector_file}")
        with open(args.vector_file, "r") as f:
            glove_vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                glove_vectors[vals[0]] = [float(x) for x in vals[1:]]
        glove_words = list(glove_vectors.keys())
        glove_word2id = {w: idx for idx, w in enumerate(glove_words)}
        glove_id2word = {idx: w for idx, w in enumerate(glove_words)}
        glove_W = np.zeros(
            (len(glove_words), len(glove_vectors[glove_id2word[0]])))
        for word, v in glove_vectors.items():
            glove_W[glove_word2id[word], :] = v
        d = (np.sum(glove_W ** 2, 1) ** (0.5))
        glove_W = (glove_W.T / d).T
    pad_id = -1
    max_len = max([len(sent) for sent in sents])
    sents = [[glove_word2id.get(word, glove_word2id["unk"])
              for word in sent] for sent in sents]
    return [glove_W[sent_batch]*(sent_batch != pad_id)[:, :, None]
            for sent_batch in sentIterByBatch(sents, max_len, args.conversion_size, pad_id)]


def wordEmbed(args, sents, logger=logger):
    return gloveEmbedding(
        args, sents, logger=logger)


def sentEmbed(args, embeds, sents):
    # averaged embedding
    count = np.count_nonzero(embeds, axis=1)
    return np.sum(embeds, axis=1)/np.where(count == 0, 1, count)


def main(args, sents, logger=logger):
    logger.info(f"tokenize by {args.tokenize_method} method")
    sents = tokenize(args, sents, logger)
    if args.no_vector:
        return sents
    word_embeds = wordEmbed(args, sents, logger)
    word_embeds = np.concatenate(word_embeds)
    sent_embeds = sentEmbed(args, word_embeds, sents)
    return sent_embeds
