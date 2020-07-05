import os
import sys
import argparse
from tqdm import tqdm, trange
from collections import defaultdict
from itertools import combinations
import numpy as np
import logging
from logging import getLogger, DEBUG, NullHandler
from itertools import zip_longest
import re
import io

logger = getLogger(__name__)
getLogger(__package__).addHandler(NullHandler())
logger.setLevel(DEBUG)
#logger.propagate = False


def parserToken(add_help=True):
    desc = """
    inputs corpus, sentence
    output tokenized sentence or word embeddings or sentence emnbedding
    """
    parser = argparse.ArgumentParser(description=desc, add_help=add_help)
    parser.add_argument("-leave", "--leave_variable",
                        action="store_true", default=False)
    parser.add_argument("-batch", "--batch_size", type=int, default=100)
    parser.add_argument("-tokenize", "--tokenize_method",
                        type=str, choices=["stanford", "bert", "sentencepiece", "tokenized", "mecab"])
    parser.add_argument("-gpu", "--gpu", action="store_true", default=False)
    parser.add_argument("-lang", "--lang", choices=["ja", "en"], default="en")

    spm = parser.add_argument_group("spm", "sentencepiece option")
    spm.add_argument("-train", "--train_file", type=str)
    spm.add_argument("-pretrained", "--pretrained",
                     action="store_true", default=False)
    spm.add_argument("-vocab_size", "--vocab_size", type=int, default=16000)
    spm.add_argument("-add_token", "--additional_tokens",
                     default=["<user_name>", "<url>"], nargs="*")
    spm.add_argument("-train_size", "--train_sentence_size",
                     type=int, default=0)
    spm.add_argument("-model_type", "--model_type", default="unigram",
                     choices=["word", "unigram", "char", "bpe"])
    spm.add_argument("-coverage", "--character_coverage",
                     type=float, default=0.9995)
    return parser


def parserSettings(add_help=True):
    desc = """
    inputs corpus, sentence, embedding file
    output tokenized sentence or word embeddings or sentence emnbedding
    """
    parser = argparse.ArgumentParser(description=desc, add_help=add_help, parents=[
                                     parserToken(add_help=False)])
    parser.add_argument("-vector", "--vector_file", type=str)
    parser.add_argument("-vectorize", "--vectorize_method",
                        type=str, choices=["bert", "glove"])
    parser.add_argument("-aggregation", "--aggregation_method",
                        type=str, choices=["ave", "extrema", "sif"])

    args = parser.parse_known_args()
    assert args[0].vectorize_method != "glove" or args[0].vector_file, "inputs directory by -vector or --vector_dir to use glove"

    return parser


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


def stanfordTokenizer(sents, batch_size, use_gpu=False, leave=False, logger=logger):
    if not "stanford_tokenizer" in globals():
        import stanfordnlp
        if leave:
            global stanford_tokenizer
        stanford_tokenizer = stanfordnlp.Pipeline(
            processors="tokenize", use_gpu=use_gpu, tokenize_batch_size=batch_size)

    # return [[tok.text for s in stanford_tokenizer(snt).sentences  for tok in s.tokens] if snt!="" else "" for snt in tqdm(sents,ncols=0,leave=False)]
    tokenized_all = []
    pattern = re.compile(r"\s")
    tokenized_new = []
    tqdm_out = TqdmToLogger(logger)
    for i in trange(0, len(sents), batch_size, ncols=0, leave=False, file=tqdm_out):
        sents_tmp = sents[i:i+batch_size]
        tokenized = [[tok.text for tok in sent.tokens]
                     for sent in stanford_tokenizer("\n\n".join(sents_tmp)).sentences]
        tok_idx = 0
        tokenized_new = []
        for sent in sents_tmp:
            if sent == "":
                tokenized_new.append([""])
                continue
            tokenized_new.append(tokenized[tok_idx])
            tok_idx += 1
            while pattern.sub("", sent) != pattern.sub("", "".join(tokenized_new[-1])):
                tokenized_new[-1] += tokenized[tok_idx]
                tok_idx += 1
        tokenized_all += tokenized_new
    return tokenized_all
    """
    with CoreNLPClient(annotators=["tokenize"], timeout=60000, memory="16G", be_quiet=True) as client:
        return [[token.word for token in client.annotate(sent).sentencelessToken] for sent in tqdm(sents, ncols=0)]
    """


def bertTokenizer(sents, lang):
    if lang == "en":
        from transformers import BertTokenizer
        pretrained = "bert-base-uncased"
    elif lang == "ja":
        from transformers import BertJapaneseTokenizer as BertTokenizer
        pretrained = "bert-base-japanese-whole-word-masking"
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained)
    return [bert_tokenizer.tokenize(sent) for sent in tqdm(sents, ncols=0, leave=False)]


def MecabTokenizer(sents, leave=False):
    if not "mecab_tokenizer" in globals():
        import MeCab
        import sys
        if leave:
            global mecab_tokenizer
        mecab_tokenizer = MeCab.Tagger("-Owakati")
    return [mecab_tokenizer.parse(sent).split() for sent in tqdm(sents, ncols=0, leave=False)]


def sentencepiece(args, sents, corpus, logger=logger):
    import sentencepiece as spm
    assert args.train_file, "input train files"
    if not args.pretrained:
        if os.path.isfile(args.train_file):
            args.train_file = os.path.dirname(args.train_file)
        write_tmp = os.path.join(args.train_file, "train_tmp.txt")
        logger.info(f"write corpus in {write_tmp} to train")
        with open(write_tmp, "w") as f:
            for sent in corpus:
                f.write(sent+"\r\n")
        base_dir = os.getcwd()
        os.chdir(args.train_file)
        model_prefix = f"{args.model_type}_vocab{args.vocab_size}"
        additional_tokens = ",".join(args.additional_tokens)
        train_param = [f"--input={os.path.basename(write_tmp)}"]
        train_param.append(f"--model_prefix={model_prefix}")
        train_param.append(f"--vocab_size={args.vocab_size}")
        train_param.append(f"--user_defined_symbols={additional_tokens}")
        train_param.append(f"--model_type={args.model_type}")
        train_param.append(f"--character_coverage={args.character_coverage}")
        if args.train_sentence_size != 0:
            train_param.append(
                f"--input_sentence_size={args.train_sentence_size}")
            train_param.append(f"--shuffle_input_sentence=true")
        logger.info("training...")
        spm.SentencePieceTrainer.Train(" ".join(train_param))
        os.chdir(base_dir)
        sp_model_path = os.path.join(args.train_file, model_prefix+".model")
        args.trainfile = sp_model_path
    else:
        sp_model_path = args.train_file
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sp_model_path)
    return [sp_model.EncodeAsPieces(sent) for sent in tqdm(sents, ncols=0, leave=False)]


def tokenize(args, sents, corpus=[""], logger=logger):
    if not args.tokenize_method:
        pass
    elif args.tokenize_method == "tokenized":
        sents = [sent.split(" ") for sent in sents]
    elif args.tokenize_method == "stanford":
        sents = stanfordTokenizer(
            sents, args.batch_size, args.gpu, args.leave_variable, logger)
    elif args.tokenize_method == "bert":
        sents = bertTokenizer(sents, args.lang)
    elif args.tokenize_method == "sentencepiece":
        sents = sentencepiece(args, sents, corpus, logger)
    elif args.tokenize_method == "mecab":
        sents = MecabTokenizer(sents, leave=args.leave_variable)
    else:
        assert False
    return sents


def sentIterByBatch(sents, max_len, batch_size, pad=0):
    for i in range(0, len(sents), batch_size):
        yield np.array([ids[:-1] for ids in zip_longest(*(sents[i:i+batch_size]+[[pad]*max_len]), fillvalue=pad)]).T


def bertEmbedding(sents, batch_size, lang="en", leave=False, use_gpu=False, logger=logger):
    import torch
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if lang == "en":
        from transformers import BertTokenizer, BertModel
        pretrained = "bert-base-uncased"
    elif lang == "ja":
        from transformers import BertJapaneseTokenizer as BertTokenizer
        from transformers import BertModel
        pretrained = "bert-base-japanese-whole-word-masking"

    if not "bert_model" in globals():
        if leave:
            global bert_model, bert_tokenizer
        pretrained = "bert-base-uncased"
        bert_model = BertModel.from_pretrained(
            pretrained, output_hidden_states=True).to(device)
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained)
    max_len = max([len(sent) for sent in sents])
    sents = [bert_tokenizer.convert_tokens_to_ids(sent) for sent in sents]
    pad_id = 0
    with torch.no_grad():
        return [bert_model(torch.LongTensor(sent_batch).to(device))[0].cpu().detach().numpy()*(sent_batch != pad_id)[:, :, None]
                for sent_batch in sentIterByBatch(sents,  max_len, batch_size, pad_id)]


def gloveEmbedding(args, sents, leave=False, logger=logger):
    if not "glove_W" in globals():
        if args.leave_variable:
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
            for sent_batch in sentIterByBatch(sents, max_len, args.batch_size, pad_id)]


def wordEmbed(args, sents, logger=logger):
    if args.vectorize_method == "bert":
        embeds = bertEmbedding(
            sents, args.batch_size, leave=args.leave_variable, use_gpu=args.gpu, logger=logger)
    elif args.vectorize_method == "glove":
        embeds = gloveEmbedding(
            args, sents, leave=args.leave_variable, logger=logger)
    else:
        assert False
    return embeds


def extremaEmbedding(embeds):
    embeds_abs = np.abs(embeds)
    argmax = np.argmax(embeds_abs, axis=1)
    shp = np.array(embeds.shape)
    dim_idx = list(np.ix_(*[np.arange(i) for i in shp]))
    dim_idx[1] = argmax[:, None]
    return embeds[tuple(dim_idx)].squeeze()


def sifEmbedding(embeds, sents, corpus, hyparam=1e-3, leave=False):
    from SIF.src import params, SIF_embedding
    if not "word_count" in globals():
        if leave:
            global word_count, sif_word2weight
        word_count = defaultdict(int)
        for sent in corpus:
            for token in sent:
                word_count[token] += 1
        N = sum(word_count.values())
        sif_word2weight = {token: hyparam/(hyparam+count/N)
                           for token, count in word_count.items()}

    rmpc = 1
    params = params.params()
    params.rmpc = rmpc

    weights = [[sif_word2weight[word] for word in sent] for sent in sents]
    max_len = max([len(sent) for sent in sents])
    weights = next(sentIterByBatch(weights, max_len, len(sents), 0))

    embeds = np.sum(weights[:, :, None]*embeds, axis=1) / \
        np.count_nonzero(weights, axis=1)[:, None]
    if params.rmpc > 0:
        embeds = SIF_embedding.remove_pc(embeds, params.rmpc)
    return embeds


def sentEmbed(args, embeds, sents, corpus=[""]):
    if args.aggregation_method == "ave":
        count = np.count_nonzero(embeds, axis=1)
        return np.sum(embeds, axis=1)/np.where(count == 0, 1, count)
    elif args.aggregation_method == "extrema":
        return extremaEmbedding(embeds)
    elif args.aggregation_method == "sif":
        return sifEmbedding(embeds, sents, corpus, leave=args.leave_variable)
    else:
        assert False


def main(args, sents, corpus=[""], logger=logger):
    logger.info(f"tokenize by {args.tokenize_method} method")
    sents = tokenize(args, sents, corpus, logger)
    if not args.vectorize_method:
        return sents
    logger.info(f"vectorize by {args.vectorize_method} method")
    word_embeds = wordEmbed(args, sents, logger)
    word_embeds = np.concatenate(word_embeds)
    logger.info(f"aggregate by {args.aggregation_method} method")
    sent_embeds = sentEmbed(args, word_embeds, sents, corpus)
    return sent_embeds
