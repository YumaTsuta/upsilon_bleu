import argparse
import torch
import pickle
from tqdm import tqdm, trange
import os
from itertools import zip_longest
import numpy as np
import logging
from logging import getLogger, StreamHandler, DEBUG
import wordPreprocess as preprocess  # nopep8
from gensim.summarization.bm25 import BM25


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(DEBUG)

ge_logger = preprocess.logger
ge_logger.addHandler(handler)
ge_logger.setLevel(DEBUG)

parser = argparse.ArgumentParser(
    parents=[preprocess.parserSettings(add_help=False)])
# test_example means the dialogues for test (see ../../data_arrange/test)
parser.add_argument("test_example", type=str,
                    help="the directory which includes test examples")
# cirpus means the dialogues for retrieval (see ../../data_arrange/corpus/tokenized)
parser.add_argument("corpus",
                    type=str, help="the directory which includes dialogues. the sentences have to be tokenized.")
# cirpus_original means the dialogues for retrieval (see ../../data_arrange/corpus)
parser.add_argument("-corpus_original",
                    type=str, help="the directory which includes dialogues. the sentences is retrieved from this directory when this option is activated.")
parser.add_argument("-wp", "--write_path",
                    type=str, help="the directory where retrieved dialogues are stored")
parser.add_argument("-method", "--similarity_method", choices=[
                    "bm25", "embedding"], default="embedding", help="function to compute similarity")
parser.add_argument("-target", "--similarity_target", choices=[
                    "utr", "dialog"], default="utr", help="target to compute similarity")
parser.add_argument("-iter_size", "--iter_size", type=int, default=100)
parser.add_argument("-n_search", "--number_search", type=int, default=0,
                    help="the number of sentences to read from corpus (for debug)")
# extracted responses are rated by nn-rater (see ../nn-rater)
parser.add_argument("-n_cand", "--max_candidate", type=int,
                    default=15, help="the number of candiate dialogues extracted from corpus")
# parrot response
parser.add_argument("-add_utr", action="store_true", default=False,
                    help="add original utterance to the candidate responses")
parser.add_argument("-add_rpl", action="store_true", default=False,
                    help="add original reference to the candidate responses")

args = parser.parse_args()

if args.write_path == None:
    print(
        f"No input path for saving\n save on the directory: {args.test_example}/pseudos")


def load(data_file):
    with open(data_file, "r") as f:
        return [line.strip("\r\n") for line in f.readlines()]


def getEmbAndNorm(args, input_):
    embeds = preprocess.main(args, input_,).astype(np.float32)
    try:
        norm = np.linalg.norm(embeds, axis=1)
    except:
        print(embeds)
        assert False
    norm = np.where(norm == 0, np.ones_like(norm), norm)
    return embeds, norm


def file_multi_enumerator(file1, file2=None, iter_size=1):
    file1_list_tmp = []
    file2_list_tmp = []
    line_list_tmp = []
    for i, line in enumerate(file1):
        line_list_tmp.append(i)
        file1_list_tmp.append(line)
        if file2 != None:
            file2_list_tmp.append(file2.readline())
        if len(line_list_tmp) < iter_size:
            continue
        yield line_list_tmp, file1_list_tmp, file2_list_tmp
        file1_list_tmp = []
        file2_list_tmp = []
        line_list_tmp = []
    if len(line_list_tmp) != 0:
        yield line_list_tmp, file1_list_tmp,


logger.info("loading inputs")
input_utr = load(os.path.join(args.test_example, "ref.utr"))
if args.similarity_target == "dialog" or args.add_rpl:
    input_rpl = load(os.path.join(args.test_example, "ref.rep"))
    assert len(input_utr) == len(input_rpl),\
        "utterance and reply is not same number"

if args.similarity_method == "bm25":
    utr_tokenized = preprocess.tokenize(args, input_utr)
    utr_BM25 = BM25(utr_tokenized)
    if args.similarity_target == "dialog":
        rpl_tokenized = preprocess.tokenize(args, input_rpl)
        rpl_BM25 = BM25(rpl_tokenized)
    similarity_all = np.ones((len(utr_tokenized), 1), dtype="float32")*-1

elif args.similarity_method == "embedding":
    utr_embeds, utr_norm = getEmbAndNorm(args, input_utr)
    if args.similarity_target == "dialog":
        rpl_embeds, rpl_norm = getEmbAndNorm(args, input_rpl)
    similarity_all = np.ones((utr_norm.shape[0], 1), dtype="float32")*-1

else:
    assert False, "invalid input to args.similarity_method"

alpha = 0.5
epsilon = 0.1

id_array = np.ones_like(similarity_all, dtype="int64")*-1


token_meth = args.tokenize_method
args.tokenize_method = None
ge_logger.setLevel(logging.WARNING)

corpus_utr = open(os.path.join(args.corpus, "tokenized_utterance"))
corpus_rpl = None
if args.similarity_target == "dialog":
    corpus_rpl = open(os.path.join(args.corpus, "tokenized_reply"))


for i_list, utr_list, rpl_list in file_multi_enumerator(corpus_utr, corpus_rpl, args.iter_size):
    utr_list = [sent.split(" ") for sent in utr_list]
    rpl_list = [sent.split(" ") for sent in rpl_list]
    if args.similarity_method == "embedding":
        utr_tmp_embeds, utr_tmp_norm = getEmbAndNorm(args, utr_list)
        similarity = utr_embeds.dot(
            utr_tmp_embeds.T)/utr_norm[:, None]/utr_tmp_norm[None, :]
        if args.similarity_target == "dialog":
            rpl_tmp_embeds, rpl_tmp_norm = getEmbAndNorm(args, rpl_list)
            rpl_similarity = rpl_embeds.dot(
                rpl_tmp_embeds.T)/rpl_norm[:, None]/rpl_tmp_norm[None, :]
            similarity = similarity * \
                (rpl_similarity*alpha+(1-alpha) *
                 epsilon*np.ones_like(rpl_similarity))

    elif args.similarity_method == "bm25":
        similarity_tmp = []
        for utr_tmp, rpl_tmp in zip_longest(utr_list, rpl_list):
            sim = np.array(utr_BM25.get_scores(utr_tmp), dtype="float32")
            if args.similarity_target == "dialog":
                rpl_sim = np.array(rpl_BM25.get_scores(
                    rpl_tmp), dtype="float32")
                sim *= rpl_sim*alpha+(1-alpha)*epsilon
            similarity_tmp.append(sim)
        similarity = np.array(similarity_tmp).T

    similarity_all = np.hstack([similarity_all, similarity])
    new_ids = np.array([i_list] * len(similarity_all), dtype="int64")
    id_array = np.hstack([id_array, new_ids])
    sim_argsort = np.argsort(similarity_all, axis=-1)[:, ::-1]
    dim_idx = list(np.ix_(*[np.arange(i) for i in similarity_all.shape]))
    dim_idx[1] = sim_argsort[:, :args.max_candidate]
    similarity_all = similarity_all[tuple(dim_idx)]
    id_array = id_array[tuple(dim_idx)]
    if args.number_search and i+args.batch_size >= args.number_search:
        break

logger.info("calculate similarity is finished")

used_line_list = set(id_array.flatten())
line2utr = dict()
line2rpl = dict()
if args.corpus_original:
    logger.info("loading original corpus")
    corpus_utr = load(os.path.join(args.corpus_original, "utterance"))
    corpus_rpl = load(os.path.join(args.corpus_original, "reply"))
else:
    corpus_utr = load(os.path.join(args.corpus, "tokenized_utterance"))
    corpus_rpl = load(os.path.join(args.corpus, "tokenized_reply"))
for i, (utr, rpl) in enumerate(zip(corpus_utr, corpus_rpl)):
    if i in used_line_list:
        line2utr[i] = utr
        line2rpl[i] = rpl

if args.write_path == None:
    args.write_path = os.path.join(
        args.test_example, "pseudos", args.similarity_method+"_"+args.similarity_target)
    if args.add_utr:
        args.write_path += "_add_utr"

    if args.add_rpl:
        args.write_path += "_add_rpl"

if not os.path.exists(os.path.dirname(args.write_path)):
    os.mkdir(os.path.dirname(args.write_path))
if not os.path.exists(args.write_path):
    os.mkdir(args.write_path)

assert os.path.isdir(args.write_path), f"{args.write_path} is not directory"

logger.info(f"writeing to {args.write_path}")
for i, line_ids in enumerate(tqdm(id_array, ncols=0)):
    utr = open(os.path.join(args.write_path,
                            "candidate_"+str(i)+"_utr.txt"), "w")
    rpl = open(os.path.join(args.write_path,
                            "candidate_"+str(i)+"_rpl.txt"), "w")
    if args.add_utr:
        utr.write(input_utr[i]+"\n")
        rpl.write(input_utr[i]+"\n")

    if args.add_utr:
        utr.write(input_utr[i]+"\n")
        rpl.write(input_rpl[i]+"\n")

    for line_id in line_ids:
        utr.write(line2utr[line_id]+"\n")
        rpl.write(line2rpl[line_id]+"\n")
