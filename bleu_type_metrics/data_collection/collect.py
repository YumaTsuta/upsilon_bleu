import argparse
import torch
import pickle
from tqdm import tqdm, trange
import os
from itertools import zip_longest
import numpy as np
import logging
from logging import getLogger, StreamHandler, DEBUG
import GetEmbeddings  # nopep8

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(DEBUG)

ge_logger = GetEmbeddings.logger
ge_logger.addHandler(handler)
ge_logger.setLevel(DEBUG)

parser = argparse.ArgumentParser(
    parents=[GetEmbeddings.parserSettings(add_help=False)])
parser.add_argument("-ref_utr", "--reference_utr",
                    type=str, help="/path/to/read(directory)")
parser.add_argument("-ref_rpl", "--reference_rpl",
                    type=str, help="/path/to/read(directory)")
parser.add_argument("-corpus",
                    type=str, help="/path/to/read(directory). the sentences have to be tokenized")
parser.add_argument("-corpus_original",
                    type=str, default="", help="/path/to/read(directory).  the sentences is not tokenized")
parser.add_argument("-wp", "--write_path",
                    type=str, help="/path/to/write(directory)")
parser.add_argument("-iter_size", "--iter_size", type=int, default=100)
parser.add_argument("-n_search", "--number_search", type=int,
                    default=0, help="the number of sentences to read (for debug)")
parser.add_argument("-n_cand", "--max_candidate", type=int,
                    default=100, help="the number of extracting condidate responses")
parser.add_argument("-add_utr", action="store_true", default=False,
                    help="add original utterance to the candidate responses")
parser.add_argument("-add_rpl", action="store_true", default=False,
                    help="add original reference to the candidate responses")
args = parser.parse_args()
assert os.path.exists(args.write_path)


def load(data_file):
    with open(data_file, "r") as f:
        return [line.strip("\r\n") for line in f.readlines()]


def loadDialogues(dir_path):
    dialogues = set()
    utterance_dict = dict()
    filelist = [file_name for file_name in os.listdir(
        dir_path) if ".pickle" in file_name]
    for read_file in tqdm(filelist, ncols=0):
        with open(os.path.join(dir_path, read_file), "rb") as f:
            dl, ud = pickle.load(f)
        dialogues |= dl
        utterance_dict.update(ud)
    return dialogues, utterance_dict


def getEmbAndNorm(args, input_, corpus):
    embeds = GetEmbeddings.main(args, input_, corpus).astype(np.float32)
    try:
        norm = np.linalg.norm(embeds, axis=1)
    except:
        print(embeds)
        assert False
    norm = np.where(norm == 0, np.ones_like(norm), norm)
    return embeds, norm


logger.info("loading inputs")
input_utr = load(args.reference_utr)
input_rpl = load(args.reference_rpl)
dialogues, id2content = loadDialogues(args.corpus)
utr_ids = list(set(d[0] for d in tqdm(dialogues, ncols=0, leave=False)))
utr_corpus = [id2content[utr_id][3].split(
    " ") for utr_id in tqdm(utr_ids, ncols=0, leave=False)]

input_embeds, input_norm = getEmbAndNorm(args, input_utr, utr_corpus)

token_meth = args.tokenize_method
args.tokenize_method = None
ge_logger.setLevel(logging.WARNING)


similarity_all = np.ones((input_norm.shape[0], 1), dtype="float32")*-1
id_array = np.ones_like(similarity_all, dtype="int64")*-1
for i in trange(0, len(utr_ids), args.iter_size, ncols=0):
    try:
        hyp_embeds, hyp_norm = getEmbAndNorm(
            args, utr_corpus[i:i+args.batch_size], utr_corpus)
    except:
        print(utr_corpus[i:i+args.batch_size])
        assert False

    similarity = input_embeds.dot(
        hyp_embeds.T)/input_norm[:, None]/hyp_norm[None, :]
    similarity_all = np.hstack([similarity_all, similarity])
    new_ids = np.array([utr_ids[i:i+args.batch_size]] *
                       input_embeds.shape[0], dtype="int64")
    id_array = np.hstack([id_array, new_ids])
    sim_argsort = np.argsort(similarity_all, axis=-1)[:, ::-1]
    dim_idx = list(np.ix_(*[np.arange(i) for i in similarity_all.shape]))
    dim_idx[1] = sim_argsort[:, :args.max_candidate]
    similarity_all = similarity_all[tuple(dim_idx)]
    id_array = id_array[tuple(dim_idx)]
    if args.number_search != 0 and i+args.batch_size >= args.number_search:
        break

if args.corpus_original != "":
    logger.info("loading original corpus")
    dialogues, id2content = loadDialogues(args.corpus_original)

logger.info("calculate similarity is finished")
utr2rpl = {d[0]: d[1] for d in dialogues}
args.write_path = os.path.join(args.write_path, "pseudos")
if args.add_utr:
    args.write_path += "_add_utr"

if args.add_rpl:
    args.write_path += "_add_rpl"

if not os.path.exists(args.write_path):
    os.mkdir(args.write_path)
filename = "_".join(
    [token_meth, args.vectorize_method, args.aggregation_method])
args.write_path = os.path.join(args.write_path, filename)
if not os.path.exists(args.write_path):
    os.mkdir(args.write_path)
logger.info(f"writeing to {args.write_path}")
for i, (utr_ids, sim_scores) in enumerate(zip(tqdm(id_array, ncols=0), similarity_all)):
    utr = open(os.path.join(args.write_path,
                            "candidate_"+str(i)+"_utr.txt"), "w")
    rpl = open(os.path.join(args.write_path,
                            "candidate_"+str(i)+"_rpl.txt"), "w")
    sim = open(os.path.join(args.write_path,
                            "candidate_"+str(i)+"_sim.txt"), "w")
    if args.add_utr:
        utr.write(input_utr[i]+"\r\n")
        rpl.write(input_utr[i]+"\r\n")
        sim.write(str(1)+"\r\n")

    if args.add_utr:
        utr.write(input_utr[i]+"\r\n")
        rpl.write(input_rpl[i]+"\r\n")
        sim.write(str(1)+"\r\n")

    for utr_id, sim_score in zip(utr_ids, sim_scores):
        utr.write(id2content[int(utr_id)][3]+"\r\n")
        rpl.write(id2content[utr2rpl[int(utr_id)]][3]+"\r\n")
        sim.write(str(float(sim_score))+"\r\n")
