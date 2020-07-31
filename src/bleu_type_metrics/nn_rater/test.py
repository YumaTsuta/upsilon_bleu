import torch
from torch import optim
import data_tool
import model
from torch import nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import pickle
import os
import logging
from logging import getLogger, StreamHandler, DEBUG
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import wordPreprocess  # nopep8

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(DEBUG)

ge_logger = wordPreprocess.logger
ge_logger.addHandler(handler)
ge_logger.setLevel(DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

parser = argparse.ArgumentParser(
    parents=[wordPreprocess.parserToken(add_help=False)])
parser.add_argument("reference", type=str,
                    help="the directory of test examples")
parser.add_argument("hypothesis", type=str,
                    help="the directory of collected responses")
parser.add_argument("model", type=str,
                    help="the directory of saved model file")
parser.add_argument("-save", "--save_dir", type=str,
                    help="the directory where nn-rater save the prediction scores ")
args = parser.parse_args()

if not args.save_dir:
    args.save_dir = args.hypothesis

checkpoint = torch.load(args.model)
gru_sd = checkpoint['gru']
ffnn_sd = checkpoint['ffnn']
gru_optimizer_sd = checkpoint['gru_opt']
ffnn_optimizer_sd = checkpoint['ffnn_opt']
embedding_sd = checkpoint['embedding']
param = checkpoint["param"]


seed_id = 0
np.random.seed(seed_id)
torch.manual_seed(seed_id)


gru_hidden_size = param["gru_hidden_size"]
gru_n_layers = param["gru_n_layers"]
ffnn_hidden_size = param["ffnn_hidden_size"]
ffnn_output_size = 2
ffnn_n_layers = param["ffnn_n_layers"]
lr = param["lr"]
file_path = param["read"]
feature_list = [[0, 1, 3], [2, 3, 1]]
ffnn_init_num = len(feature_list[0])
batch_size = param["batch_size"]


def doc2ID(file_path, word2id):
    with open(file_path, "r") as f:
        data = [sent.strip("\r\n") for sent in f.readlines()]
    data = wordPreprocess.tokenize(args, data)
    return [[word2id.get(word, word2id["<unk>"]) for word in sent] for sent in data]


def test():
    dataset = data_tool.load(file_path)
    word2id = dataset["dict"]

    embedding = nn.Embedding(len(word2id), gru_hidden_size)
    embedding.load_state_dict(embedding_sd)
    gru = model.BiGRU(gru_hidden_size, embedding,
                      gru_n_layers)
    gru.load_state_dict(gru_sd)
    ffnn = model.MLP(
        gru_hidden_size*ffnn_init_num*2, ffnn_hidden_size, 2, ffnn_n_layers)
    ffnn.load_state_dict(ffnn_sd)

    gru = gru.to(device)
    ffnn = ffnn.to(device)

    gru.eval()
    ffnn.eval()

    ref_src = doc2ID(os.path.join(args.reference, "ref.utr"), word2id)
    ref_tgt = doc2ID(os.path.join(args.reference, "ref.rep"), word2id)

    for i in trange(len(ref_src), ncols=0):
        hyp_src = doc2ID(os.path.join(args.hypothesis,
                                      "candidate_"+str(i)+"_utr.txt"), word2id)
        hyp_tgt = doc2ID(os.path.join(args.hypothesis,
                                      "candidate_"+str(i)+"_rpl.txt"), word2id)
        input_var = []
        input_var.append([ref_src[i] for _ in hyp_src])
        input_var.append([ref_tgt[i] for _ in hyp_src])
        input_var.append(hyp_src)
        input_var.append(hyp_tgt)
        input_var = [inputs for inputs in zip(*input_var)]

        dummy_label = [1]*len(input_var)
        pres = []
        probs = []
        data_iterator = data_tool.iterator(
            input_var, dummy_label, batch_size, random=False)
        for data, length, _ in data_iterator:
            with torch.no_grad():
                data = data_tool.padding(data)
                predictions, probability = modelAll(
                    data, length, gru, ffnn, embedding)
                pres += [int(label) for label in predictions]
                probs += [float(prob) for prob in probability[0].cpu()]

        with open(os.path.join(args.save_dir, "candidate_"+str(i)+"_predictions.txt"), "w") as f:
            for label, prob in zip(pres, probs):
                f.write(str((label-0.5)*2*(prob-0.5)*2)+"\r\n")


def modelAll(input_tensor, lengths, gru, ffnn, embedding):

    batch_size = len(lengths)//4
    argsort = np.argsort(lengths)[::-1]
    input_tensor = torch.from_numpy(input_tensor[:, argsort]).to(device)
    lengths = torch.from_numpy(lengths[argsort]).to(device)

    _, gru_hidden = gru(input_tensor, lengths)
    gru_hidden = torch.cat(
        [gru_hidden[gru_n_layers-1], gru_hidden[gru_n_layers]], dim=1)

    reverse = np.argsort(argsort)
    features = []
    for i in range(batch_size):
        for f_idxs in feature_list:
            feature_ = [gru_hidden[reverse[idx+4*i]] for idx in f_idxs]
            features.append(torch.cat(feature_))

    features = torch.cat(features).view(batch_size*2, -1)
    features = ffnn(features)

    prob = torch.exp(features).view(batch_size, -1, 2)[:, 0]

    predictions = torch.argmax(prob, dim=1)
    probabilities = torch.max(prob, dim=1)

    return predictions, probabilities


if __name__ == "__main__":
    test()
