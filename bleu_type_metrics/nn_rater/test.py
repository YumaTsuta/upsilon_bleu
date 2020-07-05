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
import GetEmbeddings  # nopep8

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(DEBUG)

ge_logger = GetEmbeddings.logger
ge_logger.addHandler(handler)
ge_logger.setLevel(DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

parser = argparse.ArgumentParser(
    parents=[GetEmbeddings.parserToken(add_help=False)])
parser.add_argument("-method", choices=["max", "avg", "one", "another"])
parser.add_argument("-load_ref_utr", "--load_reference_utr", type=str,
                    default="path/to/reference")
parser.add_argument("-load_ref_rpl", "--load_reference_rpl", type=str,
                    default="path/to/reference")
parser.add_argument("-load_hyp", "--load_hypothesis", type=str,
                    default="path/to/hypothesis")
parser.add_argument("-load_m", "--load_model", type=str)
parser.add_argument("-save", "--save_dir", type=str,
                    default="")
args = parser.parse_args()

if args.save_dir == "":
    args.save_dir = args.load_hypothesis

checkpoint = torch.load(args.load_model)
# If loading a model trained on GPU to CPU
# checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
compressor_sd = checkpoint['com']
encoder_optimizer_sd = checkpoint['en_opt']
compressor_optimizer_sd = checkpoint['com_opt']
embedding_sd = checkpoint['embedding']
param = checkpoint["param"]


seed_id = 0
np.random.seed(seed_id)
torch.manual_seed(seed_id)


enc_hidden_size = param["enc_hidden_size"]
enc_n_layers = param["enc_n_layers"]
cpr_hidden_size = param["cpr_hidden_size"]
cpr_output_size = param["cpr_output_size"]
cpr_n_layers = param["cpr_n_layers"]
dropout = param["dropout"]
enc_lr = param["enc_lr"]
cpr_lr = param["cpr_lr"]
file_path = param["read"]
feature_list = param["feature_list"]
cpr_init_num = len(feature_list[0])
clf_init_num = len(feature_list)
batch_size = param["batch_size"]


def doc2ID(file_path, word2id):
    with open(file_path, "r") as f:
        data = [sent.strip("\r\n") for sent in f.readlines()]
    data = GetEmbeddings.tokenize(args, data)
    return [[word2id.get(word, word2id["<unk>"]) for word in sent] for sent in data]


def test():
    dataset = data_tool.load(file_path)
    word2id = dataset["dict"]
    #source = dataset["train"]
    #dev_source = dataset["dev"]
    #test = dataset["test"]

    embedding = nn.Embedding(len(word2id), enc_hidden_size)
    embedding.load_state_dict(embedding_sd)
    encoder = model.Encoder(enc_hidden_size, embedding,
                            enc_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    compressor = model.MLP(
        enc_hidden_size*cpr_init_num*2, cpr_hidden_size, cpr_output_size, cpr_n_layers)
    compressor.load_state_dict(compressor_sd)

    encoder = encoder.to(device)
    compressor = compressor.to(device)

    encoder.eval()
    compressor.eval()

    ref_src = doc2ID(args.load_reference_utr, word2id)
    ref_tgt = doc2ID(args.load_reference_rpl, word2id)

    for i in trange(len(ref_src), ncols=0):
        hyp_src = doc2ID(os.path.join(args.load_hypothesis,
                                      "candidate_"+str(i)+"_utr.txt"), word2id)
        hyp_tgt = doc2ID(os.path.join(args.load_hypothesis,
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
        total = len(dummy_label)//batch_size + \
            (len(dummy_label) % batch_size != 0)
        for data, length, _ in data_iterator:
            with torch.no_grad():
                data = data_tool.padding(data)
                predictions, probability = modelAll(
                    data, length, encoder, compressor, embedding)
                pres += [int(label) for label in predictions]
                probs += [float(prob) for prob in probability[0].cpu()]

        with open(os.path.join(args.save_dir, "candidate_"+str(i)+"_"+args.method+"_predictions.txt"), "w") as f:
            for label, prob in zip(pres, probs):
                f.write(str((label-0.5)*2*prob)+"\r\n")


def modelAll(input_tensor, lengths, encoder, compressor, embedding):

    batch_size = len(lengths)//4
    argsort = np.argsort(lengths)[::-1]
    input_tensor = torch.from_numpy(input_tensor[:, argsort]).to(device)
    lengths = torch.from_numpy(lengths[argsort]).to(device)

    _, encoder_hidden = encoder(input_tensor, lengths)
    encoder_hidden = torch.cat(
        [encoder_hidden[enc_n_layers-1], encoder_hidden[enc_n_layers]], dim=1)

    reverse = np.argsort(argsort)
    features = []
    for i in range(batch_size):
        for f_idxs in feature_list:
            feature_ = [encoder_hidden[reverse[idx+4*i]] for idx in f_idxs]
            features.append(torch.cat(feature_))

    features = torch.cat(features).view(batch_size*clf_init_num, -1)
    features = compressor(features)

    if args.method == "avg":
        prob = torch.sum(torch.exp(features).view(
            batch_size, -1, 2), dim=1)/clf_init_num
    elif args.method == "max":
        prob = torch.max(torch.exp(features).view(batch_size, -1, 2), dim=1)[0]
    elif args.method == "one":
        prob = torch.exp(features).view(batch_size, -1, 2)[:, 0]
    elif args.method == "another":
        prob = torch.exp(features).view(batch_size, -1, 2)[:, 1]

    predictions = torch.argmax(prob, dim=1)
    probabilities = torch.max(prob, dim=1)

    return predictions, probabilities


if __name__ == "__main__":
    test()
