from scipy.stats import kendalltau, spearmanr, pearsonr
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import data_tool
from model import oneBatch_test, UnreferencedModel
import numpy as np
from tqdm import tqdm, trange
import argparse
import shutil

np_rdm_gen = np.random
np_rdm_gen.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="desc")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-b1", "--adam_b1", type=float, default=0.9)
parser.add_argument("-b2", "--adam_b2", type=float, default=0.999)
parser.add_argument("-margin", "--margin", type=float, default=0.075)
parser.add_argument("-gru", "--gru_hidden", type=int, default=500)
parser.add_argument("-epoch", "--max_epoch", type=int, default=1000)
parser.add_argument("-batch", "--max_batch_size", type=int, default=1000)
parser.add_argument("-multi_batch", "--multiply_batch_size",
                    type=int, default=2, help="the batch size on test. multiplied by the size on train")
parser.add_argument("-model", "--load_model", type=str)
parser.add_argument("-reference", "--load_reference", type=str)
parser.add_argument("-read", "--read_dir", type=str,
                    help="the directory of data to read which includes *_rpl.txt and *_utr.txt")
parser.add_argument("-log_step", "--log_step", type=int,
                    default=100)
parser.add_argument("-spm", "--spm_model", type=str,
                    required=True, help="the file created by sentencepiece")

parser.add_argument("-score", "--score", type=str, required=True,
                    help="the annotation file of human scores")

parser.add_argument("-score_aggregation", "--score_aggregation",
                    type=str, choices=["avg", "median"], default="avg")

parser.add_argument("-write", help="the file where score is written")
parser.add_argument(
    "-data", choices=["test", "dev"], default=False, help="for test or validation")


args = parser.parse_args()
args.read_dir = os.path.abspath(args.read_dir)


def batchProcess_test(data, model):
    query_lens = torch.LongTensor([len(sents)
                                   for sents in data[:, 0]]).to(device)
    query = torch.from_numpy(data_tool.padding(data[:, 0])).to(device)
    pos_rpl_lens = torch.LongTensor([len(sents)
                                     for sents in data[:, 1]]).to(device)
    pos_rpl = torch.from_numpy(data_tool.padding(data[:, 1])).to(device)
    score = oneBatch_test(query, pos_rpl, query_lens, pos_rpl_lens, model)
    return score


def sentencepiece(sents):
    import sentencepiece as spm
    assert args.spm_model, "input train files"
    sp_model_path = args.spm_model
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sp_model_path)
    return [sp_model.EncodeAsPieces(sent) for sent in tqdm(sents, ncols=0, leave=False)]


def doc2ID(file_path, word2id):
    with open(file_path, "r") as f:
        data = [sent.strip("\r\n") for sent in f.readlines()]
    data = sentencepiece(data)
    return [[word2id.get(word, word2id["<unk>"]) for word in sent] for sent in data]


def test(dataset, model):
    model.eval()
    data_iterator = data_tool.iterator(
        dataset, args.max_batch_size*args.multiply_batch_size)
    with torch.no_grad():
        for i, data in enumerate(data_iterator):
            scores = batchProcess_test(data, model)

    return scores


def torch_save(path, n_iter, model, optimizer, loss):
    torch.save({
        'iteration': n_iter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }, path)


dataset = data_tool.load(args.read_dir)
#source = dataset["train"]
#valid = dataset["valid"]
#test_data = dataset["test"]
dic = dataset["dict"]

checkpoint = torch.load(args.load_model)
model_param = checkpoint["model"]


model = UnreferencedModel(len(dic), args.gru_hidden)
model.load_state_dict(model_param)
moodel = model.to(device)

ref_src = doc2ID(os.path.join(
    args.load_reference, "hyp_untoken.txt"), dic)
ref_tgt = doc2ID(os.path.join(
    args.load_reference, "src_untoken.txt"), dic)
test_data = np.array([inputs for inputs in zip(*[ref_src, ref_tgt])])
if args.data == "dev":
    test_data = test_data[:len(test_data)//2]
elif args.data == "test":
    test_data = test_data[len(test_data)//2:]

score_all = []
for i_iter in trange(args.max_epoch, ncols=0):
    tqdm.write(f"iter: {i_iter}")
    scores = test(test_data, model)
score_all += [float(s) for s in scores]

with open(args.score, mode="r") as f:
    annotation_scores = \
        [[int(score) for score in line.strip().split(",")]
            for line in f.readlines()]
    annotation_scores = np.array(annotation_scores)

if args.score_aggregation == "avg":
    score = np.average(annotation_scores, axis=1)
elif args.score_aggregation == "median":
    score = np.median(annotation_scores, axis=1)
if args.data == "dev":
    score = score[:len(score)//2]
elif args.data == "test":
    score = score[len(score)//2:]


print_sent = f"pearson:{pearsonr(score, score_all)}\n\n"
print_sent += f"spearman:{spearmanr(score, score_all)}\n\n"
print_sent += f"kendalltau:{kendalltau(score, score_all)}\n\n"
print(print_sent)

score_all = [str(s) for s in score_all]
with open(args.write, "w") as f:
    f.write(",".join(score_all))
