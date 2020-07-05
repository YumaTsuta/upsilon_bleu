import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import data_tool
from model import oneBatch, UnreferencedModel
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
                    type=int, default=10, help="the batch size on test. multiplied by the size on train")
parser.add_argument("-save", "--save_path", type=str)
parser.add_argument("-read", "--read_dir", type=str,
                    help="the directory of data to read which includes *_rpl.txt and *_utr.txt")
parser.add_argument("-log_step", "--log_step", type=int,
                    default=100)
args = parser.parse_args()
args.read_dir = os.path.abspath(args.read_dir)


if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
if os.path.exists(os.path.join(args.save_path, "logs")):
    shutil.rmtree(os.path.join(args.save_path, "logs"))

writer = SummaryWriter(log_dir=os.path.join(args.save_path, "logs"))


def batchProcess(data, model, optimizer, mode):
    query_lens = torch.LongTensor([len(sents)
                                   for sents in data[:, 0]]).to(device)
    query = torch.from_numpy(data_tool.padding(data[:, 0])).to(device)
    pos_rpl_lens = torch.LongTensor([len(sents)
                                     for sents in data[:, 1]]).to(device)
    pos_rpl = torch.from_numpy(data_tool.padding(data[:, 1])).to(device)
    neg_rpl_lens = torch.LongTensor([len(sents)
                                     for sents in data[:, 2]]).to(device)
    neg_rpl = torch.from_numpy(data_tool.padding(data[:, 2])).to(device)
    loss = oneBatch(query, pos_rpl, neg_rpl, query_lens, pos_rpl_lens,
                    neg_rpl_lens, model, optimizer, args.margin, mode)
    return loss


def train(dataset, model, optimizer, np_rdm_gen):
    model.train()
    loss_all = 0
    dataset = np.concatenate(
        [dataset, np_rdm_gen.permutation(dataset[:, 1])[:, None]], 1)
    data_iterator = data_tool.iterator(
        dataset, args.max_batch_size, np_rdm_gen=np_rdm_gen)
    with tqdm(data_iterator, total=len(dataset)//args.max_batch_size, ncols=0) as t:
        for i, data in enumerate(t):
            loss = batchProcess(data, model, optimizer, "train")
            loss_all += loss.item()
            t.set_postfix({"ave loss": loss_all/(i+1)})

    return loss


def test(dataset, model, optimizer, np_rdm_gen):
    model.eval()
    loss_all = 0
    dataset = np.concatenate(
        [dataset, np_rdm_gen.permutation(dataset[:, 1])[:, None]], 1)
    data_iterator = data_tool.iterator(
        dataset, args.max_batch_size*args.multiply_batch_size, np_rdm_gen=np_rdm_gen)
    with torch.no_grad():
        for i, data in enumerate(data_iterator):
            loss = batchProcess(data, model, optimizer, "test")
            loss_all += loss.item()

    return loss


def torch_save(path, n_iter, model, optimizer, loss):
    torch.save({
        'iteration': n_iter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }, path)


dataset = data_tool.load(args.read_dir)
source = dataset["train"]
valid = dataset["valid"]
#test_data = dataset["test"]
dic = dataset["dict"]
dev_loss_best = 10**5
best_iter = 0

model = UnreferencedModel(len(dic), args.gru_hidden).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(
    args.adam_b1, args.adam_b2))

for i_iter in trange(args.max_epoch, ncols=0):
    tqdm.write(f"iter: {i_iter}")
    loss = train(source, model, optimizer, np_rdm_gen)
    tqdm.write(f"TRAIN loss: {loss/len(source)}")
    writer.add_scalar("loss/train", loss/len(source), i_iter)
    loss = test(valid, model, optimizer, np_rdm_gen)
    writer.add_scalar("loss/test", loss/len(valid), i_iter)

    for f in os.listdir(args.save_path):
        if "_last" in f:
            os.remove(os.path.join(args.save_path, f))
        torch_save(os.path.join(args.save_path, f'model_{i_iter}_last.tar'),
                   i_iter, model, optimizer, loss)

    if dev_loss_best > loss:
        for f in os.listdir(args.save_path):
            if "_best" in f:
                os.remove(os.path.join(args.save_path, f))
        torch_save(os.path.join(args.save_path, f'noclf_{i_iter}_best.tar'),
                   i_iter, model, optimizer, loss)
        dev_loss_best = loss
        best_iter = i_iter
    print_sent = f"TEST loss: {loss/len(valid)}"
    print_sent += f", Best loss: {dev_loss_best/len(valid)}"
    print_sent += f" (iter:{best_iter})"
    tqdm.write(print_sent)

writer.close()
