import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import data_tool
import model
from torch import nn
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import shutil
from collections import defaultdict

parser = argparse.ArgumentParser(description="description")
parser.add_argument("-enc_hidden", type=int, default=512)
parser.add_argument("-enc_layer", type=int, default=1)
parser.add_argument("-cpr_hidden", type=int, default=1024)
parser.add_argument("-cpr_layer", type=int, default=5)
parser.add_argument("-cpr_out", type=int, default=2)
parser.add_argument("-dropout", type=float, default=0.2)
parser.add_argument("-lr", "--learning_rate", type=float, default=1.e-3)
parser.add_argument("-epoch", "--max_epoch", type=int, default=15)
parser.add_argument("-batch", "--batch_size", type=int, default=1000)
parser.add_argument("-sum", action="store_true", default=False)
parser.add_argument("-feature", choices=["all", "half", "utr", "rpl"])
parser.add_argument("-save", "--save_path", type=str)
parser.add_argument("-read", "--read_path", type=str)
parser.add_argument("-log_step", "--log_step", type=int,
                    default=100)
args = parser.parse_args()


if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
if os.path.exists(os.path.join(args.save_path, "logs")):
    shutil.rmtree(os.path.join(args.save_path, "logs"))

writer = SummaryWriter(log_dir=os.path.join(args.save_path, "logs"))

seed_id = 0
np.random.seed(seed_id)
torch.manual_seed(seed_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
enc_hidden_size = args.enc_hidden
enc_n_layers = args.enc_layer
cpr_hidden_size = args.cpr_hidden
cpr_output_size = args.cpr_out
cpr_n_layers = args.cpr_layer
dropout = args.dropout
lr = args.learning_rate
enc_lr = lr
cpr_lr = lr
n_epoch = args.max_epoch
file_path = os.path.abspath(args.read_path)
if args.feature == "all":
    feature_list = [[0, 1, 3], [2, 3, 1]]
elif args.feature == "half":
    feature_list = [[0, 1, 3]]
elif args.feature == "utr":
    feature_list = [[0]]
elif args.feature == "rpl":
    feature_list = [[1]]
cpr_init_num = len(feature_list[0])
clf_init_num = len(feature_list)
batch_size = args.batch_size

param = dict()
param["read"] = file_path
param["enc_hidden_size"] = enc_hidden_size
param["enc_n_layers"] = enc_n_layers
param["cpr_hidden_size"] = cpr_hidden_size
param["cpr_output_size"] = cpr_output_size
param["cpr_n_layers"] = cpr_n_layers
param["dropout"] = dropout
param["lr"] = lr
param["enc_lr"] = lr
param["cpr_lr"] = lr
param["n_epoch"] = n_epoch
param["feature_list"] = feature_list
param["batch_size"] = batch_size


def getConfMatrix(references, prediction):
    conf_mat = [[0, 0], [0, 0]]
    for r, p in zip(references, prediction):
        conf_mat[1-r][1-p] += 1
    return np.array(conf_mat)


def train_iter():
    dataset = data_tool.load(file_path)
    source = dataset["train"]
    dev_source = dataset["valid"]
    test = dataset["test"]
    dic = dataset["dict"]
    dev_loss_best = 10**5

    embedding = nn.Embedding(len(dic), enc_hidden_size)
    encoder = model.Encoder(enc_hidden_size, embedding,
                            enc_n_layers, dropout).to(device)
    compressor = model.MLP(
        enc_hidden_size*cpr_init_num*2, cpr_hidden_size, cpr_output_size, cpr_n_layers).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=enc_lr)
    compressor_optimizer = optim.Adam(compressor.parameters(), lr=cpr_lr)
    step_tmp = 0
    for i_iter in trange(n_epoch, ncols=0):
        encoder.train()
        compressor.train()
        loss_all = 0
        data_iterator = data_tool.iterator(
            source[0], source[1], batch_size)
        pres = []
        target = []
        loss_tmp = 0
        length_all = []
        with tqdm(data_iterator, total=len(source[0])//batch_size, ncols=0) as t:
            for i, (data, length, label) in enumerate(t):
                data = data_tool.padding(data)
                loss, pre = modelAll(data, length, label, encoder, compressor, embedding,
                                     encoder_optimizer, compressor_optimizer, "train")
                loss_all += loss.item()
                loss_tmp += loss.item()
                pres += [int(label) for label in pre]
                target += label
                t.set_postfix({"ave loss": loss_all/(i+1)})
                a_l = loss_all/(i+1)
                step_tmp += 1
                length_all += list(length)
                if step_tmp % args.log_step == 0:
                    writer.add_scalar("loss_step/train", loss_tmp /
                                      min(step_tmp+1, args.log_step), step_tmp)
                    loss_tmp = 0
        loss_all = a_l
        tqdm.write(f"TRAIN loss: {loss_all}")
        try:
            conf_mat = getConfMatrix(target, pres)
            mat_flt = conf_mat.ravel()
            acc = (conf_mat[0, 0]+conf_mat[1, 1])/np.sum(conf_mat)
            prc = conf_mat[0, 0]/np.sum(conf_mat[:, 0])
            rcl = conf_mat[0, 0]/np.sum(conf_mat[0])
            if prc+rcl != 0:
                fms = 2*prc*rcl/(prc+rcl)
            else:
                fms = 0
        except:
            tqdm.write("use sklearn")
            mat_flt = confusion_matrix(target, pres).ravel()
            acc = accuracy_score(target, pres)
            prc = precision_score(target, pres)
            rcl = recall_score(pres, target)
            fms = f1_score(target, pres)
        tqdm.write(
            f"Train accuracy: {acc:.2f}, f1: {fms:.2f}, precision: {prc:.2f}, recall: {rcl:.2f}")
        tqdm.write(
            f"#TP:{mat_flt[0]}, #FN:{mat_flt[1]}, #FP:{mat_flt[2]}, #TN:{mat_flt[3]}\n")
        writer.add_scalar("accuracy/train", acc, i_iter)
        writer.add_scalar("loss/train", loss_all, i_iter)
        for i, part in enumerate(["utr1", "rpl1", "utr2", "rpl2"]):
            num_per_len = defaultdict(int)
            cor_per_len = defaultdict(int)
            lengths = [l for j, l in enumerate(length_all) if j % 4 == i]
            for t, p, l in zip(target, pres, lengths):
                num_per_len[l] += 1
                cor_per_len[l] += int(t == p)
            counts = [l for l in set(lengths) for _ in range(
                round(100*cor_per_len[l]/num_per_len[l]))]
            writer.add_histogram(
                f"accPer{part}Len/train", np.array(counts), i_iter)
        if i_iter == 0:
            writer.add_histogram(f"Len/train", np.array(length_all), i_iter)

        encoder.eval()
        compressor.eval()
        data_iterator = data_tool.iterator(
            dev_source[0], dev_source[1], batch_size)
        pres = []
        target = []
        loss_all = 0
        length_all = []
        with torch.no_grad():
            for data, length, label in tqdm(data_iterator):
                data = data_tool.padding(data)
                loss, pre = modelAll(data, length, label, encoder, compressor, embedding,
                                     encoder_optimizer, compressor_optimizer, "test")
                pres += [int(label) for label in pre]
                target += label

                loss_all += loss.item()
                length_all += list(length)

        loss_all /= (len(dev_source[0])//batch_size +
                     (len(dev_source[0]) % batch_size != 0))
        mat_flt = confusion_matrix(target, pres).ravel()
        acc = accuracy_score(target, pres)
        prc = precision_score(target, pres)
        rcl = recall_score(pres, target)
        fms = f1_score(target, pres)
        tqdm.write(
            f"Dev loss: {loss_all}, best_loss: {min(dev_loss_best,loss_all)}\n")
        tqdm.write(
            f"DEV accuracy: {acc:.2f}, f1: {fms:.2f}, precision: {prc:.2f}, recall: {rcl:.2f}")
        tqdm.write(
            f"#TP:{mat_flt[0]}, #FN:{mat_flt[1]}, #FP:{mat_flt[2]}, #TN:{mat_flt[3]}\n")
        writer.add_scalar("accuracy/test", acc, i_iter)
        writer.add_scalar("loss/test", loss_all, i_iter)
        for i, part in enumerate(["utr1", "rpl1", "utr2", "rpl2"]):
            num_per_len = defaultdict(int)
            cor_per_len = defaultdict(int)
            lengths = [l for j, l in enumerate(length_all) if j % 4 == i]
            for t, p, l in zip(target, pres, lengths):
                num_per_len[l] += 1
                cor_per_len[l] += int(t == p)
            counts = [l for l in set(lengths) for _ in range(
                round(100*cor_per_len[l]/num_per_len[l]))]
            writer.add_histogram(
                f"accPer{part}Len/test", np.array(counts), i_iter)
        if i_iter == 0:
            writer.add_histogram(f"Len/test", np.array(length_all), i_iter)

        torch.save({
            'iteration': i_iter,
            'en': encoder.state_dict(),
            'com': compressor.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'com_opt': compressor_optimizer.state_dict(),
            'loss': loss_all,
            'embedding': embedding.state_dict(),
            'param': param
        }, os.path.join(args.save_path, f'noclf_last.tar'))

        if dev_loss_best > loss_all:
            torch.save({
                'iteration': i_iter,
                'en': encoder.state_dict(),
                'com': compressor.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'com_opt': compressor_optimizer.state_dict(),
                'loss': loss_all,
                'embedding': embedding.state_dict(),
                'param': param
            }, os.path.join(args.save_path, f'noclf_best.tar'))
            dev_loss_best = loss_all

    encoder.eval()
    compressor.eval()
    data_iterator = data_tool.iterator(
        test[0], test[1], batch_size)
    pres = []
    target = []
    with torch.no_grad():
        for data, length, label in tqdm(data_iterator):
            data = data_tool.padding(data)
            loss, pre = modelAll(data, length, label, encoder, compressor, embedding,
                                 encoder_optimizer, compressor_optimizer, "test")
            pres += [int(label) for label in pre]
            target += label

    mat_flt = confusion_matrix(target, pres).ravel()
    acc = accuracy_score(target, pres)
    prc = precision_score(target, pres)
    rcl = recall_score(target, pres)
    fms = f1_score(target, pres)
    tqdm.write(
        f"TEST accuracy: {acc:.2f}, f1: {fms:.2f}, precision: {prc:.2f}, recall: {rcl:.2f}\n#TP:{mat_flt[0]}, #FN:{mat_flt[1]}, #FP:{mat_flt[2]}, #TN:{mat_flt[3]}\n")
    writer.close()


def modelAll(input_tensor, lengths, target, encoder, compressor, embedding, encoder_optimizer, compressor_optimizer, mode):

    encoder_optimizer.zero_grad()
    compressor_optimizer.zero_grad()

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

    if args.sum:
        prob = torch.sum(features.view(batch_size, -1, 2), dim=1)
    else:
        prob = torch.max(features.view(batch_size, -1, 2), dim=1)[0]
    target = torch.LongTensor(
        [int(t) for t in target for _ in range(clf_init_num)]).to(device)
    criterion = nn.NLLLoss()
    loss = criterion(features, target)
    label = torch.argmax(prob, dim=1)

    if mode == "train":
        loss = loss.to(device)
        loss.backward()
        encoder_optimizer.step()
        compressor_optimizer.step()
    return loss, label


if __name__ == "__main__":
    train_iter()
