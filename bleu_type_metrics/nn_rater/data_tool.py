import pickle
import os
import random
import numpy as np
from itertools import zip_longest
from collections import defaultdict, Counter
from tqdm import tqdm


def readFiles(read_dir, prefix):
    ext = ".txt"
    for i, postfix in enumerate(["_ref_utr", "_ref_rpl", "_hyp_utr", "_hyp_rpl"]):
        with open(os.path.join(read_dir, prefix+postfix+ext), "r") as f:
            for line in tqdm(f.readlines(), desc=f"{i+1} / 4 iters", leave=False, ncols=0):
                yield i, line


def constructTrain(read_dir, dict_len=33000, default_w=["<pad>", "<unk>"]):
    w_counter = defaultdict(int)
    train_data = [[] for _ in range(4)]
    for i, line in readFiles(read_dir, "train"):
        line = line.strip("\r\n").split(" ")
        for w in line:
            w_counter[w] += 1
        train_data[i].append(line)
    w_dict = defaultdict(lambda: len(w_dict))
    for w in default_w:
        w_dict[w]
    used_num_tokens = 0
    for w, n in Counter(w_counter).most_common(dict_len):
        w_dict[w]
        used_num_tokens += n
    print_sent = f"use {min(len(w_dict),dict_len)} tokens"
    print_sent += f", unused {len(w_counter)-len(w_dict)} type tokens"
    print_sent += f", Coverage: {min(100,100*used_num_tokens/sum(w_counter.values())):.2f} %"
    print(print_sent)
    # print(
    #    f"use {min(len(w_dict),dict_len)} tokens, Coverage: {min(100,100*used_num_tokens/sum(w_counter.values())):.2f} %")
    train_data = [[[w_dict.get(w, w_dict["<unk>"]) for w in sent]
                   for sent in sents] for sents in train_data]
    train_data = [list(sents) for sents in zip(*train_data)]
    return train_data, w_dict, w_counter


def converId2word(sent, word2id={}):
    if not "id2word" in globals():
        assert len(word2id) != 0
        global id2word
        id2word = {v: k for k, v in word2id.items()}
    return [id2word[word] for word in sent]


def constructTest(read_dir, prefix, w_dict):
    test_data = [[] for _ in range(4)]
    w_types = set()
    unk_types = set()
    unk_counter = 0
    for i, line in readFiles(read_dir, prefix):
        line = line.strip("\r\n").split(" ")
        w_types |= set(line)
        line = [w_dict.get(w, w_dict["<unk>"]) for w in line]
        test_data[i].append(line)
        unk_types |= set(line)
        unk_counter += line.count(w_dict["<unk>"])
    print(
        f"convert {unk_counter} tokens to <unk>, type Coverage: {100*len(unk_types)/len(w_types):.2f} %")
    test_data = [list(sents) for sents in zip(*test_data)]
    return test_data


def readLabels(read_dir):
    prefixs = ["train", "valid", "test"]
    data_labels = {}
    for prefix in prefixs:
        with open(os.path.join(read_dir, prefix+"_label.txt"), "r") as f:
            data_labels[prefix] = [int(line.strip("\r\n"))
                                   for line in f.readlines()]
    return data_labels


def load(read_dir, dict_len=33000, default_w=["<pad>", "<unk>"]):
    file_names = os.listdir(read_dir)
    pkl_files = ["train.pkl", "valid.pkl", "test.pkl", "dict.pkl"]
    dataset = {}
    if set(pkl_files) <= set(file_names):
        print("loading pkl files...", end="")
        for file_name in pkl_files:
            with open(os.path.join(read_dir, file_name), "rb") as f:
                dataset[file_name.split(".")[0]] = pickle.load(f)

        w_counter, dict_len = dataset["dict"]
        w_dict = defaultdict(lambda: len(w_dict))
        for w in default_w:
            w_dict[w]
        for w, _ in Counter(w_counter).most_common(dict_len):
            w_dict[w]
        print("done")
    else:
        print("read train data")
        train_data, w_dict, w_counter = constructTrain(
            read_dir, dict_len, default_w)
        print("read valid data")
        valid_data = constructTest(read_dir, "valid", w_dict)
        print("read test data")
        test_data = constructTest(read_dir, "test", w_dict)
        labels = readLabels(read_dir)
        dataset = {}
        dataset["train"] = [train_data, labels["train"]]
        dataset["valid"] = [valid_data, labels["valid"]]
        dataset["test"] = [test_data, labels["test"]]
        dataset["dict"] = [w_counter, dict_len]
        for file_name in pkl_files:
            with open(os.path.join(read_dir, file_name), "wb") as f:
                pickle.dump(dataset[file_name.split(".")[0]], f)
        with open(os.path.join(read_dir, "vocabulary.txt"), "w") as f:
            for w, _ in Counter(w_counter).most_common(dict_len):
                f.write(w+"\n")
    dataset["dict"] = w_dict
    return dataset


def padding(dialogue, pad=0):
    return np.array(list(zip_longest(*dialogue, fillvalue=pad)))


def iterator(data, label, batch_size, random=True):
    input_tensort = []
    input_lengths = []
    labels = []
    idx_list = list(range(len(data)))
    if random:
        np.random.shuffle(idx_list)
    for idx in idx_list:
        length = np.array([len(sent) for sent in data[idx]])
        input_tensort += list(data[idx])
        input_lengths.append(length)
        labels.append(int(label[idx]))
        if len(labels) >= batch_size:
            yield input_tensort, np.hstack(input_lengths), labels
            input_tensort = []
            input_lengths = []
            labels = []
    if len(labels) != 0:
        yield input_tensort, np.hstack(input_lengths), labels
