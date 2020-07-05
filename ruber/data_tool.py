import pickle
import os
import numpy as np
from itertools import zip_longest
from collections import defaultdict, Counter
from tqdm import tqdm


def readFiles(read_dir, prefix):
    ext = ".txt"
    for i, postfix in enumerate(["_utr", "_rpl"]):
        read_files = [f for f in os.listdir(read_dir) if f.endswith(
            postfix+ext) and f.startswith(prefix)]
        for rf in read_files:
            with open(os.path.join(read_dir, rf), "r") as f:
                for line in tqdm(f.readlines(), desc=f"{i+1} / 2 iters: read {rf}", leave=False, ncols=0):
                    yield i, line


def constructTrain(read_dir, dict_len=33000, default_w=["<pad>", "<unk>"]):
    w_counter = defaultdict(int)
    train_data = [[] for _ in range(2)]
    print("read train file")
    for i, line in readFiles(read_dir, "train"):
        line = line.strip("\r\n").split(" ")
        for w in line:
            w_counter[w] += 1
        train_data[i].append(line)
    assert len(train_data[0]) == len(train_data[1]), "different line number"
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
    #print(f"use {min(len(w_dict),dict_len)} tokens, Coverage: {min(100,100*len(w_dict)/len(w_counter)):.2f} %")
    train_data = [[[w_dict.get(w, w_dict["<unk>"]) for w in sent]
                   for sent in sents] for sents in train_data]

    train_data = [list(sents) for sents in zip(*train_data)]
    w_counter = dict(w_counter)
    w_dict = dict(w_dict)
    return train_data, w_dict, w_counter


def converId2word(sent, word2id={}):
    if not "id2word" in globals():
        assert len(word2id) != 0
        global id2word
        id2word = {v: k for k, v in word2id.items()}
    return [id2word[word] for word in sent]


def constructTest(read_dir, prefix, w_dict):
    test_data = [[] for _ in range(2)]
    w_counter = Counter()
    unk_counter = 0
    for i, line in readFiles(read_dir, prefix):
        line = line.strip("\r\n").split(" ")
        w_counter.update(Counter(line))
        line = [w_dict.get(w, w_dict["<unk>"]) for w in line]
        test_data[i].append(line)
        unk_counter += line.count(w_dict["<unk>"])
    print(
        f"convert {unk_counter} tokens to <unk>, frequency ratio: {100*unk_counter/sum(w_counter.values()):.2f} %")
    test_data = [list(sents) for sents in zip(*test_data)]
    return test_data


def load(read_dir, dict_len=33000, default_w=["<pad>", "<unk>"]):
    dataset = {}
    prefix = ["train", "valid", "test", "dict"]
    if os.path.exists(os.path.join(read_dir, "ruber_pickled")):
        for file_name in prefix:
            with open(os.path.join(read_dir, "ruber_pickled", file_name+".pkl"), "rb") as f:
                dataset[file_name] = pickle.load(f)

        w_counter, dict_len = dataset["dict"]
        w_dict = defaultdict(lambda: len(w_dict))
        for w in default_w:
            w_dict[w]
        for w, _ in Counter(w_counter).most_common(dict_len):
            w_dict[w]
        w_dict = dict(w_dict)
    else:
        train_data, w_dict, w_counter = constructTrain(
            read_dir, dict_len, default_w)
        print("read valid file")
        valid_data = constructTest(read_dir, "valid", w_dict)
        print("read test file")
        test_data = constructTest(read_dir, "test", w_dict)
        dataset = {}
        dataset["train"] = train_data
        dataset["valid"] = valid_data
        dataset["test"] = test_data
        dataset["dict"] = [w_counter, dict_len]
        os.mkdir(os.path.join(read_dir, "ruber_pickled"))
        for file_name in prefix:
            with open(os.path.join(read_dir, "ruber_pickled", file_name+".pkl"), "wb") as f:
                pickle.dump(dataset[file_name], f)

    dataset["dict"] = w_dict
    for prefix in ["train", "valid", "test"]:
        dataset[prefix] = np.array(dataset[prefix])
    return dataset


def padding(dialogue, pad=0):
    """
    :Input Param:
    dialogue: array of utr-rpl pairs (np.array (n*2*each length))
    pad: padding id (int)
    :Output:
    the padded data with pad (np.array (n*2*max length))
    """
    return np.array(list(zip_longest(*dialogue, fillvalue=pad)))


def count_lens(data):
    """
    :Input Param:
    data: array of utr-rpl pairs (np.array (n*2*each length))
    :Output:
    the lengths of data (np.array (n*2))
    """
    return np.array([[len(sent) for sent in sents] for sents in data])


def iterator(data, batch_size, np_rdm_gen=None):
    """
    :Input Param:
    data: array of utr-rpl pairs (np.array (n*2*each length))
    batch_size: batch_size (int)
    rdm_gen: np.random
    :Output:
    a part of data (np.array (m*2*each length) (m<=n))
    """
    idx_list = list(range(len(data)))
    if np_rdm_gen is not None:
        np_rdm_gen.shuffle(idx_list)
    for i in range(0, len(data), batch_size):
        yield data[idx_list[i:i+batch_size]]
