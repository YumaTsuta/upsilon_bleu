import torch
from torch import nn


def oneBatch(query, pos_rpl, neg_rpl, query_lens, pos_rpl_lens, neg_rpl_lens, model, optimizer, margin, mode):
    assert mode in ["train", "test"]
    model.zero_grad()

    # batch_size = len(query_lens)
    pos = model(query, pos_rpl, query_lens, pos_rpl_lens)
    neg = model(query, neg_rpl, query_lens, neg_rpl_lens)
    criterion = nn.ReLU()
    loss = torch.sum(criterion(margin-pos+neg))

    if mode == "train":
        loss.backward()
        optimizer.step()
    return loss


def oneBatch_test(query, pos_rpl, query_lens, pos_rpl_lens, model):
    model.zero_grad()

    # batch_size = len(query_lens)
    pos = model(query, pos_rpl, query_lens, pos_rpl_lens)

    return pos


class UnreferencedModel(nn.Module):
    def __init__(self, vocab_size, gru_hidden, embed_dim=128):
        super(UnreferencedModel, self).__init__()
        self.layer = nn.Linear(gru_hidden*2, gru_hidden*2)
        self.BiGRU_que = BiGRU(vocab_size, gru_hidden, embed_dim=embed_dim)
        self.BiGRU_rpl = BiGRU(vocab_size, gru_hidden, embed_dim=embed_dim)
        self.MLP = MLP(gru_hidden*4+1)

    def forward(self, query, reply, query_lens, reply_lens):
        _, que = self.BiGRU_que(query, query_lens)
        _, rpl = self.BiGRU_rpl(reply, reply_lens)
        que = torch.cat([que[0], que[1]], dim=1)
        rpl = torch.cat([rpl[0], rpl[1]], dim=1)
        M = (self.layer(que)*(rpl)).sum(1, keepdim=True)
        return self.MLP(torch.cat((que, M, rpl), -1))


class BiGRU(nn.Module):
    def __init__(self, vocab_size, gru_hidden, embed_dim=128):
        super(BiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, 0, 0.05)
        self.gru_hidden = gru_hidden
        self.GRU = nn.GRU(embed_dim, gru_hidden, bidirectional=True)

    def forward(self, input_seq, input_len):
        gru_input = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, input_len, enforce_sorted=False)
        return self.GRU(packed)


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_batch):
        h_ = self.tanh(self.layer1(input_batch))
        return self.sigmoid(self.layer2(h_))
