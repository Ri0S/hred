import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

use_cuda = torch.cuda.is_available()


def custom_collate_fn(batch):
    batch = Variable(torch.LongTensor(batch)).squeeze(0)
    if use_cuda:
        batch = batch.cuda()

    return batch.transpose(0, 1)


class MovieTriples(Dataset):
    def __init__(self, data_type, batch_size):
        if data_type == 'train':
            path = './data/smart_ko_multi_train_extend.pkl'
            plen = './data/smart_ko_multi_train_extend_len.txt'
            # path = './data/movie_triple_train.pkl'
            # plen = './data/movie_triple_train_len.txt'
        elif data_type == 'test':
            path = './data/smart_ko_multi_test_extend.pkl'
            plen = './data/smart_ko_multi_test_extend_len.txt'
            # path = './data/movie_triple_test.pkl'
            # plen = './data/movie_triple_test_len.txt'
        elif data_type == 'valid':
            path = './data/smart_ko_multi_valid_extend.pkl'
            plen = './data/smart_ko_multi_valid_extend_len.txt'
            # path = './data/movie_triple_valid.pkl'
            # plen = './data/movie_triple_valid_len.txt'

        self.ds = []

        with open(path, 'rb') as f:
            f = pickle.load(f)
            index = 0
            with open(plen, encoding='utf-8') as f2:
                length = f2.read().split()
                for tl, seqlen in enumerate(length):
                    maxlen = 0
                    t = []
                    for i in range(int(seqlen)):
                        us = f[index]
                        index += 1
                        utters = []
                        utter = []
                        for w in us:
                            utter.append(w)
                            if w == 2:
                                utters.append(utter)
                                if len(utter) > maxlen:
                                    maxlen = len(utter)
                                utter = []
                        t.append(utters)
                        if i % batch_size == batch_size - 1:
                            for i in range(batch_size):
                                for j in range(tl + 2):
                                    t[i][j].extend([0 for _ in range(maxlen - len(t[i][j]))])
                            self.ds.append(t)
                            t = []
                            maxlen = 0

                    if len(t) != 0:
                        for i in range(len(t)):
                            for j in range(tl + 2):
                                t[i][j].extend([0 for _ in range(maxlen - len(t[i][j]))])
                        self.ds.append(t)  # batch turn word
                    # break  # single turn test

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def tensor_to_sent(x, inv_dict, greedy=False):
    sents = []
    inv_dict[0] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
        for i in seq:
            sent.append(inv_dict[i])
            if i == 2:
                break
        sents.append((" ".join(sent), scr))
    return sents

# dsets = MovieTriples('train')
# dataloader = DataLoader(dsets, batch_size=100, shuffle=True, collate_fn=custom_collate_fn)
# for i in dataloader:
#     print(i)