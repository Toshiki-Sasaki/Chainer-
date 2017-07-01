#!usr/bin/env python3
# coding: utf-8

from sklearn import datasets
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# データの処理・加工
# コーパスを読み込み、単語にidをつける
vocab = {}
def load_data(filename):
    global vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    # enumerate関数 = リストのインデックスと要素をそれぞれ取り出す
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']


# モデルの定義
class MyRNN(chainer.Chain):
    # 合成関数
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed = L.EmbedID(v, k),
            H = L.Linear(k, k),
            W = L.Linear(k, v),
        )

    # 損失関数
    def __call__(self, s):
        accum_loss = None
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1,k), dtype=np.float32))
        for i in range(len(s)):
            next_w_id = eos_id if (i==len(s)-1) else s[i+1]
            tx = Variable(np.array([next_w_id], dtype=np.int32))
            x_k = self.embed(Variable(np.array([s[i]], dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss


# モデルと最適化
demb = 100
model = MyRNN(len(vocab), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習
for epoch in range(5):
    s = []
    for pos in range(len(train_data)):
        id = train_data[pos]
        s.append(id)
        if id == eos_id:
            model.zerograds()
            loss = model(s)
            loss.backward()
            optimizer.update()
            s = []
    outfile = "myrnn-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
