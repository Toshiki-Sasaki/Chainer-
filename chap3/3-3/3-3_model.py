import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import _MyChain as M

'''
基本的なコードの例
（動きません）
'''


def initialize():
    model = M.MyChain() #モデルの生成
    optimizer = optimizers.SGD() # 最適化アルゴリズムの選択
    optimizer.setup(model) # アルゴリズムにモデルをセット
    return model

def loopProcess(model, x, y):
    loss = model(x, y) # 順方向計算して誤差を算出
    loss.backward() # 逆方向の計算、勾配の計算
    optimizer.update() # パラメータを更新

model = initialize()
model.zerograds() # 勾配の初期化

for _ in range(3):
    loopProcess(model, x, y)
