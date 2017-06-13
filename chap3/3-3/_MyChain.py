import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class MyChain(Chain):
    # 合成関数内のlinks内の関数
    def __init__(self):
        super(MyChain, self).__init__(
        l1 = L.Linear(4, 3),
        l2 = L.Linear(3, 3),)

    # 損失関数
    def __call__(self, x, y):
        fv = self.fwd(x, y)
        loss = F.mean_squared_error(fv, y)
        return loss

    # モデルの順方向計算
    def fwd(self, x, y):
        return F.sigmoid(self.l1(x))
