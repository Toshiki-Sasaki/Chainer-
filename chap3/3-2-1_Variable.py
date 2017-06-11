import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

x1 = Variable(np.array([1]).astype(np.float32))
x2 = Variable(np.array([2]).astype(np.float32))
x3 = Variable(np.array([3]).astype(np.float32))

print(x1.data)

z = (x1 - 2 * x2 - 1) ** 2 + (x2 * x3 - 1) ** 2 + 1
print(z.data)

z.backward()

print(x1.grad)
print(x2.grad)
print(x3.grad)
