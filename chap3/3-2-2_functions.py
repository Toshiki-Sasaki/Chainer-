import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

x = Variable(np.array([-1]).astype(np.float32))

print(F.sin(x).data)
print(F.sigmoid(x).data)
# >>> [-0.84147096]
# >>> [ 0.2689414]

x = Variable(np.array([-0.5]).astype(np.float32))
z = F.cos(x)
print(z.data)
# >>> [ 0.87758255]
z.backward()
print(x.grad) # -sin(x)=-sin(-0.5)
# >>> [ 0.47942555]
print(((-1) * F.sin(x)).data) # 確認 = x.grad
# >>> [ 0.47942555]

x = Variable(np.array([-1, 0, 1]).astype(np.float32))
z = F.sin(x)
z.grad = np.ones(3, dtype=np.float32)
z.backward()
print(x.grad)
# >>> [ 0.54030228  1.          0.54030228]
