import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# 重みの行列サイズ
h = L.Linear(3, 4)

# 重みの行列の値は適当に入っている
print(h.W.data)
'''
>>> [[-0.64436221 -0.42398417  0.06947934]
 [-0.00293008  1.41035414  0.19109707]
 [ 0.11882336 -0.73664939 -0.47767338]
 [-0.52032989 -0.21953587  0.38276732]]
'''
# バイアス項は0
print(h.b.data)
# >>> [ 0.  0.  0.  0.]

x = Variable(np.array(range(6)).astype(np.float32).reshape(2,3))
print(x.data)
'''
>>> [[ 0.  1.  2.]
 [ 3.  4.  5.]]
'''
y = h(x)
print(y.data)
'''
>>> [[-0.28502548  1.7925483  -1.6919961   0.54599875]
 [-3.2816267   6.58811188 -4.97849417 -0.52529657]]
'''

# y = Wx + b
# が正しく計算できているのか確認
w = h.W.data
x0 = x.data
print(x0.dot(w.T) + h.b.data) #dot->行列の積
'''
>>> [[-0.28502548  1.7925483  -1.6919961   0.54599875]
 [-3.2816267   6.58811188 -4.97849417 -0.52529657]]
'''
