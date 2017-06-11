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
>>> [[ 0.04812841  0.76700783 -0.4824613 ]
 [-0.47151822  1.11389458  0.85145938]
 [ 0.2713083  -0.25092939  0.64247173]
 [ 0.14352991 -0.52951211 -0.45783317]]
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
>>> [[ 0.33174902  2.17291737 -1.45434558 -1.38527989]
 [ 0.69124097  6.41357517 -2.47088194 -5.65372753]]
'''

# y = Wx + b
# が正しく計算できているのか確認
w = h.W.data
x0 = x.data
print(x0.dot(w.T) + h.b.data) #dot->行列の積
'''
>>> [[-0.95913559  0.9041096   0.63596058 -0.39530826]
 [-3.67030454  3.58149552  1.04833257  1.33901405]]
'''
