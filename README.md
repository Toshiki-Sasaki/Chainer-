# Chainerによる実践深層学習

## 第0章 Chainerとは
### Chainer のインストール方法
```
$ pip install chainer

$ # 本では version 1.10.0 を使用
$ # 最新版は2.0.0(2017/6/11)
$ # アップグレード
$ pip install -U chainer
```

### chainer の特徴
https://research.preferred.jp/2015/06/deep-learning-chainer/

## 第1章 Numpyで最低限知っておくこと
### 乱数の配列
```
$ # 標準正規分布に従う乱数を5つ生成
$ np.random.randn(5)

$ # 区間(0,1)の一様分布に従う乱数を3個生成
$ np.random.uniform(0, 1, 3)

$ # 平均1.5、標準偏差2の正規分布に従う乱数を3個生成
$ np.random.normal(1.5, 2.0, 3)
```

## 第2章 ニューラルネットのおさらい
一旦飛ばす


## 第3章 Chainer の使い方
### 基本オブジェクト
```
$ import numpy as np
$ import chainer
$ from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
$ from chainer import Link, Chain, ChainList
$ import chainer.functions as F
$ import chainer.links as L
```

### 変数のノードに対応するオブジェクトの生成
```
$ # 実数->np.float32, 整数->np.int32
$ x1 = Variable(np.array([1]).astype(np.float32))
```



### Optimizer の種類
http://docs.chainer.org/en/stable/reference/optimizers.html
