## Chaierプログラムの全体図
```
(1) データの準備・設定

(2) class MyModel(Chain):
      def __init__(self):
        super(MyModel, self).__init__(
          パラメータを含む関数の宣言
          )
      def __call__(self, .....):
        損失関数

(3) model = MyModel()
    optimizer = optimizer.Adam()
    optimizer.setup(model)

(4) for epoch in range(繰り返し回数):
      データの加工
      model.zerograds()
      loss = model(......)
      loss.backward()
      optimzer.update()

(5) 結果の出力
```
