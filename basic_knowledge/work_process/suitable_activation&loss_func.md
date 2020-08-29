# 合適的 loss func 與 輸出端 activation

<br>

------------------------

<br>

以下列舉幾項常用的 loss func 與 activation 對應的深度學習問題。

<br>


| problem | output layer ativation | loss func |
|---|---|---|
|二元分類問題|sigmoid|binary_crossentropy|
|多類別，單標籤分類問題|softmax|categorical_crossentropy|
|多類別，多標籤分類問題|sigmoid|binary_crossentropy|
|回歸任意數值問題|None|mse|
|回歸 0~1 之間數值問題|sigmoid|mse 或 binary_crossentropy|