# 關於 Dropout

<br>

---

<br>

Dropout 是用來解決訓練的 Model 過度擬合的問題的。一般來說甚麼樣的情況會影響擬合呢 ? 有兩個主要因素

1. 模型的容量

2. 資料的多寡

模型的容量指 Hidden Layer 的個數，資料多寡應該不需太多解釋。Dropout 是一種對抗過擬合的正則化方法。在訓練時每一次的迭代 （epoch）皆以一定的機率 "丟棄"　Hidden Layer　神經元。被丟掉其實就是阻斷他們不往後傳遞訊號。

![PetarV-/TikZ](https://camo.githubusercontent.com/06e05aa7d7cd9803221f4a7f0c113aa0bf6a97b4/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f347a36646636656a393075687935352f64726f706f75742e706e673f7261773d31)

[(圖片來源)](https://github.com/PetarV-/TikZ/tree/master/Dropout)

<br>

同時再做反向傳播更新 Layer 參數時，被隱藏的神經元梯度為 0。所以不會造成過度依賴某些特定神經元。

<br>

在 tensorflow 中很輕易的就可以建立一個 Dropout 層 : 

```py
rate = 0.1
dropout = tf.keras.layers.Dropout(rate)  # 隨機關閉 10% 的神經元
```

<br>

在實際使用時記得要告知 dropout 目前是訓練還是測試 : 

```py
x = dropout(x, training=True)  # 很重要!
```

告知 Dropout 的目的首先其一是在實際 demo 使用時我們不會想要漏掉任何一個神經元的運算結果。但是這樣一來會造成訓練結果筆平時訓練階段大一點，因為每次都以 i 的機率丟棄神經元嘛，總不能說 demo 時神經元全開然後就安心使用吧。

實際上使用時，結果會大上 1/(1−p) 倍，需要將結果除以 1−p。讓訓練時的期望值保持不變。

當然，這些步驟已經不需要我們寫實現了，使用 keras 的 Dropout 時，改變 `training` 參數就好了。

