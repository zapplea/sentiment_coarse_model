1. transfer: 加载pre-trained coarse model， 然后输入fine grained data中某个aspect#attribute对应的所有句子，这样在coarse model中得到所有aspect的score. 用分值最高的aspect 的aspect matrix 去初始化该 fine graine model中 aspect#attribute的 attribute matrix。

2. train: 在初始化完成后，用fine grained data训练 fine grained model.

注意要确保fine grained model中被初始化的attribute matrix的序号，与output layer中该attribute的序号一一对应。
为了达到这个目的，可以：transfer时，如果aspect#attribute是第i个选择aspect matrix的，那么，在训练fine grained model时， 该aspect#attribute是output layer中第i个输出。

运行程序时，先调用transfer,输出initializer_A与initializer_O, 然后调用train函数，并把这两个矩阵输入。

代码结构：
transfer_nn/transfer/transfer.py 包含transfer函数以及需要输入初始化的attribute_mat, attribute_vec(之前的版本都是随机初始化).
transfer_nn/1pNw/classifier.py, 对1pNw 的transfer learning. coarse model调用coarse_nn中原代码. fine model重新写了一下,是fine_classifier函数，由于attribute_mat与attriubte_vec要用需要输入初始化的版本。
functions/train/trans_atr_train.py里是训练fine_nn的函数。


输入：
transfer: [aspect#attribute1 涉及的所有句子，aspect#attribute2涉及的所有句子, ...] 可以在transfer_nn/transfer/transfer.py的transfer函数中看到，有一个self.coarse_data_generator.fine_sentences() 函数，这个返回transfer要用的数据， 需要写一个可以提供数据的函数。

train: 同之前fine_model的输入