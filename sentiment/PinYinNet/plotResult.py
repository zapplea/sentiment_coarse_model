import pickle
import matplotlib.pyplot as plt

hist_onlyWord = pickle.load(open( "OnlyWordHistoryDict.pkl", "rb" ))
hist_onlyPinYin = pickle.load(open( "OnlyPinYinHistoryDict.pkl", "rb" ))
hist_WordAndPinYin = pickle.load(open("WordAndPinYinHistoryDict.pkl","rb"))

plt.plot(hist_onlyWord['val_acc'])
plt.plot(hist_onlyPinYin['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Only word embedding', 'Only pinyin embedding'], loc='lower right')
plt.show()

plt.plot(hist_onlyWord['loss'])
plt.plot(hist_onlyPinYin['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Only word embedding', 'Only pinyin embedding'], loc='upper left')
plt.show()

plt.plot(hist_onlyWord['val_acc'])
plt.plot(hist_WordAndPinYin['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Only word embedding', 'word and pinyin embedding'], loc='lower right')
plt.show()

plt.plot(hist_onlyWord['loss'])
plt.plot(hist_WordAndPinYin['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Only word embedding', 'word and pinyin embedding'], loc='upper left')
plt.show()