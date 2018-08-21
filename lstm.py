# LSTM for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# from sklearn.datasets import load_svmlight_file

# def shuffleData(X, y) :
# 	state = np.random.get_state()
# 	np.random.shuffle(X)
# 	np.random.set_state(state)
# 	np.random.shuffle(y)
# 	return X, y

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
# (X_train, y_train) = load_svmlight_file("aclImdb/train/labeledParaVec.feat")
# (X_test, y_test) = load_svmlight_file("aclImdb/test/labeledParaVec.feat")
# X_train = X_train.toarray()
# X_test = X_test.toarray()
# shuffleData(X_train, y_train)
# shuffleData(X_test, y_test)
# y_train = [(lambda x : 1 if x >= 7 else 0)(i) for i in y_train]
# y_test = [(lambda x : 1 if x >= 7 else 0)(i) for i in y_test]
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
# model.add(Dense(32, input_shape=(150,)))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))