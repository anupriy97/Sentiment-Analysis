import tensorflow as tf
import numpy as np
from sklearn.datasets import load_svmlight_file
import sys

def shuffleData(X, y) :
	state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(state)
	np.random.shuffle(y)
	return X, y


if len(sys.argv) > 1 :
	fileName = sys.argv[1]
else :
	fileName = "labeledParaVec.feat"

trainX, trainY = load_svmlight_file("aclImdb/train/" + fileName)
testX, testY = load_svmlight_file("aclImdb/test/" + fileName)

trainY = np.asarray([(lambda x : [1, 0] if x >= 7 else [0, 1])(i) for i in trainY])
testY = np.asarray([(lambda x : [1, 0] if x >= 7 else [0, 1])(i) for i in testY])

# trainY = np.reshape(np.asarray([(lambda x : 1 if x >= 7 else 0)(i) for i in trainY]), (len(trainY), 1))
# testY = np.reshape(np.asarray([(lambda x : 1 if x >= 7 else 0)(i) for i in testY]), (len(testY), 1))

trainX = trainX.toarray()
testX = testX.toarray()

featDim = trainX.shape[1]
labelDim = 2

x = tf.placeholder(tf.float64, [None, featDim])
y = tf.placeholder(tf.float64, [None, labelDim])

W1 = tf.Variable(tf.truncated_normal([featDim, 10], dtype = tf.float64))
b1 = tf.Variable(tf.constant(0.1, dtype = tf.float64, shape = [10]))

W2 = tf.Variable(tf.truncated_normal([10, 10], dtype = tf.float64))
b2 = tf.Variable(tf.constant(0.1, dtype = tf.float64, shape = [10]))

W3 = tf.Variable(tf.truncated_normal([10, labelDim], dtype = tf.float64))
b3 = tf.Variable(tf.constant(0.1, dtype = tf.float64, shape = [labelDim]))

h1 = tf.matmul(x, W1) + b1
h2 = tf.matmul(h1, W2) + b2
yPred = tf.matmul(h2, W3) + b3

# W = tf.Variable(tf.truncated_normal([featDim, labelDim], tf.float64))
# b = tf.Variable(tf.constant(0.1, [labelDim], tf.float64))
# yPred = tf.matmul(x, W) + b


cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = yPred))
# binary_cross_entropy = tf.reduce_mean(
# 	tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = yPred))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-1).minimize(binary_cross_entropy)
correct_prediction = tf.equal(tf.argmax(yPred, 1), tf.argmax(y, 1))
# correct_prediction = tf.equal(tf.round(yPred), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(100) :
	trainX, trainY = shuffleData(trainX, trainY)
	for i in range(50) :
		batch_x = trainX[i*500:(i+1)*500]
		batch_y = trainY[i*500:(i+1)*500]

		if (i + epoch * 50) % 50 == 0 :
			train_accuracy = sess.run(accuracy, feed_dict = {x : batch_x, y : batch_y})
			loss = sess.run(cross_entropy, feed_dict = {x : batch_x, y : batch_y})
			print("step %d, training accuracy %g, loss %g" % (i + epoch * 50, train_accuracy, loss))
			# print(sess.run(W))
		sess.run(train_step, feed_dict = {x : batch_x, y : batch_y})

testX, testY = shuffleData(testX, testY)
print(sess.run(accuracy, feed_dict = {x: testX, y: testY}))
# print(sess.run(yPred, feed_dict = {x : testX[:100], y : testY[:100]}))
# print(testX[0])
# print(testX[1])
# print(sess.run(W1))
# print(sess.run(W2))
# print(sess.run(W3))
# print(sess.run(b1))
# print(sess.run(b2))
# print(sess.run(b3))
# print(sess.run(W))
# print(sess.run(b))



# AvgWord2Vec : 1e-3
# AvgTfIdfWord2Vec : 1e-3