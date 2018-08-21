from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_svmlight_file

def trainMultinomialNB(X, y) :
	print("Training started with MultinomialNB...")
	classifier = MultinomialNB()
	classifier.fit(X, y)
	print("Training completed...")
	return classifier


def trainLogisticReg(X, y) :
	print("Training started with LogisticRegression...")
	classifier = LogisticRegression()
	classifier.fit(X, y)
	print("Training completed...")
	return classifier


def trainSVM(X, y) :
	print("Training started with SVM...")
	classifier = svm.SVC(decision_function_shape = 'OVO')
	classifier.fit(X, y)
	print("Training completed...")
	return classifier


def trainFeedFwdNN(X, y, hiddenLayerShape) :
	print("Training started with FeedforwardNeuralNet...")
	classifier = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
		hidden_layer_sizes = hiddenLayerShape, random_state = 1)
	classifier.fit(X, y)
	print("Training completed...")
	return classifier


def scoreSKLearn(model, X, y) :
	return model.score(X, y)


def predictSKLearn(model, X) :
	return model.predict(X)


def main() :
	trainFold = "aclImdb/train/"
	testFold = "aclImdb/test/"

	hiddenLayerShapeNN = {
		"labeledBbow.feat" : (50, 2),
		"labeledBow.feat" : (5, 2),
		"labeledNormTf.feat" : (80, 4),
		"labeledTfIdf.feat" : (50, 2),
		"labeledAvgWord2Vec.feat" : (80, 6),
		"labeledAvgTfIdfWord2Vec.feat" : (80, 4),
		"labeledAvgGlove.feat" : (80, 6),
		"labeledAvgTfIdfGlove.feat" : (20, 8),
		"labeledParaVec.feat" : (80, 4)
	}

	classifiers = ["MultinomialNB", "LogisticRegression", "SVM", "FeedforwardNN"]

	print("Which representation of document to use ?")
	print("Choose among the following :")
	print("labeledBbow.feat\tlabeledBow.feat\t\t\tlabeledNormTf.feat")
	print("labeledTfIdf.feat\tlabeledAvgWord2Vec.feat\t\tlabeledAvgTfIdfWord2Vec.feat")
	print("labeledAvgGlove.feat\tlabeledAvgTfIdfGlove.feat\tlabeledParaVec.feat")
	docRep = input("Write here : ")

	if not (docRep in hiddenLayerShapeNN.keys()) :
		print("The representation " + docRep + " not found")
		docRep = input("Write again : ")

		if not (docRep in hiddenLayerShapeNN.keys()) :
			print("The representation " + docRep + " not found")
			print("Choosing default representation : labeledBbow.feat")
			docRep = "labeledBbow.feat"

	print("Which classifier to use ?")
	print("Choose among the following :")
	print("MultinomialNB\t\tLogisticRegression\t\tSVM")
	print("FeedforwardNN")
	classifier = input("Write here : ")

	if not (classifier in classifiers) :
		print("The classifer " + classifier + " not found")
		classifier = input("Write again : ")

		if not (classifier in classifiers) :
			print("The classifer " + classifier + " not found")
			print("Choosing default classifier : MultinomialNB")
			classifier = "MultinomialNB"

	trainX, trainY = load_svmlight_file(trainFold + docRep)
	testX, testY = load_svmlight_file(testFold + docRep)
	trainY = [(lambda x : 1 if x >= 7 else 0)(i) for i in trainY]
	testY = [(lambda x : 1 if x >= 7 else 0)(i) for i in testY]

	if classifier == "MultinomialNB" :
		clf = trainMultinomialNB(trainX, trainY)
		acc = scoreSKLearn(clf, testX, testY)
		print("Accuracy (", docRep, ", MultinomialNB) : ", acc)
	elif classifier == "LogisticRegression" :
		clf = trainLogisticReg(trainX, trainY)
		acc = scoreSKLearn(clf, testX, testY)
		print("Accuracy (", docRep, ", LogisticRegression) : ", acc)
	elif classifier == "SVM" :
		clf = trainSVM(trainX, trainY)
		acc = scoreSKLearn(clf, testX, testY)
		print("Accuracy (", docRep, ", SVM) : ", acc)
	elif classifier == "FeedforwardNN" :
		print("Default hidden layer shape : ", hiddenLayerShapeNN[docRep])

		clf = trainFeedFwdNN(trainX, trainY, hiddenLayerShapeNN[docRep])
		acc = scoreSKLearn(clf, testX, testY)
		print("Accuracy (", docRep, ", FeedforwardNeuralNet) : ", acc)


if __name__ == "__main__" :
	main()
