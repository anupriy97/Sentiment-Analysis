import gensim, os
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np

def docLabel(doc) :
	texts = []
	for text in doc.split('_') :
		texts.extend(text.split('.'))
	return int(texts[1])


def genParaVec(model, dataType) :
	print(dataType + " paragraph vectors started...")

	dataParaVecX = []
	dataParaVecY = []

	for doc in os.listdir("aclImdb/" + dataType + "/pos") :
		dataParaVecY.append(docLabel(doc))
		dataParaVecX.append(model.docvecs[dataType + "_" + doc])

	for doc in os.listdir("aclImdb/" + dataType + "/neg") :
		dataParaVecY.append(docLabel(doc))
		dataParaVecX.append(model.docvecs[dataType + "_" + doc])

	dump_svmlight_file(np.asarray(dataParaVecX), np.asarray(dataParaVecY), "aclImdb/" + dataType + "/labeledParaVec.feat")

	print(dataType + " paragraph vectors done...")


def main() :
	d2v_model = gensim.models.doc2vec.Doc2Vec.load("Doc2Vec.model")
	print("Doc2Vec model loaded...")

	genParaVec(d2v_model, "train")
	genParaVec(d2v_model, "test")


if __name__ == "__main__" :
	main()
