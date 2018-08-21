import gensim
from scipy.sparse import find
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

def genAvgWord2Vec(inputFile, w2v_model, modelVocab, vocab) :
	dataBowX, dataBowY = load_svmlight_file(inputFile + "labeledBow.feat")
	print("Loaded 'labeledBow.feat'...")

	print("Averaging of word vectors started...")

	dataAvgWord2Vec = []

	for i in range(len(dataBowY)) :
		if (i % 500 == 0) and i :
			print(i, " reviews done...")

		_, cols, values = find(dataBowX[i])
		sumWv = 0
		countWv = 0
		for j in range(len(cols)) :
			word = vocab[cols[j]]
			if word in modelVocab :
				sumWv += w2v_model[word] * values[j]
				countWv += values[j]
		avgWv = sumWv / countWv
		dataAvgWord2Vec.append(avgWv)

	dump_svmlight_file(dataAvgWord2Vec, dataBowY, inputFile + "labeledAvgWord2Vec.feat")

	print("Averaging of word vectors completed (Saved as labeledAvgWord2Vec.feat)...")


def main() :
	w2v_model = gensim.models.Word2Vec.load('imdb_train_model_word2vec')
	modelVocab = list(w2v_model.wv.vocab)
	print("Word2Vec model loaded...")
	
	vocab = open("aclImdb/imdb.vocab", 'r').read()
	vocab = vocab.split()
	print("Vocabulary loaded...")

	print("Generating AvgWord2Vec for training dataset...")
	inputFile = "aclImdb/train/"
	genAvgWord2Vec(inputFile, w2v_model, modelVocab, vocab)

	print("Generating AvgWord2Vec for test dataset...")
	inputFile = "aclImdb/test/"
	genAvgWord2Vec(inputFile, w2v_model, modelVocab, vocab)


if __name__ == "__main__" :
	main()
