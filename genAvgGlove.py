import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from scipy.sparse import find
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import os

def genAvgGlove(inputFile, glove_model, modelVocab, vocab) :
	dataBowX, dataBowY = load_svmlight_file(inputFile + "labeledBow.feat")
	print("Loaded 'labeledBow.feat'...")

	print("Averaging of Glove word vectors started...")

	dataAvgGlove = []

	for i in range(len(dataBowY)) :
		if (i % 500 == 0) and i :
			print(i, " reviews done...")

		_, cols, values = find(dataBowX[i])
		sumWv = countWv = 0
		for j in range(len(cols)) :
			word = vocab[cols[j]]
			if word in modelVocab :
				sumWv += glove_model[word] * values[j]
				countWv += values[j]
		avgWv = sumWv / countWv
		dataAvgGlove.append(avgWv)

	dump_svmlight_file(dataAvgGlove, dataBowY, inputFile + "labeledAvgGlove.feat")

	print("Averaging of Glove word vectors completed (Saved as labeledAvgGlove.feat)...")


def main() :
	if not os.path.isfile("gensim_glove_vectors_100d.txt") :
		print("Converting Glove input file to gensim Word2Vec file format...")
		glove2word2vec(glove_input_file = "glove.6B.100d.txt", word2vec_output_file = "gensim_glove_vectors_100d.txt")

	glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors_100d.txt", binary = False)
	modelVocab = list(glove_model.wv.vocab)
	print("Glove Word2Vec model loaded...")

	vocab = open("aclImdb/imdb.vocab", 'r').read()
	vocab = vocab.split()
	print("Vocabulary loaded...")

	print("Generating AvgGlove for training dataset...")
	inputFile = "aclImdb/train/"
	genAvgGlove(inputFile, glove_model, modelVocab, vocab)

	print("Generating AvgGlove for test dataset...")
	inputFile = "aclImdb/test/"
	genAvgGlove(inputFile, glove_model, modelVocab, vocab)


if __name__ == "__main__" :
	main()
