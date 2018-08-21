from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np

def genBbow(dataBow) :
	dataBbowX = dataBow[0].sign()
	return dataBbowX, dataBow[1]


def genNormTf(dataBow) :
	sumDocs = dataBow[0].sum(1)
	dataNormTfX = dataBow[0].multiply(1 / sumDocs).tocsr()
	return dataNormTfX, dataBow[1]


def genTfIdf(dataBow, dataNormTfX) :
	numWordDoc = dataBow[0].sign().sum(0)
	idf = np.log(len(dataBow[1]) / numWordDoc)

	dataTfIdfX = dataNormTfX.multiply(idf).tocsr()
	return dataTfIdfX, dataBow[1]


def genAll3(inputFile) :
	dataBow = load_svmlight_file(inputFile + "labeledBow.feat")
	print(inputFile + " -- Loaded 'labeledBow.feat'...")

	dataBbowX, dataBbowY = genBbow(dataBow)
	dump_svmlight_file(dataBbowX, dataBbowY, inputFile + "labeledBbow.feat")
	print("Generated Labeled Binary Bag of Words data (Saved as labeledBbow.feat)...")

	dataNormTfX, dataNormTfY = genNormTf(dataBow)
	dump_svmlight_file(dataNormTfX, dataNormTfY, inputFile + "labeledNormTf.feat")
	print("Generated Labeled Normalized Tf data (Saved as labeledNormTf.feat)...")

	dataTfIdfX, dataTfIdfY = genTfIdf(dataBow, dataNormTfX)
	dump_svmlight_file(dataTfIdfX, dataTfIdfY, inputFile + "labeledTfIdf.feat")
	print("Generated Labeled TfIdf data (Saved as labeledTfIdf.feat)...")

def main() :
	inputFile = "aclImdb/train/"
	print("Generating BBoW, NormTf, TfIdf data for training dataset...")
	genAll3(inputFile)

	inputFile = "aclImdb/test/"
	print("Generating BBoW, NormTf, TfIdf data for test dataset...")
	genAll3(inputFile)


if __name__ == "__main__" :
	main()
