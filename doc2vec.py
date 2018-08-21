import gensim
import os

class LabeledLineSentence(object):
	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list
	def __iter__(self):
		for idx, doc in enumerate(self.doc_list):
			yield gensim.models.doc2vec.LabeledSentence(words = doc.split(),
				tags = [self.labels_list[idx]])

def load_data(dataType, data, docLabels) :
	docLabelsPos = os.listdir("aclImdb/" + dataType + "/pos/")

	for doc in docLabelsPos :
		data.append(open("aclImdb/" + dataType + "/pos/" + doc, 'r').read().lower())
		docLabels.append(dataType + "_" + doc)

	docLabelsNeg = os.listdir("aclImdb/" + dataType + "/neg/")

	for doc in docLabelsNeg :
		data.append(open("aclImdb/" + dataType + "/neg/" + doc, 'r').read().lower())
		docLabels.append(dataType + "_" + doc)

	print(dataType + " text data loaded...")


data = []
docLabels = []

# loading text from train data
load_data("train", data, docLabels)

# loading text from test data
load_data("test", data, docLabels)


# iterator returned over all documents
it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size = 150, min_count = 0, alpha = 0.025,
	min_alpha = 0.025)

model.build_vocab(it)

print("Model training starting...")

for epoch in range(50):
	print("iteration " + str(epoch+1))
	model.train(it, epochs = 1, total_examples = model.corpus_count)
	model.alpha -= 0.002
	model.min_alpha = model.alpha

model.save("Doc2Vec.model")

print("Doc2Vec model saved...")
