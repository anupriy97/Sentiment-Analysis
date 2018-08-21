from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import os

corpus = ""

for filename in os.listdir("pos/") :
	corpus += open("pos/" + filename, 'r').read()
	corpus += "\n"

print("Positive reviews read...")

for filename in os.listdir("neg/") :
	corpus += open("neg/" + filename, 'r').read()
	corpus += "\n"

print("Negative reviews read...")

sentCorpus = sent_tokenize(corpus)

print("Sentence tokenization completed...")

for i in range(len(sentCorpus)) :
	sentCorpus[i] = word_tokenize(sentCorpus[i].lower())

print("Word tokenization completed...")

model = gensim.models.Word2Vec(sentCorpus, min_count = 1, size = 300, workers = 6)

print("Word2Vec training completed...")

model.save("imdb_train_model_word2vec")

print("Word2Vec model saved...")