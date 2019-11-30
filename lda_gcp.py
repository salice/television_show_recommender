import numpy as np
import pandas as pd

import spacy
import re, pickle

import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import gensim
from gensim import corpora, similarities
from gensim.utils import simple_preprocess, lemmatize
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

#read in data
path_to_file = "show_text_combined.csv"
text_df = pd.read_csv(path_to_file)

#add stop words from spacy, nltk, Stanford, Ranks NL, and custom list
#combine into one set

nlp = spacy.load("en_core_web_lg")
nlp.remove_pipe('ner')
nlp.max_length = 93621305
#spacy_stop_words = nlp.Defaults.stop_words
#nltk_stop_words = stopwords.words("english")

#stop_words = spacy_stop_words.union(nltk_stop_words)

path_to_stanford_stop_words = "stanford_stopwords.txt"
stanford_file = open(path_to_stanford_stop_words, "r")
stanford_stopwords = stanford_file.read()
stanford_stopwords = stanford_stopwords.replace("\n", ",").split(",")
stanford_stopwords = set(stanford_stopwords)

stop_words = stanford_stopwords

path_to_nl_file = "ranks_nl_stopwords.txt"
nl_file = open(path_to_nl_file, "r")
ranks_nl_stopwords = nl_file.read()
ranks_nl_stopwords = ranks_nl_stopwords.replace("\n", ",").replace("\t", ",").split(",")[1:]
ranks_nl_stopwords = set(ranks_nl_stopwords)

stop_words = stop_words.union(ranks_nl_stopwords)

path_to_custom_file = "custom_stopwords.txt"
custom_file = open(path_to_custom_file, "r")
custom_stop_words = custom_file.read()
custom_stop_words = set(custom_stop_words.split(","))

stop_words = stop_words.union(custom_stop_words)
print("made it to stopwords")
#tokenize the text
#(setting deacc=True in order to remove punctuation)
def tokenizer(text):
    for word in text:
        yield(gensim.utils.simple_preprocess(str(word), deacc=True))
print("tokenizer worked")
#remove stopwords
def remove_stopwords(text):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in text]

text_df["text_tokenized"] = list(tokenizer(text_df["text"]))
print("text tokenized successfully")
text_df["text_no_stopwords"] = remove_stopwords(text_df["text_tokenized"])

#dictionary
id2word = corpora.Dictionary(text_df["text_no_stopwords"])

#the text
texts = text_df["text_no_stopwords"]

#term doc frequency (corpus)
corpus = [id2word.doc2bow(text) for text in texts]

print("made it to model")
#gensim lda model
gensim_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=30, 
                                           random_state=412,
                                           chunksize=200,
                                           passes=1,
                                           per_word_topics=True, update_every=20,alpha="auto")

print("model ran")

#function to caluclate coherence score for both LDA models
def coherence_score(model, data, dictionary):
	coherence_model_lda = CoherenceModel(model=model, 
		texts=data, 
		dictionary=dictionary, 
		coherence='c_v')

	coherence_lda = coherence_model_lda.get_coherence()
	return (str(model) + " LDA Coherence Score is: " + str(coherence_lda))

#gensim lda coherence score
#gensim_coherence_score = coherence_score(model=gensim_lda_model,
#							data=text_df["text_no_stopwords"],
#							dictionary=id2word)
#Mallet coherence score
#mallet_coherence_score = coherence_score(model=mallet_lda_model,
						#	texts=text_df["text_no_stopwords"],
						#	dictionary=dictionary)

#print(f"gensim coherence score is: {gensim_coherence_score}")
#print(f"mallet coherence score is: {mallet_coherence_score}")
#Topics for Gensim LDA and Mallet
gensim_topics = gensim_lda_model.print_topics()
#mallet_topics = mallet_lda_model.print_topics()



#pickle models
pickle.dump(gensim_lda_model, open("gensim_lda_model.p", "wb"))
#pickle.dump(mallet_lda_model, open("mallet_lda_model.p", "wb"))






