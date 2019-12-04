import numpy as np
import pandas as pd

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pickle, time, json

import nltk
from nltk.corpus import stopwords

import gensim
from gensim import corpora, similarities
from gensim.utils import simple_preprocess, lemmatize
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

#This code creates a list of stopwords, tokenizes, removes the stopwords, lemmatizes, 
#creates a dictionary and corpus, runs the code through LDA, tests multiple values for the 
#number of clusters, outputs the scores and number of clusters as a json,
#chooses the value with the largest coherence score as the final model,
#runs that as the final model, outputs the model as a pickle and saves the topic list as a json
#This code is too large to run on a local machine it needs at least 50 GBs of RAM to work

start = time.time()
#read in data
path_to_file = "show_text_combined.csv"
text_df = pd.read_csv(path_to_file)

#add stop words from spacy, nltk, Stanford, Ranks NL, and custom list
#combine into one set

nlp = spacy.load("en_core_web_lg")
nlp.remove_pipe('ner')
nlp.max_length = 93621305
spacy_stop_words = STOP_WORDS
nltk_stop_words = "english"


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

stop_words = stop_words.union(nltk_stop_words)
stop_words = stop_words.union(spacy_stop_words)

print("made it to stopwords")

#tokenize the text
#(setting deacc=True in order to remove punctuation)
def tokenizer(text):
    for word in text:
        yield(gensim.utils.simple_preprocess(str(word), deacc=True))

#remove stopwords
def remove_stopwords(text):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in text]

#then lemmatize
def lemmatization(text):
    text_out = []
    for sent in text:
        doc = nlp(" ".join(sent), disable=["ner", "parser"]) 
        text_out.append([token.lemma_ for token in doc])
    return text_out

text_df["text_tokenized"] = list(tokenizer(text_df["text"]))

#checkpoint to make sure the code is running
print("text tokenized successfully")
elapsed_time = round((time.time()- start) / 60, 2)
print(elapsed_time)

text_df["text_no_stopwords"] = remove_stopwords(text_df["text_tokenized"])

text_df["text_lemmatized"] = lemmatization(text_df["text_no_stopwords"])

print("text lemmatized!")

elapsed_time = round((time.time()- start) / 60, 2)
print(elapsed_time)

#outputting dataframe for later use 
text_df.to_csv("text_lemmatized.csv", index=False)
print("CSV with lemmatized text output")

#the text
texts = text_df["text_lemmatized"]

#dictionary
id2word = corpora.Dictionary(texts)


#term doc frequency (corpus)
corpus = [id2word.doc2bow(text) for text in texts]

print("corpus and dictionary made")

#gensim lda model with untuned number of topics
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=id2word,
                                          num_topics=500, 
                                          random_state=412,
                                          chunksize=200,
                                          passes=1,
                                          per_word_topics=True, 
                                          update_every=20,
                                          alpha="auto")

print("gensim model ran")

#function to caluclate coherence score for LDA model
def coherence_score(model, texts, dictionary):
	coherence_model_lda = CoherenceModel(model=model, 
		texts=texts, 
		dictionary=dictionary, 
		coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	return coherence_lda

#gensim lda coherence score for untuned model
#gensim_coherence_score = coherence_score(model=gensim_lda_model,
#							texts=texts,
#							dictionary=id2word)
#print("coherence score worked")
#print(f"gensim coherence score is: {gensim_coherence_score}")

#checkpoint
print("testing tuning")

#test different number of topics with goal of maximizing coherence score
def coherence_score_topic_number_tuning(corpus, texts, dictionary, start, end, step_size):	
	k_vals = []
	scores = []
	for k in range(start, end, step_size):
		model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=k, 
                                        random_state=412,
                                        chunksize=200,
                                        passes=1,
                                        per_word_topics=True, 
                                        update_every=20,
                                        alpha="auto")
		score = coherence_score(model=model, texts=texts, dictionary=id2word)
		k_vals.append(k)
		scores.append(score)
		print("loop worked!")
	return dict(zip(k_vals, scores))


scores = coherence_score_topic_number_tuning(corpus=corpus, 
											texts=texts, 
											dictionary=id2word, 
											start=20, 
											end=501, 
											step_size=20)

#save scores dictionary to graph locally
f = open("num_topics_scores.json", "w")
f.write(json.dumps(scores))
f.close()

#running model with highest coherence score
best_k = max(scores, key=scores.get)

tuned_lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=best_k, 
                                           random_state=412,
                                           chunksize=200,
                                           passes=1,
                                           per_word_topics=True, 
                                           update_every=20,
                                           alpha="auto")

#Topics for Gensim LDA
gensim_topics = tuned_lda.print_topics()
print(gensim_topics)
#save topics to json
g = open("topic_list.json", "w")
g.write(json.dumps(gensim_topics))
g.close()

#pickle model for recommender system
pickle.dump(tuned_lda, open("tuned_lda_model.p", "wb"))

elapsed_time = round((time.time()- start) / 60, 2)

print(f"Done! It took {elapsed_time} minutes")




