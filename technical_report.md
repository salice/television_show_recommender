# Technical Report

### Executive Summary
____

I used transcripts of television episodes obtained from [springfieldspringfield.co.uk
](https://www.springfieldspringfield.co.uk/) to create a content based recommender system using the Latent Dirichlet Allocation (LDA) model, which creates unsupervised clusters of similar television shows. I then used the output from the model to compute the cosine similarities and return the top five most similar shows to a given input.

#### The Data
____
##### Source of and Scraping the Data
The data for my project is 117,937 transcripts of television episodes from 4,667 different television shows. All of the transcripts, plus show name, episode name, season number, and episode number were obtained from a scrape of the site [springfieldspringfield.co.uk
](https://www.springfieldspringfield.co.uk/) built using BeautifulSoup. The scrape worked by first creating a list of all the pages containing television episodes through concatinating the base url with the page number, which ran from 1 to 281 at the time of my scrape. Then, using the list of television pages, I scraped each of those pages to create a list of every television show page, and used the list of every television page to create a list of every episode page. Since there were so many episodes to scrape, I then used Python's Pool class from the multiprocessing library in order to scrape the episodes in parallel and then combine them at the end. Adding multiprocessing to the scrape greatly improved the speed of the scrape and it went from taking 1 hour to scrape one page of television show to 10 minutes.

##### Cleaning the Data
The output from my scrape was a csv file containing 117,937 rows contianing three columns: the episode text, the episode name, and the show name. The show name column did not have a regular format, though, most of the entries were strings of the form "show name" + "(year)"+ "s" + season number + "e" + episode number. But there were exceptions to the pattern where the string did not contain the year or a season or episode number. In order to extract the show name for each show and not lose any of the other information contained within the strings, I created a function looking for patterns using regex to split up the strings first by whether or not it contained a year value, then splitting further to extract the season and episode number if there. Out of all 117,937 rows, the function successfully cleaned all but 7 of them! Those 7 rows all contained typos (such as mistyping an "a" instead of an "s" for season). Using the cleaned show name column I then created an aggregation function to combine all of the text by show into one large document and saved it as a csv file. I assumed that this would be the most efficient way to create cohesive documents for my model but I failed to realize that the text of, for example, every episode of *Doctor Who* contained within one document would prove difficult for spacy to tokenize and lemmatize. 

##### Tokenizing, Lemmatizing, Stop Words, Corpus, and Dictionary 
In order to tokenize all of my text (break my text into individual words), I used the simple_preprocess function in gensim which, in addition to tokenizing, also removes any punctuation in the text and makes all of the text lowercase. I then worked on creating a list of stopwords, common words that my model will exclude since they occur too frequently or have some strong predictive power. I used a combination of five sources to create my list of stop words: spacy's stop words list, nltk's English stop words, a list of common English words from[Stanford's CoreNLP research group](https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt),  the long English stopwords list from [ranks.nl](https://www.ranks.nl/stopwords), and a list of stop words I created manually. I created my list of stop words by running my model (discussed in the next section) on a random small subset of my data (because that was all my laptop could handle), looking at the words output that were strongly predictive and adding any that did not seem to have any predictive power to my stop words list. I continued this process and created a list that is 2,000 words long. Most of the words in my list of custom stop words are names that are unique to certain shows (such as Kramer for *Seinfeld*, Squidward for *Spongebob Squarepants*, etc.) and also verbs that are frequent but do not have any predictive information (such as said, played, yelled, etc).

After tokenizing and removing the stop words, the text is then lemmatized using spacy's large English word list. Lemmatizing is the process of stripping all each to its base form if it has one, such as removing the plural ending from a noun or changing the tense or conjugation of a verb, so that the frequency of each word can more accurately be counted. This step is the slowest out of all of my cleaning and modeling steps because the function has to go one word at a time, and even using a remote instance with 64 GBs of RAM it takes about 70 minutes. 

After lemmatizing the text I then build the dictionary and corpus for the text, where the dictionary attaches a unique id number to each word and the corpus then takes the id number and calculates the term document frequency of that word.

#### The Model
_____

##### Latent Dirichlet Allocation (LDA)
[Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158) is a way to find similarities between different documents by modeling each document as a set of words and calculating the similarites between documents. It takes a document (in this case the transcripts of all of the episodes of a given tv show) and determines which words used most frequently and have th strongest association with the text. The alogrithm compares the words from one document to words in the other documents and creates clusters based on whether the documents share similar predictive words. The relative frequencies of words are the only text feature used to create these clusters, this model treats text as a bag of words and ignores word order and part of speech. 

##### Running LDA with Gensim
Gensim has a built in LDA function that takes the corpus, dictionary, and number of clusters, as parameters and outputs a sparse matrix where each document has a probabiliity of belonging to one of the clusters, where the probabilites for each document then add up to 1. LDA also performs dimensionality reduction and each cluster is represented as a linear combination of the words that are most strongly associated with predicting that a document belongs to that cluster and its weight. Using the list of strongly associated words it is possible to hypothesize what each cluster may represent. The resulting matrix can then be used to calculate the cosine similarities using the MatrixSimilarity class in Gensim and turned into a recommender system.


##### Using Google Cloud Platform
Due to the amount of data I used to create this model, I could not run the model on my laptop and instead used a virtual machine I spun up using Google Cloud Platform. Setting up the machine was not intuitive at all and their documentation is not clear, but after some trial error, I finally installed the right packages on my local machine and updated and installed the necessary Python packages on the instance. I also turned my LDA Jupyter notebook into a .py file so that it could more easily run on the machine. I ran my code on a machine with 64 GBs of RAM and from creating the stop words list to running a basic model it took approximately 2 hours.

##### Tuning the Number of Clusters and Coherence Scores
The LDA algorithm has one hyperparameter to tune: the number of clusters. In order to determine the number of clusters I ran a series of models with different values for the number of clusters and compared their coherence scores. The [coherence score](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0) is a measure of how similar the most predictive words in each cluster are to each other, that ranges from 0 to 1, where 0 represents no similarity and 1 would be perfect similarity (if such a thing could exist). I used the $C_v$ coherence measure to calculate my coherence scores since it is calculated using cosine similarity, which will also be used later to calculate similarites between shows. I chose the model with the highest coherence score as my final model for my recommendation system.  

#### Recommender System and Results
_____
Results to come! The model is running (and no longer breaking!) now and will have results ASAP.

#### Next Steps
____





