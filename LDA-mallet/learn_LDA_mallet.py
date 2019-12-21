import os,sys,re
import json
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from preprocess_text import *
from compute_optimal_k import *

NUM_TOPICS = 50
db_dir  = '../data/ImageCLEF_Wikipedia/'
train_dict_path = 'train_dict_ImageCLEF_Wikipedia.json'
img_dir = db_dir+'images/'
xml_dir = db_dir+'metadata/'

with open(train_dict_path) as f:
    train_dict = json.load(f)

limit = 500
data = []
for text_path in train_dict.values():
    if limit==0:
       break
    with open(db_dir+text_path) as f: raw = f.read()
    #print("\n\ncurrent article data is: ", raw)
    data.append(raw)
    print('\rCreating a list of contents of all documents: %d/%d documents processed...' % (len(data),len(train_dict.values())))
    #sys.stdout.flush()
    limit-=1
print(' Done!\n')

#print("the list of articles is: ",data)

## create a list of tokens of all documents
data_words = list(sent_to_words(data))
#print("\n\ntokens created are: ", data_words)


## remove stopwords
data_words_nostops = remove_stopwords(data_words)
#print("\nafter removing stop words, the data looks like: ", data_words_nostops)

print("\n", len(data_words[0]),"  ", len(data_words_nostops[0]))


## create trigram model
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100) 
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
#print("\n\ntrigram model -------------\n\n", trigram_mod[bigram_mod[data_words[0]]])

# form trigrams
data_words_trigrams = make_trigrams(trigram_mod, bigram_mod, data_words_nostops)
#print("\n\ntrigram words formed are ---------\n\n", data_words_trigrams)
print("\ntrigrams formed\n\n")

# initialize spacy 'en' model
nlp = spacy.load('en', disable=['parser', 'ner'])

# do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(nlp, data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#print("\n\nlemmatized data----- \n\n", data_lemmatized[:1])
#print("\nsize of lemmatized data: ",len(data_lemmatized)) 
print("\nlemmatization is done\n\n")

## Create the Dictionary and Corpus needed for Topic Modeling
id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]



## Building the Topic Model

# first download the file: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip

mallet_path = 'mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=NUM_TOPICS, id2word=id2word)


#show topics
print("topics are :\n" )
pprint(ldamallet.show_topics(formatted=False))


ldamallet.save('models/ldamodel_'+str(NUM_TOPICS)+'.lda')
print(' Done! model saved in .lda file')


# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score for ',NUM_TOPICS, '=', coherence_ldamallet)
"""


#compute optimal value for no. of topics
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

# Show graph

limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
plt.savefig('graph.png')

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))



# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))
"""
