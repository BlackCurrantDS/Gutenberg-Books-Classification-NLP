#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################word2vec#################################


# In[2]:


import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gensim
get_ipython().run_line_magic('pylab', 'inline')


# In[4]:


# Create a list of all of our book files.
book_filenames = sorted(glob.glob("Books\*.html"))
print("Found books:")
book_filenames


# In[5]:


# Read and add the text of each book to corpus_raw.
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[6]:


# Tokenize each sentence
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)


# In[7]:


def sentence_to_wordlist(raw):
    '''Remove all characters except letters'''
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[8]:


# Clean the raw_sentences and add them to sentences.
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[10]:


# Take a look at a sentence before and after it is cleaned.
print(raw_sentences[25])
print(sentence_to_wordlist(raw_sentences[25]))


# In[11]:


# Find the total number of tokens in sentences
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# In[12]:


#######################################word2vec########################################


# In[13]:


# Set the parameteres for Word2Vec
num_features = 300
min_word_count = 20
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-4
seed = 2


# In[14]:


books2vec = w2v.Word2Vec(
    sg=1, #skip-gram
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[15]:


# Build the vocabulary
books2vec.build_vocab(sentences)
print("books2vec vocabulary length:", len(books2vec.wv.vocab))


# In[20]:


total_examples=books2vec.corpus_count
total_examples


# In[21]:


# Create a vector matrix of all the words
all_word_vectors_matrix = books2vec.wv.syn0


# In[22]:


# Use TSNE to reduce all_word_vectors_matrix to 2 dimensions. 
tsne = sklearn.manifold.TSNE(n_components = 2, 
                             early_exaggeration = 6,
                             learning_rate = 500,
                             n_iter = 2000,
                             random_state = 2)


# In[23]:


all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[24]:


# Create a dataframe to record each word and its coordinates.
points = pd.DataFrame(
    [(word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[books2vec.wv.vocab[word].index])
            for word in books2vec.wv.vocab
        ]],
    columns=["word", "x", "y"])


# In[25]:


# Preview the points
points[100:105]


# In[26]:


# Display the layout of all of the points.
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(10, 6))


# In[27]:


def plot_region(x_bounds, y_bounds):
    '''Plot a limited region with points annotated by the word they represent.'''
    slice = points[(x_bounds[0] <= points.x) & (points.x <= x_bounds[1]) & 
                   (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 6))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[28]:


# Find the coordinates for Alice - Alice's Adventures in Wonderland
points[points.word == 'Alice']


# In[29]:


plot_region(x_bounds=(0.5, 1.5), y_bounds=(3.2, 4.2))


# In[30]:


# Find the coordinates for (Tom) Sawyer - The Adventures of Tom Sawyer
points[points.word == 'Sawyer']


# In[31]:


plot_region(x_bounds=(-4.5, -3.5), y_bounds=(1.0, 2.0))


# In[32]:


books2vec.most_similar("monster") 


# In[34]:


books2vec.most_similar("dog")


# In[ ]:




