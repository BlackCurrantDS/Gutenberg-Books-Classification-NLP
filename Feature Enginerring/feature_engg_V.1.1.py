#!/usr/bin/env python
# coding: utf-8

# In[29]:



############################################################################
################## Importing Required Pacakges##############################
############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import re
import os
import nltk

""" This block is for EDA on the given data"""

####################Importing csv file with names and authors################

#Assuming File location is local "Need to change it"

#Getting data into dataframe

get_ipython().run_line_magic('time', '')
data = pd.read_csv("master996.csv", delimiter=';',encoding= 'unicode_escape')

#Checking memory usage in detail with distribution of data
data.info(memory_usage="deep")

#Checking the first 5 lines of the file

print(data.head(5))

#Checking the shape of the dataframe

data.shape

#Checking the count of the values in dataframe, should match wiht the file

data.count()

#Extracting the bookid into Dataframe column FileNo, book id pg10067- FileNo - 10067

data['FileNo'] = data['book_id'].str.replace(r'\D+', '').astype(int)

#Checkign the values

data.head(5)

#Checking Null Values

data.info() 

#There are no null values

#Getting list of gernes

print(data.guten_genre.value_counts().unique())

#Checking distribution of Gerne columns

data.guten_genre.describe()

#Getting distribution of authors

data.Author_Name.describe()

""" List is quite long"""

#Getting list of authors

# print(data.Author_Name.unique())


##################################### Trying some plots on the file ######################################

plt.figure(figsize=(18,5))

#See how much the Gernes are distributed
sns.countplot(data['guten_genre'])
plt.show()

"""Clearly Classes are imbalanced, Literarcy genre has higest values and Allegories almost have no books"""

#chekcing the actualy counts of each gerne

data.guten_genre.value_counts()


"""
Literary                       794
Detective and Mystery          111
Sea and Adventure               36
Love and Romance                18
Western Stories                 18
Humorous and Wit and Satire      6
Ghost and Horror                 6
Christmas Stories                5
Allegories                       2

"""

### Wordcloud to see most frequent author ###

stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(str(data['Author_Name']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

"""Looks like we have most fictions from Charles dickson and Stevenson"""


##################################  Here add more analysis and graphs for more insigths #####################
#############################################################################################################



# Add here number of books per author

# Add here number of gerne per author



################################################################################################################
##################################Importing html content into pnadas and joining it wiht meta data##############


# In[30]:


import glob

# Getting all files and its names into dict and then converting to dataframe"""

Content = {}

#Its placed in Books folder at local , need to change it later 
get_ipython().run_line_magic('timeit', '')
files = glob.glob("Books\*.html")
for f in files:
    with open(f,mode ='r', encoding = 'utf8') as myfile:
        Content[f]=myfile.read().replace("<br>", '\n')

#Can check for few files from below code        
"""for i in data:
    print(i, data[i])
"""
#Putting it in dataframe
df = pd.DataFrame.from_dict(Content,orient='index').reset_index()
#Renaming column names
df.columns = ['File_Name', 'Data']
#Adding FilNo column to join
df['FileNo'] = df['File_Name'].str.replace(r'\D+', '').astype(int)

df.info(memory_usage="deep")

left = df
right = data

#merging by matching the bookid, ignoring which files has not match
result = pd.merge(left, right , how='inner', on=['FileNo', 'FileNo'])

#Checking resulting dataframe
result.info()

#Dropping unnecessery columns

books=result.drop(['File_Name', 'book_id'], axis=1)
books.head(5)


# In[31]:


##################################### Pre-Processing of the html file content########################################

from sklearn import preprocessing
from bs4 import BeautifulSoup
import string

# labeling the classes
le = preprocessing.LabelEncoder()
books['guten_genre'] = le.fit_transform(books['guten_genre'])


#checking raw content
books['Data'].head(5)

#cleaning it with soup
books['Data']= [BeautifulSoup(text).get_text() for text in books['Data'] ]

#Make everything in lowe case
books['Data'] = books['Data'].apply(lambda x: x.lower())

#Remove puchtuation
books['Data'] = books['Data'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))

#Remove Digits
books['Data'] = books['Data'].apply(lambda x: x.translate(str.maketrans('','', string.digits)))
books['Data'].str.strip()

books['Data'].head(5)


# In[32]:


#######################################BAG OF WORDS##########################################################


# In[33]:


#import the necessary libraries
from sklearn.feature_extraction.text import CountVectorizer

# df_min= 0.and df_max = 1. the value could be its frequency in the document, 
#occurrence (denoted by 1 or 0) or even weighted values.

cv_vect = CountVectorizer(min_df=0., max_df=1.)

cv_matrix = cv_vect.fit_transform(books['Data'])

cv_matrix


# In[34]:


#To view thw result in the matrix form
cv_matrix = cv_matrix.toarray()
cv_matrix


# In[35]:


# get all unique words in the corpus

vocab1 = cv_vect.get_feature_names()

# show document feature vectors

pd.DataFrame(cv_matrix, columns=vocab1)


# In[36]:


#########################################BAG OF N-GRAM MODEL##########################################
#N gram model is implemented in order to get the words that occur in sequence. In case of BOW the order of
# the words is not considered.So inorder to extract phrases or collection of words which occur in a sequence we use N gram
#The below code depicts bigram model, n=2


# **FOR SIMPLICITY I AM USING ONLY 152 rows out of 996. When you are executing the code there is no need to change anything except the size used which has been indicated whereever it is**

# In[ ]:


# you can set the n-gram range to 1,2 to get unigrams as well as bigrams

bv = CountVectorizer(ngram_range=(1,2))

start = 0
while start < len(books['Data']):
    bv.fit(books['Data'][start:(start+20)]) #when executing all the 996 docs change the batch size to 300 or more according to your convinience
    start += 20 #same change is applied inorder to iterate
bv_matrix = bv.fit_transform(books['Data'])
bv_matrix = bv_matrix.toarray()

vocab2 = bv.get_feature_names()

pd.DataFrame(bv_matrix, columns = vocab2)

#bv = CountVectorizer(ngram_range=(1,2))

#bv_matrix = bv.fit_transform(books['Data'])

#bv_matrix = bv_matrix.toarray()

#vocab2 = bv.get_feature_names()

#pd.DataFrame(bv_matrix, columns=vocab2)


# In[37]:


#######################TF IDF MODEL###########################################
#tfidf(w, D) is the TF-IDF score for word w in document D. The term tf(w, D) represents the term frequency of the word w 
#in document D, which can be obtained from the Bag of Words model. 
#The term idf(w, D) is the inverse document frequency for the term w

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)

tv_matrix = tv.fit_transform(books['Data'])

print(tv.vocabulary_)
tv_matrix = tv_matrix.toarray()

vocab3 = tv.get_feature_names()

pd.DataFrame((tv_matrix), columns=vocab3)


# In[38]:


#cosine similarity and compare pairwise document similarity based on their TF-IDF feature vectors
#####################Document Similarity###############################################


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)

similarity_df = pd.DataFrame((similarity_matrix), columns = books['guten_genre'])

similarity_df


# In[40]:


##############################LDA MODEL#######################
#when LDA is applied the document term matrix gets decomposed into document term matrix and topic term matrix
####to get the document-topic matrix 


# In[41]:


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components = 3, max_iter=50, random_state=0, batch_size = 30) #change the max_iter to 1000 or more and also 
#Batch_size to 300 or more

dt_matrix = lda.fit_transform(tv_matrix)

features = pd.DataFrame(dt_matrix, columns= None)

features


# In[42]:


tt_matrix = lda.components_

for topic_weights in tt_matrix:

    topic = [(token, weight) for token, weight in zip(vocab3, topic_weights)]

    topic = sorted(topic, key=lambda x: -x[1])

    topic = [item for item in topic if item[1] > 0.6]

    print(topic)

    print()


# In[43]:


################################FEATURE SCALING##################################
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# # to handle values with varying magnitude 
# fs_f = data.iloc[:, 1:3].values 
# 
# print ("\nOriginal data values : \n",  fs) 
#   
#   
# 
# """"Handling the missing values """
#   
# from sklearn import preprocessing 
#   
# """ MIN MAX SCALER """
#   
# min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
#   
# # Scaled feature 
# fs_after_min_max_scaler = min_max_scaler.fit_transform(fs) 
#   
# print ("\nAfter min max Scaling : \n", fs_after_min_max_scaler) 
#   
#   
# """ Standardisation """
#   
# Standardisation = preprocessing.StandardScaler() 
#   
# # Scaled feature 
# fs_after_Standardisation = Standardisation.fit_transform(fs) 
#   
# print ("\nAfter Standardisation : \n", fs_after_Standardisation) 

# In[44]:


# Standardize Features
sc = StandardScaler(with_mean = True, with_std = True)

# Fit the scaler to the features and transform
fe_std = sc.fit_transform(tv_matrix)

# View the new feature data's shape    
print(); print(fe_std.shape)
print(); print(fe_std)


# In[45]:


# Create a pca object with the 3 components
pca = decomposition.PCA(n_components=3)

# Fit the PCA and transform the data
fe_std_pca = pca.fit_transform(fe_std)

# View the new feature data's shape
#print(); print(fe_std_pca.shape)
#print(); print(fe_std_pca)
principalDf = pd.DataFrame(data = fe_std_pca, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
principalDf


# In[46]:


finalDf = pd.concat([principalDf, books['guten_genre']], axis = 1)
finalDf


# In[47]:


finalDf = pd.concat([principalDf, books['Book_Name']], axis = 1)
finalDf


# In[48]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(fe_std_pca[:, 0], fe_std_pca[:, 1], fe_std_pca[:, 2], c=books['guten_genre'],
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# **Feature Selection**
# 
# **CHI SQUARE**
# 
# **MUTUAL INFORMATION**
# 
# when to use chi2?
# 
# When the data type of our feature to be tested and the target variable are both categorical (i.e. we have a classification problem) we can use Chi-Squared test.

# In[49]:


####################CHI SQUARE######################### Need of verification if the implementation is correct
from sklearn.feature_selection import chi2, SelectKBest
chi2score = chi2(tv_matrix, books['guten_genre'])
chi2score


# In[50]:


#####SELECT K BEST#######
kbest = SelectKBest(score_func = chi2, k = 30)
cv_kbest = kbest.fit_transform(tv_matrix, books['guten_genre']) #check the parameters passed is correct
dfscores = pd.DataFrame(cv_kbest)
dfcolumns = pd.DataFrame(columns = vocab3)
cv_kbest


# In[51]:


X_new = SelectKBest(chi2, k=20).fit_transform(tv_matrix, books['guten_genre'])
X_new.shape


# In[ ]:


############################Mutual Information#################
from sklearn.feature_selection import mutual_info_classif
res = dict(zip(vocab3, mutual_info_classif(tv_matrix, books['Data'], discrete_features=True)))
print(res)


# In[ ]:




