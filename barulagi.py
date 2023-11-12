#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys. version


# In[5]:


get_ipython().system('pip install emoji')
get_ipython().system('pip install Sastrawi')
get_ipython().system('pip3 install swifter')


# In[6]:


pip install wheel setuptools pip --upgrade


# In[7]:


pip install gensim


# In[8]:


pip install textblob


# In[9]:


get_ipython().system('pip install -U deep-translator')


# In[10]:


pip install imblearn


# In[11]:


pip install emoji --upgrade


# In[12]:


get_ipython().system('pip install wordcloud')


# In[13]:


import nltk
from nltk.tokenize import word_tokenize


# UNCOMMENT "nltk.download()" untuk  pertamakali run, untuk run selanjutnya comment kembali
# nltk.download() 


# In[14]:


import pandas as pd 
import numpy as np
import re
import string
# import emoji

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

import swifter
from tqdm import tqdm 

# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download()

import ast 

import gensim
from gensim import corpora, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[15]:


from deep_translator import GoogleTranslator


# In[16]:


dfc = pd.read_csv("youtube-comments (5).csv");


# In[18]:


dfc['textDisplay'][0]


# In[20]:


dfc.info()


# In[19]:


dfc


# In[17]:


dfc.head()


# In[21]:


# hapus data nan
df_cleaned = dfc.dropna()


# In[22]:


for tweet in tqdm(df_cleaned['textDisplay']):
    print(tweet)
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    


# In[23]:


clean_texts = []
token_texts = []
freq_words = []

# stopword
stop_factory = StopWordRemoverFactory().get_stop_words() #load defaul stopword
more_stopword = ['•',"stengah", "kopit", "china","g","yaaaaa","gengs","gada","dgn",
"nich","yg","padan", "juoro","nya","js","kl","","co","ga","lg","gw","jg","walu",
"grrabbpoodd","klo","jeben","makane","kakean","sek","mb","skp","tpi","bgt","lgi",
"lu","rb","rban","mura","pd","nih","lii","enel","dr","exo","ipo","trus","d","shm",
"skrg","byk","mang","ots","dah""dg","bp","n","arsjadrasjid","mmc"] #menambahkan stopword
data_stopword = stop_factory + more_stopword #menggabungkan stopword

for tweet in tqdm(df_cleaned['textDisplay']):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = re.sub(r"\d+", "", tweet) # Remove number
#     tweet = ''.join(c for c in tweet if c not in emoji.EMOJI_DATA) #Remove Emojis
    tweet = tweet.replace('"','') #remove quotation mark
    tweet = tweet.lower() #Lower Case
    tweet = tweet.strip() # Remove Whitespace
    tweet = tweet.translate(str.maketrans("","",string.punctuation)) #Remove Punctuation 
 
    dictionary = ArrayDictionary(data_stopword)
    swr = StopWordRemover(dictionary)
    tweet = swr.remove(tweet)
    
    #Tokenization
    tokens = nltk.tokenize.word_tokenize(tweet)
    tweet = " ".join(tokens) #Disatukan Kembali
    
    freq_words.append(nltk.FreqDist(word_tokenize(tweet)))
    
    token_texts.append(tokens)
    clean_texts.append(tweet)
df_cleaned['clear'] = clean_texts
df_cleaned['token_texts'] = token_texts
df_cleaned['freq_words'] = freq_words

all_freq_words = nltk.FreqDist(sum(df_cleaned['clear'].map(word_tokenize), []))

df_cleaned


# In[24]:


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in df_cleaned['token_texts']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

df_cleaned['stemmed_texts'] = df_cleaned['token_texts'].swifter.apply(get_stemmed_term)
print(df_cleaned['stemmed_texts'])


# In[25]:


FreqWord = pd.DataFrame(all_freq_words.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)


# In[26]:


FreqWord


# In[ ]:


dfc.shape
dfc.isnull().sum()
dfc['clear'].isnull().sum()
df_final = dfc.drop(dfc[dfc.clear == ''].index)
dfc["clear"].duplicated().sum()
df_final.drop_duplicates(subset='clear',inplace=True)
df_final.shape


# In[27]:


df_final = df_cleaned.copy()


# In[28]:


df_cleaned.shape


# In[29]:


df_cleaned.isnull().sum()


# In[30]:


df_cleaned['clear'].isnull().sum()


# In[31]:


df_final = df_cleaned.drop(df_cleaned[df_cleaned.clear == ''].index)


# In[ ]:





# In[32]:


df_cleaned["clear"].duplicated().sum()


# In[33]:


df_final.drop_duplicates(subset='clear',inplace=True)


# In[34]:


df_cleaned.shape


# In[35]:


from textblob import TextBlob


# In[36]:


translate_texts = []
label_texts = []
polarity_text = []
subjectivity_texts = []
scores=[]
for comment in tqdm(df_final['clear']):
    blob = TextBlob(comment)

    # Menerjemahkan teks ke Bahasa Inggris
#     terjemahan_inggris = blob.translate(from_lang='id', to='en')
    
    terjemahan_inggris=GoogleTranslator(source='auto', target='en').translate(comment)
    terjemahan_inggris=TextBlob(terjemahan_inggris)
    
    # Melakukan analisis sentimen
    sentimen = terjemahan_inggris.sentiment
    
    # Mendapatkan nilai sentimen
    polarity = sentimen.polarity  # Nilai antara -1 hingga 1 (positif hingga negatif)
    subjectivity = sentimen.subjectivity  # Nilai antara 0 hingga 1 (faktual hingga subjektif)

    # Menampilkan hasil analisis sentimen
    if polarity > 0:
        sentiment_label = "Positif"
        score=1
    elif polarity < 0:
        sentiment_label = "Negatif"
        score=-1
    else:
        sentiment_label = "Netral"
        score=0

#     print("Teks:", terjemahan_inggris.string)
    scores.append(score)
    label_texts.append(sentiment_label)
    polarity_text.append(polarity)
    translate_texts.append(terjemahan_inggris.string)
    subjectivity_texts.append(subjectivity)
    


# In[37]:


df_final['label']=label_texts
df_final['score']=scores
df_final['translate']=translate_texts
df_final['polarity']=polarity_text
df_final['subjectivity']=subjectivity_texts
    


# In[38]:


df_final


# In[ ]:





# In[39]:


# fisualisasi

fig, ax = plt.subplots(figsize = (6, 6))
sizes = [count for count in df_final['label'].value_counts()]
labels = list(df_final['label'].value_counts().index)
explode = (0.1, 0, 0)
ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
ax.set_title('Sentiment Polarity on Tweets Data', fontsize = 16, pad = 20)
plt.show()


# In[40]:


pd.set_option('display.max_colwidth', 3000)
positive_comments = df_final[df_final['label'] == 'Positif']
positive_comments = positive_comments[['clear', 'polarity', 'label','translate']].sort_values(by = 'polarity', ascending=False).reset_index(drop = True)
positive_comments.index += 1
positive_comments[0:10]


# In[41]:


pd.set_option('display.max_colwidth', 3000)
positive_comments = df_final[df_final['label'] == 'Negatif']
positive_comments = positive_comments[['clear', 'polarity', 'label','translate']].sort_values(by = 'polarity', ascending=False).reset_index(drop = True)
positive_comments.index += 1
positive_comments[0:10]


# In[42]:


# Visualize word cloud

list_words=''
for tweet in df_final['token_texts']:
    for word in tweet:
        list_words += ' '+(word)

wordcloud = WordCloud(width = 600, height = 400, background_color = 'black', min_font_size = 10).generate(list_words)
fig, ax = plt.subplots(figsize = (8, 6))
ax.set_title('Word Cloud of Tweets Data', fontsize = 18)
ax.grid(False)
ax.imshow((wordcloud))
fig.tight_layout(pad=0)
ax.axis('off')
plt.show()


# In[43]:


tokens = nltk.tokenize.word_tokenize(list_words)
fdist=nltk.FreqDist(tokens)
fdist

