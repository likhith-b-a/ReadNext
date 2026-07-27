#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("NLTK resources downloaded.")


# In[3]:


books_df = pd.read_csv("./Dataset/books.csv")
books_df.shape


# In[4]:


books_df = books_df.drop_duplicates(subset='book_title', keep='first')

# Keep only the top 14 categories that add the most value
top_categories = [
    'Fiction', 'Juvenile fiction', 'Biography & autobiography', 'Humor',
    'History', 'Religion', 'Body, mind & spirit', 'Social science',
    'Family & relationships', 'True crime', 'Self-help', 
    'Juvenile nonfiction', 'Business & economics', 'Health & fitness'
]
books_df = books_df[books_df['Category'].isin(top_categories)].reset_index(drop=True)
books_df.shape


# In[5]:


def advanced_text_preprocessing(text):
    """
    Apply advanced text preprocessing including:
    - Remove special characters and numbers
    - Convert to lowercase
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

books_df['processed_summary'] = books_df['Summary'].apply(advanced_text_preprocessing)


# In[6]:


books_df.head()


# In[7]:


# A constant sentinel token separates concatenated fields so that bigrams
# never span two different fields (e.g. last author word + first category word).
# It appears in nearly every document, so TF-IDF's IDF term drives its weight to ~0.
FIELD_SEP = ' xsepx '

books_df['weighted_content'] = (
    books_df['book_title'] + FIELD_SEP + books_df['book_title'] + FIELD_SEP +
    books_df['book_author'] + FIELD_SEP + books_df['book_author'] + FIELD_SEP +
    books_df['Category'] + FIELD_SEP +
    books_df['processed_summary']
)
books_df['weighted_content'][0]


# In[8]:


tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
tfidf_matrix = tfidf.fit_transform(books_df['weighted_content'])


# In[9]:


# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[10]:


indices = pd.Series(books_df.index, index=books_df['book_title']).drop_duplicates()


# In[11]:


with open('model.pkl', 'wb') as f:
    pickle.dump({
        'tfidf_vectorizer': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'indices': indices,
        'books_df': books_df
    }, f)


# In[ ]:




