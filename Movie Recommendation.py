#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')


# In[3]:


credits_df.head()


# In[4]:


movies_df.head()


# In[5]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[6]:


credits_df


# In[7]:


movies_df


# In[10]:


movies_df = movies_df.merge(credits_df, on = 'title')


# In[11]:


movies_df.shape


# In[12]:


movies_df.head()


# In[13]:


movies_df.info()


# In[14]:


# We will recommnend the movie using 7 columns only
movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[15]:


movies_df.head()


# In[16]:


movies_df.info()


# In[17]:


movies_df.isnull().sum()


# In[18]:


movies_df.dropna(inplace = True)


# In[19]:


movies_df.duplicated().sum()


# In[20]:


movies_df.iloc[0].genres


# In[21]:


import ast


# In[22]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[23]:


movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df


# In[24]:


def convert3(obj):
    l=[]
    c=0
    for i in ast.literal_eval(obj):
        if c != 3:
            l.append(i['name'])
            c = c + 1
        else:
            break
        return l


# In[25]:


movies_df['cast'] = movies_df['cast'].apply(convert3)


# In[26]:


movies_df.head()


# In[27]:


movies_df['overview'][0]


# In[28]:


movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())


# In[29]:


movies_df.head()


# In[30]:


movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x]if isinstance(x, list) else x)
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[31]:


movies_df.head()


# In[32]:


movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']


# In[33]:


movies_df.head()


# In[34]:


new = movies_df[['movie_id','title','tags']]


# In[35]:


new.head()


# In[36]:


new['tags'] = new['tags'].apply(lambda x:' '.join(x)if isinstance(x, list) else x)


# In[37]:


new.head()


# In[38]:


new['tags'][0]


# In[39]:


new['tags'] = new['tags'].apply(lambda x:x.lower()if isinstance(x, list) else x)


# In[40]:


new.head()


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
new['tags'] = new['tags'].fillna('')
cv  = CountVectorizer(max_features=5000, stop_words='english')


# In[42]:


cv.fit_transform(new['tags']).toarray().shape
#print(transformed.shape)


# In[43]:


vectors = cv.fit_transform(new['tags']).toarray()


# In[44]:


vectors[0]


# In[48]:


import nltk


# In[49]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[50]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[51]:


new['tags'] = new['tags'].apply(stem)


# In[52]:


from sklearn.metrics.pairwise import cosine_similarity


# In[53]:


cosine_similarity(vectors)


# In[54]:


cosine_similarity(vectors).shape


# In[55]:


similarity = cosine_similarity(vectors)


# In[56]:


similarity[0]


# In[57]:


similarity[0].shape


# In[58]:


sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])[1:6]


# In[59]:


def recommend(movie):
    movie_index = new[new['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new.iloc[i[0]].title)


# In[60]:


recommend('Avatar')


# In[61]:


recommend('Iron Man')


# In[62]:


recommend('Liar Liar')


# In[ ]:




