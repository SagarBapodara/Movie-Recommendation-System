# -*- coding: utf-8 -*-
"""Movie_Recommender_System.ipynb

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""## Loading Data"""

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies.head(5)

movies.info()

credits.info()

credits.head(5)

"""## Merging both dataframes : Movies & Credits"""

movies = movies.merge(credits,on='title')

movies.shape

movies.head(1)

"""## Data Pre-Processing"""

#> important columns to be used in recommendation system : 

# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id','title','overview','genres','cast','keywords','crew']]

movies.head(5)

movies.isnull().sum()

movies.dropna(inplace=True)

movies.isnull().sum()

movies.duplicated().sum()

movies.iloc[0].genres

import ast

# extracting genres from raw data for the creation of tags

def convert(obj):
  L = []
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

movies.head(5)

#function for extracting top 3 actors from the movie 

def convert3(obj):
  L = []
  counter = 0
  for i in ast.literal_eval(obj):
    if counter !=3:
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

movies['cast'] = movies['cast'].apply(convert3)

movies.head(5)

#function to fetch the director of movie from crew column
def fetch_director(obj):
  L = []
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      L.append(i['name'])
      break
  return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies.head(5)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

# applying a transformation to remove spaces between words 

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df.head()

new_df['tags'][0]

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df['tags'][0]

"""## Text Vectorization"""

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

## Most frequent 5000 words
# cv.get_feature_names()

"""## Applying Stemming Process"""

import nltk #for stemming process

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#defining the stemming function
def stem(text):
  y=[]

  for i in text.split():
      y.append(ps.stem(i))

  return " ".join(y)

stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')

new_df['tags'] = new_df['tags'].apply(stem)

"""## Similarity Measures"""

# For calculating similarity, the cosine distance between different vectors will be used.

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

"""## Making the recommendation function"""

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]

  for i in movies_list:
    print(new_df.iloc[i[0]].title)

"""## Recommendation"""

recommend('Batman Begins')  #enter movies only which are in the dataset, otherwise it would result in error

new_df.iloc[1216]

"""## Exporting the Model"""

import pickle

pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))