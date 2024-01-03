# Movie-Recommender-System

## Software and Tools Requirements

1. [Github Account](https://github.com)
2. [VScodeIDE](https://code.visualstudio.com)
3. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)


Create a new Environment

---
conda create -p venv python==3.11 -y
---

## Introduction
This project focuses on building a movie recommendation system using Natural Language Processing (NLP) techniques and cosine similarity. The dataset used includes information about movies such as titles, overviews, genres, keywords, cast, and crew.

## Features
- Importing and preprocessing the dataset
- Exploratory Data Analysis (EDA) to understand the data structure
- Extracting important features like genres, keywords, cast, and crew
- Text processing, including word cloud visualization and sentiment analysis
- Creating a movie tagging system based on selected features
- Building a recommendation system using cosine similarity

## Getting Started

### Import Libraries
Ensure you have the necessary libraries installed by running the following command:

```python
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
```

### Import Dataset
Load the movie dataset from two CSV files: 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv'.

```python
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```

## Usage

### Exploratory Data Analysis (EDA)
Perform an exploratory data analysis to understand the dataset structure.

```python
movies.head(1)
credits.head(1)
mov = pd.merge(movies, credits, on='title')
mov.info()
```

### Extracting Important Features
Extract key features such as movie_id, title, overview, genres, keywords, cast, and crew.

```python
mov = mov[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
mov.duplicated().sum()
mov = mov.drop_duplicates()
mov.isnull().sum()
mov.dropna(inplace=True)
```

### Text Processing and Tag Creation
Perform text processing on movie overviews and create a tagging system.

```python
mov['genres'] = mov['genres'].apply(convert)
mov['keywords'] = mov['keywords'].apply(convert)
mov['cast'] = mov['cast'].apply(convert3)
mov['crew'] = mov['crew'].apply(fetch_director)
mov['overview'] = mov['overview'].apply(lambda x: x.split())
mov['genres'] = mov['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
mov['keywords'] = mov['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])
mov['cast'] = mov['cast'].apply(lambda x: [i.replace(' ', '') for i in x])
mov['crew'] = mov['crew'].apply(lambda x: [i.replace(' ', '') for i in x])

mov['tags'] = mov['overview'] + mov['genres'] + mov['keywords'] + mov['cast'] + mov['crew']
new_df = mov[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

### Stemming and Vectorization
Apply stemming to reduce words to their root form and perform vectorization using CountVectorizer.

```python
new_df['tags'] = new_df['tags'].apply(stem)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

### Recommendation System
Build a movie recommendation system based on cosine similarity.

```python
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    dist = similarity[movie_index]
    movies_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

### Saving Models
Save the processed dataframe and models for later use.

```python
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(new_df.to_dict, open('movies_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

## Results
The recommendation system can suggest movies similar to the user's input based on the tagged features, offering a personalized movie recommendation experience.

## Contributing
Contributions to enhance the recommendation system or explore additional features are welcome. Fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- Thanks to Kaggle for providing the movie datasets.
- Acknowledgments to the NLTK and scikit-learn communities for their valuable libraries.
