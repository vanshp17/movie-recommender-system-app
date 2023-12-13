import streamlit as st 
import pickle
import pandas as pd
import requests


def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=ed49e40dc9dc71f3c66585a41d55a6f0'.format(movie_id))
    data = response.json()
    return 'https://image.tmdb.org/t/p/w185/'+ data['poster_path']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    dist = similarity[movie_index]
    movies_list = sorted(list(enumerate(dist)),reverse=True,key=lambda x:x[1])[1:6]
    
    
    
    recommended_movies = []
    recommended_movie_posters = []
    
    for i in movies_list:
        
        movie_id = movies.iloc[i[0]].movie_id
        
       
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch the poster 
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movie_posters


movies = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies)

similarity = pickle.load(open('similarity.pkl','rb'))

st.title('Movie Recommender System')


selected_mov_name = st.selectbox(
    'How would you like to be contacted?',
    movies['title'].values)

if st.button('Recommend'):
    name,posters = recommend(selected_mov_name)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(name[0])
        st.image(posters[0])
    with col2:
        st.text(name[1])
        st.image(posters[1])
    with col3:
        st.text(name[2])
        st.image(posters[2])
    with col4:
        st.text(name[3])
        st.image(posters[3])
    with col5:
        st.text(name[4])
        st.image(posters[4])
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

