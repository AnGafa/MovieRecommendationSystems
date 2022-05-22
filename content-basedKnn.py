import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_folder = os.path.dirname(os.path.realpath(__file__))

movies_path = os.path.join(current_folder, 'data/Original/movies.csv')
ratings_path = os.path.join(current_folder, 'data/Original/ratings.csv')

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])

csr_sample = csr_matrix(sample)

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0], 'Genres':movies.iloc[idx]['genres'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        print(df)
        return df.to_html()
    else:
        return "No movies found. Please check your input"


df = get_movie_recommendation('Titanic')

html_path = os.path.join(current_folder, 'index.html')

text_file = open(html_path, "w")
text_file.write(df)
text_file.close()