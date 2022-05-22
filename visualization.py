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

merged_path = os.path.join(current_folder, 'data/merged.csv')
#df_merged = pd.merge(movies, ratings, how='inner')
#df_merged.to_csv(merged_path)

df_merged = pd.read_csv(merged_path)

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)
#print(final_dataset.head())

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
# plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
# plt.axhline(y=10,color='r')
# plt.xlabel('MovieId')
# plt.ylabel('No. of users voted')
# plt.show()

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]

# f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
# plt.axhline(y=50,color='r')
# plt.xlabel('UserId')
# plt.ylabel('No. of votes by user')
# plt.show()

final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
print(final_dataset.info())

# f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
# plt.axhline(y=50,color='r')
# plt.xlabel('UserId')
# plt.ylabel('No. of votes by user')
# plt.show()

# f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
# plt.axhline(y=50,color='r')
# plt.xlabel('MovieId')
# plt.ylabel('No. of users voted')
# plt.show()


#sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
#sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
#print(sparsity)

#csr_sample = csr_matrix(sample)
#print(csr_sample)

# #visualizing genre
#merged = pd.read_csv(merged_path)
#genres_user = merged.groupby('userId')['genres'].agg('count')

#axs = merged.plot.area(figsize=(16,4),subplots=True,sharex=False)
#plt.show()

# f,ax = plt.subplots(1,1,figsize=(16,4))
# plt.scatter(genres_user.index,genres_user,color='mediumseagreen')
# plt.axhline(y=50,color='r')
# plt.xlabel('no of users')
# plt.ylabel('genres')
# plt.xticks()
# plt.show()