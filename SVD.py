import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import os
sns.set_style("darkgrid")

current_folder = os.path.dirname(os.path.realpath(__file__))

movies_path = os.path.join(current_folder, 'data/Original/movies.csv')
merged_path = os.path.join(current_folder, 'data/merged.csv')

mf = pd.read_csv(movies_path)
df = pd.read_csv(merged_path)

f = ['count','mean']

df_movie_summary = df.groupby('movieId')['rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

#print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('userId')['rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

#print('Customer minimum times of review: {}'.format(cust_benchmark))

#print('Original Shape: {}'.format(df.shape))
df = df[~df['movieId'].isin(drop_movie_list)]
df = df[~df['userId'].isin(drop_cust_list)]
#print('After Trim Shape: {}'.format(df.shape))

df_p = pd.pivot_table(df,values='rating',index='userId',columns='movieId')

#print(df_p.shape)

reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']][:], reader)
#data.split(n_folds=3)

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])


#what user 200 rated
df_200 = df[(df['userId'] == 120) & (df['rating'] == 5)]
#print(df_200[['title', 'rating']])



user_200 = mf.copy()
user_200 = user_200.reset_index()
user_200 = user_200[~user_200['movieId'].isin(drop_movie_list)]

# getting full dataset
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()
svd.fit(trainset)

user_200['Estimate_Score'] = user_200['movieId'].apply(lambda x: svd.predict(785314, x).est)

user_200 = user_200.drop('movieId', axis = 1)

user_200 = user_200.sort_values('Estimate_Score', ascending=False)
df = user_200.head(10)[['title', 'genres', 'Estimate_Score', ]]

print(df)

html_path = os.path.join(current_folder, 'index.html')

text_file = open(html_path, "w")
text_file.write(df.to_html())
text_file.close()
