import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_folder = os.path.dirname(os.path.realpath(__file__))

merged_path = os.path.join(current_folder, 'data/merged.csv')
titles_path = os.path.join(current_folder, 'data/Original/movies.csv')

data = pd.read_csv(merged_path)
titles = pd.read_csv(titles_path)

reviews = data.groupby('title')['rating'].agg(['count','mean']).reset_index().round(1)

movies = pd.crosstab(data['userId'], data['title'], values = data['rating'], aggfunc = 'sum')

userInput = ["Titanic (1997)"]

movies.corrwith(movies[userInput[0]], method = 'pearson')
#similarity = movies.corrwith(movies[userInput[0]], method = 'pearson') + movies.corrwith(movies[userInput[1]], method = 'pearson') + movies.corrwith(movies[userInput[2]], method = 'pearson')
similarity = movies.corrwith(movies[userInput[0]], method = 'pearson')

correlatedMovies = pd.DataFrame(similarity, columns = ['correlation'])
correlatedMovies = pd.merge(correlatedMovies, reviews, on = 'title')
correlatedMovies = pd.merge(correlatedMovies, titles, on = 'title')

output = correlatedMovies[(correlatedMovies['mean'] > 3.5) & (correlatedMovies['count'] >= 50)].sort_values('correlation', ascending = False)

#output = output[((output.title != userInput[0]) & (output.title != userInput[1]) & (output.title != userInput[2]))]
output = output[((output.title != userInput[0]))]
output = output[:10]
output = output[["title", 'genres', 'correlation']]

print(output)

html_path = os.path.join(current_folder, 'index.html')

text_file = open(html_path, "w")
text_file.write(output.to_html())
text_file.close()
