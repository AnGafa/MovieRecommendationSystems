
'''
takes the user id and the number of movies to be recommended
as input and returning the top n
movies to be recommended to the user.

:return: The top 10 movies to be recommended to the user.
'''
import os
import warnings
import pandas as pd
from surprise import KNNBasic, Reader, Dataset, SVD, KNNWithMeans# pylint: disable=unused-import
from surprise import accuracy
from surprise.model_selection import train_test_split


warnings.simplefilter('ignore')

current_folder = os.path.dirname(os.path.realpath(__file__))
html_path = os.path.join(current_folder, 'index.html')


def create_datasets():
    '''
    It reads in the ratings and movies data
    :return: ratings_data and movie_names
    '''
    # Get the current folder of the file.
    current_folder = os.path.dirname(os.path.realpath(__file__))

    movies_path = os.path.join(current_folder, 'data/Original/movies.csv')
    ratings_path = os.path.join(current_folder, 'data/Original/ratings.csv')

    ratings_data = pd.read_csv(ratings_path)
    movie_names = pd.read_csv(movies_path)

    movietitlemap = movie_names.set_index('movieId')['title']
    ratings_data['title'] = ratings_data['movieId'].map(movietitlemap)

    movie_genere_map = movie_names.set_index('movieId')['genres']# pylint: disable=no-member
    ratings_data['genres'] = ratings_data['movieId'].map(movie_genere_map)



    return ratings_data, movie_names

def train_test_model(model, ratings_data):
    '''
    The function takes in a model and a dataframe of ratings data. It then splits the data into a
    training and test set, and fits the model to the training data. It then makes predictions on the
    test data and returns the model, a list of movies that were dropped,
    and the accuracy of the model.

    :param model: The model we want to train and test
    :param ratings_data: The dataframe containing the ratings data
    :return: The model, the list of movies that were dropped, and the accuracy of the model.
    '''

    data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], Reader())

    df_movie_summary = ratings_data.groupby('movieId')['rating'].agg(['count','mean'])
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    trainset, testset = train_test_split(data, test_size=0.3,random_state=10)

    trainset = data.build_full_trainset()

    model.fit(trainset)

    predictions = model.test(testset)

    model_accuracy = accuracy.rmse(predictions, verbose=True)
    return model, drop_movie_list, model_accuracy

def recommendation(model, movie_names, drop_movie_list, user_id, n_movies):
    '''
    takes the user id and the number of movies to be recommended as input and
    returning the top n movies to be recommended to the user.

    :param given_user_id: The user id for whom the recommendations are to be made
    :param n_movies: The number of movies to be recommended to the user
    :return: The top n movies to be recommended to the user.
    '''
    # Create a copy of the movie_names dataframe.
    given_user = movie_names.copy()
    # Reset the index of the dataframe.
    given_user = given_user.reset_index()
    # Removing the movies that have less than 70% of the ratings.
    given_user = given_user[~given_user['movieId'].isin(drop_movie_list)]

    # Predict the rating for each movie for the given user.
    given_user['Estimated_Rating'] = given_user['movieId'].apply(
        lambda x: model.predict(user_id, x).est)

    # RemovE the column `movieId` from the dataframe.
    given_user = given_user.drop('movieId', axis = 1)

    # Sort the dataframe by the estimated rating.
    given_user = given_user.sort_values('Estimated_Rating', ascending=False)
    # Remove the column `index` from the dataframe.
    given_user.drop(['index'], axis = 1,inplace=True)
    # Reset the index of the dataframe.
    given_user.reset_index(inplace=True,drop=True)

    # Return the top n movies to be recommended to the user.
    return given_user.head(n_movies)

def main(model, user_id, n_movies):
    """
    It takes in a model, user_id, and n_movies, and returns a list of n_movies movie names that the
    model recommends for the user_id

    :param model: The model you want to use
    :param user_id: The user ID of the user you want to recommend movies to
    :param n_movies: The number of movies you want to recommend
    """
    ratings_data, movie_names = create_datasets()

    # get movies seen by user
    movies_seen_by_user = ratings_data[ratings_data['userId'] == user_id]

    print(f'Top {n_movies} movies seen by user {user_id} are:')
    print('------------------------------------------------------')
    print(movies_seen_by_user.head(n_movies))
    print('------------------------------------------------------\n')

    print('------------------------------------------------------')
    model, drop_movie_list, model_accuracy = train_test_model(model, ratings_data)
    print(f'Accuracy of the model is {model_accuracy}')
    print('------------------------------------------------------\n')
    
    df = recommendation(model, movie_names, drop_movie_list, user_id, n_movies)
    print(f'Recommended {n_movies} movies for user {user_id} are:')
    print('------------------------------------------------------')
    print(df)
    print('------------------------------------------------------\n')

    text_file = open(html_path, "w")
    text_file.write(df.to_html())
    text_file.close()

if __name__ == '__main__':

    SELECTED_USER_ID = 120

    # Note that we need large k value for the model to work with reasonable accuracy.
    # main(KNNBasic(k=100, sim_options={'name': 'cosine', 'user_based': True}), 1, 10)
    # main(
    #     SVD(n_factors=10, n_epochs=10, lr_all=0.005, reg_all=0.02),
    #     user_id=SELECTED_USER_ID,
    #     n_movies=10)
    main(
        KNNWithMeans(k=10, sim_options={'name': 'cosine', 'user_based': True}),
        SELECTED_USER_ID,
        11)


