"""This module generates movie data from movie_ids.
Input:
    [movie_ids]
Output:
    [{"title":, "movie_id":, "tmdb_id":, "poster_url":, "rating":}]
"""

import pickle
import user_data


with open("movie_titles.bin", mode="rb") as file:
    movie_titles = pickle.load(file) # {movie_id: movie title}

with open("tmdb_data.bin", mode="rb") as file:
    tmdb_data = pickle.load(file) # {movie_id: [tmdb_id_str, poster_file_name]}



def get_movie_data(movie_ids, user_id = None):
    """
    :param movie_ids: a list of movie ids
    :param user_id: the rating for each movie is based on user_id
    :return: a list of dictionaries: [{"title":, "movie_id":,
        "tmdb_id":, "poster_url":, "rating":}]
    """
    if movie_ids is None: return []
    if len(movie_ids) == 0: return []

    user_ratings = {}
    if user_id is not None:
        user_ratings = user_data.get_ratings(user_id)

    return _get_movie_data(movie_ids, user_ratings)



def get_user_rated_movie_data(user_id):
    """Get data for movies that a user has rated.

    :param user_id: Return movie data for the movies rated by this
        user.
    :return: a list of dictionaries: [{"title":, "movie_id":,
        "tmdb_id":, "poster_url":, "rating":}]
    """
    user_ratings = user_data.get_ratings(user_id)
    if len(user_ratings) == 0: return []

    return _get_movie_data(user_ratings.keys(), user_ratings)



def _get_movie_data(movie_ids : list, user_ratings : dict):
    """
    :param movie_ids: list of movie ids
    :param user_ratings: {movie_id: rating} lookup data
    :return: a list of dictionaries: [{"title":, "movie_id":,
        "tmdb_id":, "poster_url":, "rating":}]
    """
    movie_data = []

    for movie_id in movie_ids:
        # retrieve "tmdb_id" and "poster_url" from "tmdb_data"
        tmdb_id = "None"
        poster_url = "None"

        if movie_id in tmdb_data:
            tmdb_id = tmdb_data[movie_id][0]
            poster_url = str(tmdb_data[movie_id][1])  # poster might be "None"

        # retrieve "rating" from "user_ratings"
        rating = 0
        if movie_id in user_ratings:
            rating = user_ratings[movie_id]

        movie_data.append({
            "title": movie_titles[movie_id],
            "movie_id": movie_id,
            "tmdb_id": tmdb_id,
            "poster_url": poster_url,
            "rating": rating
        })

    return movie_data



