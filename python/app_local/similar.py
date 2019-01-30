"""This API returns similar movies.
Input:
    {"user_id":, "movie_id":}
Output:
    [{"title":, "movie_id":, "tmdb_id":, "poster_url":, "rating":}]
"""

import json, pickle
import movie_data


# "similar_movies"
with open("similar_movies.bin", mode="rb") as file:
    similar_movies = pickle.load(file) # {movie_id: [similar_movie_ids]}


def lambda_handler(event, context):
    input_data = json.loads(event['body'])
    movie_id = int(input_data["movie_id"])
    user_id = int(input_data["user_id"])

    result = similar(movie_id, user_id)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        "isBase64Encoded": 'false',
        'body': json.dumps(result)
    }


def similar(movie_id : int, user_id : int):
    """
    :param movie_id: will return movies similar to "movie_id"
    :param user_id: user id to retrieve ratings for
    :return: a list of a dictionaries: {"title":, "movie_id":,
        "tmdb_id":, "poster_url":, "rating":}
    """
    if movie_id in similar_movies:
        # load "similar_movie_ids" from "similar_movies"
        similar_movie_ids = similar_movies[movie_id]

        movie_data_list = movie_data.get_movie_data(similar_movie_ids,
                                                    user_id)
        return movie_data_list

    else:
        return []


