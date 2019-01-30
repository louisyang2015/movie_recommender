import json
import movie_data


def lambda_handler(event, context):
    input_data = json.loads(event['body'])
    user_id = int(input_data["user_id"])

    result = get_rated_movies(user_id)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        "isBase64Encoded": 'false',
        'body': json.dumps(result)
    }



def get_rated_movies(user_id : int):
    """Returns movie data for movies rated by "user_id".

    :return: A list of {"title":, "movie_id":, "tmdb_id":,
        "poster_url":, "rating":}, sorted first by rating
        and then by title.
    """
    # get movie data
    data = movie_data.get_user_rated_movie_data(user_id)

    # data is a list where each element is {"title":, "movie_id":,
    #   "tmdb_id":, "poster_url":, "rating":}

    # The final result sorts according to rating first, then according
    # to title. Python produces a stable sort, so the first sort should
    # be by title. This "order by title" is then preserved when sorting
    # by rating.
    data.sort(key = lambda e: e["title"]) # sort by title
    data.sort(key = lambda e: e["rating"], reverse=True) # sort by rating

    return data






