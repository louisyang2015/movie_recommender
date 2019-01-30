"""This API returns title search results.
Input:
    {"user_id":, "title":}
Output:
    [{"title":, "movie_id":, "tmdb_id":, "poster_url":, "rating":}]
"""

import json
import movie_data, title_search_index


indexer_config = title_search_index.IndexerConfig()
index = title_search_index.Index(indexer_config, "title_search_index.bin")


def lambda_handler(event, context):
    input_data = json.loads(event['body'])
    title = input_data["title"]
    user_id = int(input_data["user_id"])

    result = search(title, user_id)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        "isBase64Encoded": 'false',
        'body': json.dumps(result)
    }



def search(title : str, user_id : int):
    """
    :param title: movie title to search for
    :param user_id: user id to retrieve ratings for
    :return: a list of dictionaries, containing
        {"title": x, "movie_id": x, "tmdb_id": x, "poster_url": x,
        "rating": x}
    """
    if title is None: return []
    if len(title) == 0: return []

    search_results = index.search(title) # [(score, movie id)]
    search_results = search_results[:100]

    if len(search_results) == 0: return []

    _, movie_ids = zip(*search_results)
    movie_data_list = movie_data.get_movie_data(movie_ids, user_id)

    return movie_data_list



