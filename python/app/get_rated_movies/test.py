import json, requests


url = "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/get-rated-movies"


def get_rated_movies(movie_id : int):
    data = { "user_id": -1 }
    data_str = json.dumps(data)

    r = requests.post(url, data=data_str)
    results = json.loads(r.text)

    for result in results:
        print(result["rating"], result["title"], result["movie_id"],
              result["tmdb_id"], result["poster_url"])


# get_rated_movies(-1)


