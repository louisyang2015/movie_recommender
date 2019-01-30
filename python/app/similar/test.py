import json, requests


url = "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/similar"


def similar(movie_id : int):
    data = {
        "movie_id": movie_id,
        "user_id": -1
        }
    data_str = json.dumps(data)

    r = requests.post(url, data=data_str)
    results = json.loads(r.text)

    for result in results:
        print(result["title"], result["movie_id"], result["tmdb_id"],
              result["poster_url"], result["rating"])


# similar(1196)

