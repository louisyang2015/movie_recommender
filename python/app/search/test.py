import json, requests


url = "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/search"


def search(title):
    data = {
        "title": "star trek 2",
        "user_id": -1
        }
    data_str = json.dumps(data)

    r = requests.post(url, data=data_str)
    results = json.loads(r.text)

    for result in results:
        print(result["title"], result["movie_id"], result["tmdb_id"],
              result["poster_url"], result["rating"])


