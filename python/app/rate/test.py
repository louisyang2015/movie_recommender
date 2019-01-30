import json, requests

def add_rating(movie_id, rating):
    url = "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/rate"
    data = {"user_id": -1, "op": "add_rating", "movie_id": movie_id, "rating": rating}
    data_str = json.dumps(data)

    r = requests.post(url, data=data_str)
    print(r.text)
    return r


def remove_rating(movie_id):
    url = "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/rate"
    data = {"user_id": -1, "op": "remove_rating", "movie_id": movie_id}
    data_str = json.dumps(data)

    r = requests.post(url, data=data_str)
    print(r.text)


