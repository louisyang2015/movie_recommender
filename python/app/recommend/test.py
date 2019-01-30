import json, requests


url = "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/recommend"


def get_recommendations(algorithm : str):
    data = { "user_id": -1, "algorithm": algorithm }
    data_str = json.dumps(data)

    r = requests.post(url, data=data_str)
    results = json.loads(r.text)

    model_params = json.loads(results["model_params"])
    for model_param in model_params:
        print(model_param)

    for result in results["movie_data"]:
        print(result["title"], result["movie_id"], result["tmdb_id"],
              result["poster_url"], result["rating"])


# get_recommendations("tag_ls")

