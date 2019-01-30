import json, pickle
import movie_data, user_data
from models import TagCount_UserProfile, TagLS_UserProfile, ALS_Model


# globals that are loaded from disk as needed
movie_medians = None
movie_genres = None
movie_tags = None
genre_counts = None
genre_ids = None
tag_counts = None
tag_ids = None

als3_item_factors = None
als3_movie_ids = None
als5_item_factors = None
als5_movie_ids = None
als7_item_factors = None
als7_movie_ids = None
als9_item_factors = None
als9_movie_ids = None
als11_item_factors = None
als11_movie_ids = None


def lambda_handler(event, context):
    input_data = json.loads(event['body'])
    user_id = int(input_data["user_id"])
    algorithm = input_data["algorithm"]

    try:
        movie_ids, model_params = get_recommendations(user_id, algorithm)
        result_data = movie_data.get_movie_data(movie_ids)
        result = {"error": "None", "movie_data": result_data,
                  "model_params": model_params}
    except Exception as ex:
        result = {"error": str(ex)}

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        "isBase64Encoded": 'false',
        'body': json.dumps(result)
    }


def load_global_var_as_needed(var_name : str, file_name : str):
    """The "global_var" refers to "movie_medians",
    "movie_genres", and so on, that are set to None in the
    beginning."""
    if globals()[var_name] is None:
        with open(file_name, mode="rb") as file:
            globals()[var_name] = pickle.load(file)



def get_recommendations(user_id : int, algorithm : str,
                        num_results=user_data.rotation_size * 100):
    """Returns (list of movie ids), (string of list of model parameters)."""
    if user_data.is_new_recommendation_needed(user_id, algorithm):
        # check the number of user ratings
        user_ratings_dict = user_data.get_ratings(-1)
        user_ratings = list(user_ratings_dict.items())

        if len(user_ratings) < 1:
            raise Exception("No movie has been rated yet.")

        # all current models should have at least 3 reviews
        if len(user_ratings) < 3:
            raise Exception("Not enough movies have been rated to use this "
                            + "algorithm.")

        # create the recommendation model
        model = create_model(algorithm, user_ratings)
        if model is None:
            raise Exception("Unable to create model for this algorithm.")

        # Run the "model" through all movie ids. Returns a list of
        # movie ids for the top scores.
        predictions = []
        load_global_var_as_needed("movie_medians", "movie_medians_full.bin")

        for movie_id in movie_medians:
            score = model.predict(movie_id)
            if score is not None:
                predictions.append((score, movie_id))

        predictions.sort(reverse=True)

        # Go through the movie ids and take just the movies
        # that have not been rated by the user.
        movie_ids = []
        count = 0

        for _, movie_id in predictions:
            if movie_id not in user_ratings_dict:
                movie_ids.append(movie_id)
                count += 1

                if count >= num_results: break

        # Save the new recommendations so they don't have to be
        # recomputed again.
        user_data.store_recommendation(user_id, movie_ids, algorithm)

        # the model parameters will be stored as JSON string
        model_params = json.dumps(model.get_param_list())
        user_data.db_write_native(user_id, algorithm + "_params", model_params)

        # return the 0-th rotation of the movie_ids
        return movie_ids[0::user_data.rotation_size], model_params

    else:
        return user_data.get_recommendation(user_id, algorithm), \
            user_data.db_get_native(user_id, algorithm + "_params")


def create_model(algorithm : str, user_ratings : list):
    """
    :param algorithm: A string, such as "als3".
    :param user_ratings: A list: [(movie id, rating)].
    :return: A Model object, or None
    """
    if algorithm == "tag_count":
        load_global_var_as_needed("movie_genres", "movie_genres.bin")
        load_global_var_as_needed("movie_tags", "movie_tags.bin")
        load_global_var_as_needed("tag_counts", "tag_counts.bin")
        load_global_var_as_needed("genre_counts", "genre_counts.bin")
        load_global_var_as_needed("movie_medians", "movie_medians_full.bin")
        load_global_var_as_needed("genre_ids", "genre_ids.bin")
        load_global_var_as_needed("tag_ids", "tag_ids.bin")

        return TagCount_UserProfile(movie_genres, user_ratings, movie_tags,
                                    tag_counts, genre_counts, movie_medians,
                                    genre_ids, tag_ids)

    elif algorithm == "tag_ls":
        load_global_var_as_needed("movie_genres", "movie_genres.bin")
        load_global_var_as_needed("movie_tags", "movie_tags.bin")
        load_global_var_as_needed("tag_counts", "tag_counts.bin")
        load_global_var_as_needed("movie_medians", "movie_medians_full.bin")
        load_global_var_as_needed("genre_ids", "genre_ids.bin")
        load_global_var_as_needed("tag_ids", "tag_ids.bin")

        return TagLS_UserProfile(movie_genres, user_ratings, movie_tags,
                                 tag_counts, movie_medians, genre_ids, tag_ids)

    elif algorithm.startswith("als"):
        load_global_var_as_needed("movie_medians", "movie_medians_full.bin")

        # load "alsX_item_factors" and "alsX_movie_ids"
        load_global_var_as_needed(algorithm + "_item_factors",
                                  algorithm + "_item_factors.bin")
        load_global_var_as_needed(algorithm + "_movie_ids",
                                  algorithm + "_movie_ids.bin")
        als_item_factors = globals()[algorithm + "_item_factors"]
        als_movie_ids = globals()[algorithm + "_movie_ids"]

        num_factors = int(algorithm[3:])

        model = ALS_Model(num_factors, user_ratings, movie_medians,
                          als_item_factors, als_movie_ids)

        if model.is_valid() == False:
            raise Exception("Not enough movies have been rated to use this "
                            + "algorithm.")

        return model

    else:
        return None



# testing:
# movie_ids, model_params = get_recommendations(-1, "als9")
# print()

