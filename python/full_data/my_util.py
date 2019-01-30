"""
Utility functions
"""

import os, pickle
import config



class Model:
    """Interface for movie prediction models."""
    def predict(self, movie_id : int):
        """Predicts a rating for the given movie_id. Returns None if
        no prediction can be made."""
        return None



def split(start: int, length: int, num_splits: int):
    """Given a "start" and a "length", generate
    a list of (index, length) pairs. For example,
    (start=10, length=8, num_splits=4) generates
    [(10, 2), (12, 2), (14, 2), (16, 2)]."""

    if length >= num_splits:
        # standard case
        # compute the indices
        indices = []
        for i in range(0, num_splits):
            indices.append(start + int(length * i / num_splits))

        result = []
        # most of the lengths are (next index - current index)
        for i in range(0, len(indices) - 1):
            result.append((indices[i], indices[i+1] - indices[i]))

        # the length for the final index:
        final_length = start + length - indices[-1]
        result.append((indices[-1], final_length))

        return result

    else:
        # special case
        result = []
        index = start
        for i in range(0, num_splits):
            if index < start + length:
                result.append((index, 1))
                index += 1
            else:
                result.append((index, 0))

        return result


def convert_ratings_to_list_of_list(movie_ratings):
    """ Convert information in a list of (movie_id, rating)
    format to a list of lists format that is suitable
    for rank agreement computation.

    :param movie_ratings: a list of (movie_id, rating) tuples
    :return: a list that looks like [[movies with 5 stars], [movie with 4 stars], [movies with 3 stars], ...]
    """

    # collect movie ids into lists, grouped by ratings
    rating_to_movie_id = {}

    for movie_id, rating in movie_ratings:
        if rating not in rating_to_movie_id:
            rating_to_movie_id[rating] = []

        rating_to_movie_id[rating].append(movie_id)

    # create a list of lists, sorted by ratings
    rating_keys = list(rating_to_movie_id.keys())
    rating_keys.sort(reverse=True)

    list_of_lists = []
    for rating in rating_keys:
        list_of_lists.append(rating_to_movie_id[rating])

    return list_of_lists


def has_different_ratings(movie_ratings, start):
    """Check that the "movie_ratings" tuple list, starting
    at index "start", has at least two different ratings.

    :param movie_ratings: A list of (movie_id, rating) tuples.
    :param start: The index to start the check
    :return: True if there are at least two different ratings.
    """
    rating1 = movie_ratings[start][1]
    for i in range(start + 1, len(movie_ratings)):
        if movie_ratings[i][1] != rating1: return True

    return False


def compute_ranking_agreement(actual_ratings, predicted_ratings):
    """ Compute a ranking agreement percentage. Return either a
    percentage agreement value or None if the actual ratings
    are not sufficiently varied.

    :param actual_ratings: a list of (movie_id, rating) tuples
    :param predicted_ratings: a list of (movie_id, rating) tuples
    :return: a percentage indicating the agreement level between the two sets of rankings
    """
    # handle special cases first
    if len(actual_ratings) == 1:
        # Actual ratings are all the same score.
        # So no user preference is expressed.
        # This test is not valid.
        return None

    if has_different_ratings(actual_ratings, 0) == False:
        # Actual ratings are all the same score, so test invalid.
        return None

    # convert "actual_ratings" to "list of lists" format
    actual_ratings = convert_ratings_to_list_of_list(actual_ratings)

    # convert "predicted_ratings" to a dictionary, where dict[movie_id] = rating
    predicted_ratings_list = predicted_ratings
    predicted_ratings = {}

    for movie_id, rating in predicted_ratings_list:
        predicted_ratings[movie_id] = rating

    agreement = 0
    disagreement = 0
    # enumerate all pairs from "actual_ratings"
    for i in range(0, len(actual_ratings) - 1):
        for movie_id1 in actual_ratings[i]:
            for j in range(i + 1, len(actual_ratings)):
                for movie_id2 in actual_ratings[j]:
                    # the requirement is that:
                    # rating(movie_id1) > rating(movie_id2)
                    if predicted_ratings[movie_id1] > predicted_ratings[movie_id2]:
                        agreement += 1
                    else:
                        disagreement += 1

    return agreement / (agreement + disagreement)


def print_rank_agreement_results(agreements, model_name):
    """Print statistics about rank agreement values.
    If the OS is Windows, will draw a graph.

    :param agreements: A list of rank agreement percentages.
    :param model_name: Name of the model being used.
    """
    print("Rank agreement data has", len(agreements), "values in it.")
    print('Average rank agreement when using "', model_name,
          '" for prediction:', sum(agreements) / len(agreements))

    if os.name == "nt":
        # if Windows, plot a histogram
        import matplotlib.pyplot as pyplot

        pyplot.hist(agreements, bins=20)
        pyplot.xlabel("ranking agreement")
        pyplot.ylabel("frequency")
        pyplot.title("Prediction Ranking Agreement (" + model_name + ")",
                     fontsize=14)
        pyplot.show()

    print("Removing data with rank agreement < 0.05 and > 0.95.")
    agreements2 = []
    for val in agreements:
        if 0.05 <= val <= 0.95:
            agreements2.append(val)

    print("New rank agreement data has", len(agreements2), "values in it.")
    print("New average is", sum(agreements2) / len(agreements2))


