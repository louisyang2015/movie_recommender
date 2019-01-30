"""
This module investigates making recommendations by
building a profile for each user and tracking
the movie genres and tags. Scores above 3 are
considered positive, contributing to genre and
tag scores, while scores below 3 are subtracted
away from a user's genre and tag scores.
"""

import csv, numpy, math, random, os
import matplotlib.pyplot as pyplot
from collections import defaultdict


data_dir = "data" + os.sep


class CsvData:
    """This class contains data read from csv files.
    ::
        genre_ids - dictionary[genre string] = genre id
        movie_genres - dictionary[movie id] = set of genre ids
        user_ratings - dictionary[user_id] = [movie_ids_list, ratings_list]
        tag_ids - dictionary[tag string] = tag id
        movie_tags - dictionary[movie id] = list of tag ids
    """

    def __init__(self):
        self.genre_ids, self.movie_genres = self.read_movies_csv()
        self.user_ratings = self.read_ratings_csv()
        self.tag_ids, self.movie_tags = self.read_tags_csv()


    def read_movies_csv(self):
        """ Read "movies.csv". Returns genre_str_to_id, movie_id_to_genre_ids.
        ::
            genre_str_to_id - dictionary[genre string] = genre id
            movie_id_to_genre_ids - dictionary[movie id] = set of genre ids
        """
        # dictionary[genre string] = genre id
        genre_str_to_id = {}
        next_genre_id = 0

        # dictionary[movie id] = set of genre ids
        movie_id_to_genre_ids = {}

        # read from "movies.csv" to collect genres
        with open(data_dir + "movies.csv", "r", encoding="utf-8", newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)  # skip the first row

            for row in csv_reader:
                movie_id = int(row[0])
                genre_strings = row[2].lower().split('|')

                # go through each genre found
                for genre_str in genre_strings:
                    if genre_str != "(no genres listed)":

                        # add "genre_str" to "genre_str_to_id"
                        if genre_str not in genre_str_to_id:
                            genre_str_to_id[genre_str] = next_genre_id
                            next_genre_id += 1

                        genre_id = genre_str_to_id[genre_str]

                        # add genre_id to "movie_id_to_genre_ids"
                        if movie_id not in movie_id_to_genre_ids:
                            movie_id_to_genre_ids[movie_id] = set()

                        movie_id_to_genre_ids[movie_id].add(genre_id)

        return genre_str_to_id, movie_id_to_genre_ids


    def read_ratings_csv(self):
        """ Reads the "ratings.csv" file. Returns user_ratings.
        ::
            user_ratings - dictionary[user_id] = [movie_ids_list, ratings_list]
        """

        user_ratings = {}

        with open(data_dir + "ratings.csv", "r", encoding="utf-8", newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)  # skip the first row

            for row in csv_reader:
                user_id = int(row[0])

                if user_id not in user_ratings:
                    user_ratings[user_id] = [[], []]

                user_ratings[user_id][0].append(int(row[1])) # movie_id
                user_ratings[user_id][1].append(float(row[2]))  # rating

        return user_ratings


    def read_tags_csv(self):
        """Read "tags.csv". Returns tag_str_to_id, movie_id_to_tag_ids.
        ::
            tag_str_to_id - dictionary[tag string] = tag id
            movie_id_to_tag_ids - dictionary[movie id] = list of tag ids
        """
        # dictionary[tag string] = tag id
        tag_str_to_id = {}
        next_tag_id = 0

        # dictionary[movie id] = list of tag ids
        movie_id_to_tag_ids = {}

        # read from "tags.csv" to collect tags
        with open(data_dir + "tags.csv", "r", encoding="utf-8", newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)  # skip the first row

            for row in csv_reader:
                movie_id = int(row[1])
                tag = row[2].lower()

                # add "tag" to "tag_str_to_id"
                if tag not in tag_str_to_id:
                    tag_str_to_id[tag] = next_tag_id
                    next_tag_id += 1

                tag_id = tag_str_to_id[tag]

                # add "tag_id" to "movie_id_to_tag_ids"
                if movie_id not in movie_id_to_tag_ids:
                    movie_id_to_tag_ids[movie_id] = []

                movie_id_to_tag_ids[movie_id].append(tag_id)

        return tag_str_to_id, movie_id_to_tag_ids


class ModelData:
    """Additional data structures needed for modeling.
    ::
        movie_medians - {movie_id : movie_median_rating}
        genre_count - {genre_id : genre count}
        tag_min_appearance - only tags that appeared "tag_min_appearance"
            times or more are used
        movie_tags - {movie_id: tag_id: ln(count) + 1}
        tag_count - {tag_id: count}
        user_ratings_train - {user_id: [movie_ids_list, ratings_list]}
        user_ratings_test - {user_id: [movie_ids_list, ratings_list]}
    """
    def __init__(self, csv_data, tag_min_appearance = 6,
                 training_set_ratio = 0.8):
        """
        :param csv_data: A CsvData object
        :param tag_min_appearance: minimum number of times that a tag
            should appear for it to be used in the model.
        :param training_set_ratio: ratio of the "user_ratings" data to
            be used for training set
        """
        self.genre_count = self.compute_genre_count(csv_data.movie_genres)
        self.tag_min_appearance = tag_min_appearance
        self.movie_tags, self.tag_count = self.compute_new_movie_tags(csv_data.movie_tags)

        self.user_ratings_train, self.user_ratings_test = \
            self.compute_training_set(csv_data.user_ratings, training_set_ratio)

        self.movie_medians = self.compute_movie_medians(self.user_ratings_train)


    def compute_movie_medians(self, user_ratings):
        """ Returns movie_medians.

        :param user_ratings: {user_id: [movie_ids_list, ratings_list]}
        :return: {movie_id : movie_median_rating}
        """
        # collect all ratings and group them by the movie
        # dictionary[movie_id] = list of ratings
        movie_ratings = {}

        for user_id in user_ratings.keys():
            movie_ids = user_ratings[user_id][0]
            ratings = user_ratings[user_id][1]

            for i in range(0, len(movie_ids)):
                movie_id = movie_ids[i]
                rating = ratings[i]

                # record (movie_id, rating) into "movie_ratings"

                if movie_id not in movie_ratings:
                    movie_ratings[movie_id] = []

                movie_ratings[movie_id].append(rating)

        # dictionary[movie_id] = movie_median
        movie_medians = {}

        # compute a median for each movie
        for movie_id in movie_ratings.keys():
            movie_medians[movie_id] = numpy.median(movie_ratings[movie_id])
            # movie_medians[movie_id] = numpy.average(movie_ratings[movie_id])

        return movie_medians


    def compute_genre_count(self, movie_genres):
        """ Returns genre count.

        :param movie_genres: {movie_id : set of genre ids}
        :return: {genre_id : count}
        """
        genre_count = defaultdict(int) # {genre_id : genre count}

        for movie_id in movie_genres.keys():
            for genre_id in movie_genres[movie_id]:
                genre_count[genre_id] += 1

        return genre_count


    def compute_new_movie_tags(self, movie_tags):
        """Returns a movie tags data structure that uses fewer
        tags and keeps track of number of appearances of tags.

        :param movie_tags: {movie id: list of tag ids}
        :return: {movie id: tag id: ln(count) + 1}, {tag_id: count}
        """
        # count all tags
        tag_count = defaultdict(int)  # {tag_id: count}

        for movie_id in movie_tags.keys():
            for tag_id in movie_tags[movie_id]:
                tag_count[tag_id] += 1

        # use tags that appeared at least "tag_min_appearance" times
        # build a set containing tag ids that will be used
        tag_id_set = set()

        for tag_id in tag_count:
            if tag_count[tag_id] >= self.tag_min_appearance:
                tag_id_set.add(tag_id)

        # build a new_movie_tags that uses only the filtered tags
        new_movie_tags = {}  # {movie_id : tag_id : count}

        for movie_id in movie_tags.keys():
            new_movie_tags[movie_id] = {}

            for tag_id in movie_tags[movie_id]:
                if tag_id in tag_id_set:
                    if tag_id not in new_movie_tags[movie_id]:
                        new_movie_tags[movie_id][tag_id] = 0

                    new_movie_tags[movie_id][tag_id] += 1

        # remap new_movie_tags[movie_id][tag_id] score to limit growth
        # in the case that value is > 1
        # use ln(x) + 1
        for movie_id in new_movie_tags.keys():
            for tag_id in new_movie_tags[movie_id].keys():
                score = new_movie_tags[movie_id][tag_id]
                new_movie_tags[movie_id][tag_id] = math.log(score) + 1

        # build a new tag_count that has only the filtered tags
        tag_count2 = {} # {tag_id: count}
        for tag_id in tag_id_set:
            tag_count2[tag_id] =tag_count[tag_id]

        return new_movie_tags, tag_count2


    def compute_training_set(self, user_ratings, training_set_ratio):
        """ Split "user_ratings" into a training set and a testing set.
        Returns user_ratings_training, user_ratings_testing.

        :param user_ratings: {user_id: [movie_ids_list, ratings_list]}
        :param training_set_ratio: ratio of the "user_ratings" data to
            be used for training set
        :return: user_ratings_train, user_ratings_test
        """
        if training_set_ratio == 1:
            return user_ratings, None

        user_ratings_train = {} # {user_id: [movie_ids_list, ratings_list]}
        user_ratings_test = {}

        for user_id in user_ratings:
            movie_ids = user_ratings[user_id][0]
            ratings = user_ratings[user_id][1]

            if len(movie_ids) > 10:
                # shuffle the data - make a copy to preserve the original data
                movie_ids = movie_ids.copy()
                ratings = ratings.copy()
                self.shuffle(movie_ids, ratings)

                # split according to "training_set_ratio"
                split_index = int(len(movie_ids) * training_set_ratio)
                user_ratings_train[user_id] = [movie_ids[:split_index], ratings[:split_index]]
                user_ratings_test[user_id] = [movie_ids[split_index:], ratings[split_index:]]

        return user_ratings_train, user_ratings_test


    def shuffle(self, list1, list2):
        """Shuffles the content of the two lists. The two lists
        are assumed to be the same length."""
        length = len(list1)

        for i in range(0, length):
            swap_index = random.randint(0, length - 1)

            if i != swap_index:
                list1[i], list1[swap_index] = list1[swap_index], list1[i]
                list2[i], list2[swap_index] = list2[swap_index], list2[i]



class UserProfile:
    """Model for a single user.
    ::
        tag_profile - {tag_id: score}
        genre_profile - {genre_id: score}
        x_factors0 - 3 variable vector for movies with tags
        x_factors1 - 2 variable vector for movies without tags
    """

    def __init__(self, user_id, csv_data, model_data):
        """
        :param user_id: integer
        :param csv_data: a CsvData object
        :param model_data: a ModelData object
        """
        movie_genres = csv_data.movie_genres

        movie_ids = model_data.user_ratings_train[user_id][0]
        ratings = model_data.user_ratings_train[user_id][1]

        movie_tags = model_data.movie_tags
        tag_count = model_data.tag_count
        genre_count = model_data.genre_count
        movie_medians = model_data.movie_medians

        self.tag_profile = self.compute_tag_profile(movie_ids, ratings,
                                                    movie_tags, tag_count)
        self.genre_profile = self.compute_genre_profile(movie_ids, ratings,
                                                        movie_genres, genre_count)
        self.x_factors0, self.x_factors1 = \
            self.compute_x_factors(movie_ids, ratings, movie_tags,
                                   movie_genres, movie_medians)


    def compute_tag_profile(self, movie_ids, ratings, movie_tags, tag_count):
        """Returns a tag_profile object.

        :param movie_ids: list of int
        :param ratings: list of float
        :param movie_tags: {movie_id: tag_id: count}
        :param tag_count: {tag_id: count}
        :return: {tag_id: score}
        """
        tag_profile = {} # {tag_id: score}

        # for each movie review, add to "tag_profile"
        for i in range(0, len(movie_ids)):
            movie_id = movie_ids[i]
            offset = ratings[i] - 3

            # it's possible for a movie to be untagged
            if (offset != 0) and (movie_id in movie_tags):
                for tag_id in movie_tags[movie_id].keys():
                    if tag_id not in tag_profile:
                        tag_profile[tag_id] = 0

                    tag_profile[tag_id] += offset * movie_tags[movie_id][tag_id]

        # normalize the values in "tag_profile"
        for tag_id in tag_profile.keys():
            tag_profile[tag_id] /= tag_count[tag_id]

        return tag_profile


    def compute_genre_profile(self, movie_ids, ratings, movie_genres, genre_count):
        """Returns a genre_profile object.

        :param movie_ids: list of int
        :param ratings: list of float
        :param movie_genres: {movie_id: set of genre ids}
        :param genre_count: {genre_id: genre count}
        :return: {genre_id: score}
        """
        genre_profile = {} # {genre_id: score}

        # for each movie review, add to "genre_profile"
        for i in range(0, len(movie_ids)):
            movie_id = movie_ids[i]
            offset = ratings[i] - 3

            # it's possible for a movie to have no genre classified
            if (offset != 0) and (movie_id in movie_genres):
                for genre_id in movie_genres[movie_id]:
                    if genre_id not in genre_profile:
                        genre_profile[genre_id] = 0

                    genre_profile[genre_id] += offset

        # normalize the values in "genre_profile"
        for genre_id in genre_profile.keys():
            genre_profile[genre_id] /= genre_count[genre_id]

        return genre_profile


    def dot_product_with_movie_genre_ids(self, movie_genre_ids):
        """
        :param movie_genre_ids: a set of genre ids
        """
        dot_product = 0
        genre_profile = self.genre_profile

        for genre_id in movie_genre_ids:
            if genre_id in genre_profile:
                dot_product += genre_profile[genre_id]

        return dot_product


    def dot_product_with_movie_tag_scores(self, movie_tag_scores):
        """
        :param movie_tag_scores: {tag_id: score}
        """
        dot_product = 0
        tag_profile = self.tag_profile

        for tag_id in movie_tag_scores:
            if tag_id in tag_profile:
                dot_product += tag_profile[tag_id] * movie_tag_scores[tag_id]

        return dot_product


    def compute_x_factors(self, movie_ids, ratings, movie_tags,
                          movie_genres, movie_medians):
        """ Fit this user to the Ax = b model using least squares.

        :param movie_ids: list of int
        :param ratings: list of float
        :param movie_tags: {movie_id: tag_id: score}
        :param movie_genres: {movie_id: set of genre ids}
        :param movie_medians: {movie_id : movie median rating}
        :return: x_factor0, x_factor1 - x_factor0 has 3
        variables, x_factor1 has two variables
        """
        # model 0: A0 * x = b0 - when tags are present
        # model 1: A1 * x = b1 - when tags are not present

        # build up "A" and "b" in "Ax" = "b"
        A0 = []
        b0 = []
        A1 = []
        b1 = []

        for i in range(0, len(movie_ids)):
            movie_id = movie_ids[i]

            # each movie rating is a single row
            # the b value is always the same
            b = ratings[i] - movie_medians[movie_id]

            # tag and genre dot products
            tag_dot_product = 0
            genre_dot_product = 0

            if movie_id in movie_tags.keys():
                tag_dot_product = self.dot_product_with_movie_tag_scores(movie_tags[movie_id])

            if movie_id in movie_genres.keys():
                genre_dot_product = self.dot_product_with_movie_genre_ids(movie_genres[movie_id])

            if math.fabs(tag_dot_product > 0.1):
                # tags are present
                A0.append([tag_dot_product, genre_dot_product, 1.0])
                b0.append([b])
            else:
                # alternative model #1 - no "tag_dot_product"
                A1.append([genre_dot_product, 1.0])
                b1.append([b])
            # an alternative training approach is to use all data for model #1

        # least square solutions
        x_factors0 = None
        if len(A0) > 0:
            result = numpy.linalg.lstsq(A0, b0, rcond=None)
            x_factors0 = result[0]

        x_factors1 = None
        if len(A1) > 0:
            result = numpy.linalg.lstsq(A1, b1, rcond=None)
            x_factors1 = result[0]

        return x_factors0, x_factors1


    def predict(self, movie_tag_scores, movie_genre_ids, movie_median):
        """ Predicts a rating for the movie

        :param movie_tag_scores: {tag_id: movie tag score}
        :param movie_genre_ids: a set of genre ids
        :param movie_median: median score for the movie
        :return: a rating for the movie
        """
        tag_product = 0
        genre_product = 0

        if movie_tag_scores is not None:
            tag_product = self.dot_product_with_movie_tag_scores(movie_tag_scores)

        if movie_genre_ids is not None:
            genre_product = self.dot_product_with_movie_genre_ids(movie_genre_ids)

        # preferred situation: use model 0
        x = self.x_factors0
        A = [tag_product, genre_product, 1.0]

        # fall back situation: no tag, or if x_factors0 is None
        if (tag_product < 1e-6) or x is None:
            x = self.x_factors1
            A = [genre_product, 1.0]

        # it's possible that x is still None
        if x is None:
            return movie_median # last fall back

        # standard model: Ax + movie_median
        A = numpy.array([A])
        return A.dot(x)[0][0] + movie_median



class Tester:
    def __init__(self, csv_data, model_data):
        self.csv_data = csv_data
        self.model_data = model_data


    def test(self, user_id):
        """Returns ranking agreement percentage for "user_id"."""
        user_profile = UserProfile(user_id, self.csv_data, self.model_data)

        # run user_profile through the test set
        movie_ids = self.model_data.user_ratings_test[user_id][0]
        real_ratings = self.model_data.user_ratings_test[user_id][1]

        # test only on movies that have been seen before,
        # meaning movies that have a median value
        movie_ids2 = []
        real_ratings2 = []

        for (movie_id, rating) in zip(movie_ids, real_ratings):
            if movie_id in self.model_data.movie_medians:
                movie_ids2.append(movie_id)
                real_ratings2.append(rating)

        # predict ratings for "movie_ids2"
        predicted_ratings = {}
        for movie_id in movie_ids2:
            predicted_ratings[movie_id] = self.predict(user_profile, movie_id)

        # compute ranking agreement
        real_ratings2 = self.convert_ratings_to_list_of_list(movie_ids2, real_ratings2)
        return self.compute_ranking_agreement(real_ratings2, predicted_ratings)


    def predict(self, user_profile, movie_id):
        movie_tag_scores = None
        if movie_id in self.model_data.movie_tags:
            movie_tag_scores = self.model_data.movie_tags[movie_id]

        movie_genre_ids = None
        if movie_id in self.csv_data.movie_genres:
            movie_genre_ids = self.csv_data.movie_genres[movie_id]

        movie_median = self.model_data.movie_medians[movie_id]
        return user_profile.predict(movie_tag_scores, movie_genre_ids, movie_median)


    def convert_ratings_to_list_of_list(self, movie_ids, ratings):
        """ Convert information in two lists into a list of lists format
        that is suitable for rank agreement computation.

        :param movie_ids: a list
        :param ratings: a list
        :return: a list that looks like [[movies with 5 stars], [movie with 4 stars], [movies with 3 stars], ...]
        """

        # collect movie ids into lists, grouped by ratings
        rating_to_movie_id = {}
        for i in range(0, len(movie_ids)):
            movie_id = movie_ids[i]
            rating = ratings[i]

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


    def compute_ranking_agreement(self, actual_ratings, predicted_ratings):
        """ Compute a ranking agreement percentage.

        :param actual_ratings: in "list of lists" format
        :param predicted_ratings: a dictionary, where dict[movie_id] = rating
        :return: a percentage indicating the agreement level between the two sets of rankings
        """
        # handle special case first
        if len(actual_ratings) == 1:
            # Actual ratings are all the same score
            # So no user preference is expressed
            # All predictions are viewed as valid
            return 1.0

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




def main():
    csv_data = CsvData()
    model_data = ModelData(csv_data)
    tester = Tester(csv_data, model_data)

    agreements = []
    for user_id in model_data.user_ratings_test.keys():
        # print(user_id) - to see where things crashed
        agreements.append(tester.test(user_id))

    pyplot.hist(agreements, bins=20)
    pyplot.xlabel("ranking agreement")
    pyplot.ylabel("frequency")
    pyplot.title("Prediction Ranking Agreement (tag counting model)", fontsize=14)

    average_agreement = sum(agreements) / len(agreements)
    print("Average agreement:", average_agreement)
    pyplot.show()


if __name__ == "__main__":
    main()

