"""
This module investigates making recommendations by
just reporting the movie median as an estimate.
This serves as a baseline algorithm.
"""

import csv, numpy, random, os
import matplotlib.pyplot as pyplot


data_dir = "data" + os.sep


class CsvData:
    """This class contains data read from csv files.
    ::
        user_ratings - {user_id: [movie_ids_list, ratings_list]}
    """

    def __init__(self):
        self.user_ratings = self.read_ratings_csv()


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



class ModelData:
    """Additional data structures needed for modeling.
    ::
        movie_medians - {movie_id : movie_median_rating}
        user_ratings_train - {user_id: [movie_ids_list, ratings_list]}
        user_ratings_test - {user_id: [movie_ids_list, ratings_list]}
    """
    def __init__(self, csv_data, training_set_ratio = 0.8):
        """
        :param csv_data: A CsvData object
        :param training_set_ratio: ratio of the "user_ratings" data to
            be used for training set
        """
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
            # To use average instead of median:
            # movie_medians[movie_id] = sum(movie_ratings[movie_id]) / len(movie_ratings[movie_id])
            # if using average - rank agreement 0.65 ~ 0.66

        return movie_medians


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



class Tester:
    def __init__(self, csv_data : CsvData, model_data : ModelData):
        self.csv_data = csv_data
        self.model_data = model_data


    def test(self, user_id):
        """go through the test set, returns the rank agreement."""
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
            predicted_ratings[movie_id] = self.model_data.movie_medians[movie_id]

        # compute ranking agreement
        real_ratings2 = self.convert_ratings_to_list_of_list(movie_ids2, real_ratings2)
        return self.compute_ranking_agreement(real_ratings2, predicted_ratings)


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
    pyplot.title("Prediction Ranking Agreement (using only movie median)", fontsize=14)

    average_agreement = sum(agreements) / len(agreements)
    print("Average agreement:", average_agreement)
    pyplot.show()


if __name__ == "__main__":
    main()


