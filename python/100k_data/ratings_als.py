"""
This module investigates making recommendations by
building a linear model based on just "ratings.csv".
The linear model is trained using the alternating
least squares procedure.
"""

import csv, random, math, numpy, os
import matplotlib.pyplot as pyplot
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


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


    def plot_user_reviews_histogram(self):
        # build up a list of number of reviews (per user)
        num_reviews = []
        user_ratings = self.user_ratings

        for user_id in user_ratings:
            num_reviews.append(len(user_ratings[user_id][0]))

        pyplot.figure()
        pyplot.hist(num_reviews, range=(0,100), rwidth=0.9)
        pyplot.xlabel("Number of reviews")
        pyplot.ylabel("Number of users")
        pyplot.title("Histogram of reviews made per user")


    def plot_movie_reviews_histogram(self):
        # count reviews per movie
        num_reviews = {} # { movie_id: number of reviews }
        user_ratings = self.user_ratings

        for user_id in user_ratings:
            for movie_id in user_ratings[user_id][0]:
                if movie_id not in num_reviews:
                    num_reviews[movie_id] = 0

                num_reviews[movie_id] += 1

        pyplot.figure()
        pyplot.hist(list(num_reviews.values()), range=(0, 30), rwidth=0.9)
        pyplot.xlabel("Number of reviews")
        pyplot.ylabel("Number of movies")
        pyplot.title("Histogram of reviews received per movie")




class ModelData:
    """Additional data structures needed for modeling.
    ::
        movie_medians - {movie_id : movie_median_rating}
        user_ratings_train - {user_id: [movie_ids_list, ratings_list]}
        user_ratings_test - {user_id: [movie_ids_list, ratings_list]}
    """
    def __init__(self, csv_data : CsvData, training_set_ratio = 0.8):
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
            # movie_medians[movie_id] = numpy.average(movie_ratings[movie_id])
            # for ALS factor 3, using average is better than median

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



class ALS_Model:
    """ Linear model that predicts rating by using movie factors
    dot product with user factors.
    ::
        factors - the number of factor variables per movie
        movie_medians - {movie_id : movie_median_rating}
        num_movies - number of movies covered by the model
        num_users - number of users covered by the model

        movie_id_lookup - {data set movie id : ALS model movie id}
        user_id_lookup - {data set user id : ALS model user id}

        movies - numpy vector representing movie factors
        users - numpy vector representing user factors
    """

    def __init__(self, model_data : ModelData, factors : int):

        self.movie_medians = model_data.movie_medians
        self.factors = factors

        # Unpack "user_ratings_train" into parallel lists
        user_ratings = model_data.user_ratings_train

        user_ids = []
        movie_ids = []
        ratings = []

        for user_id in user_ratings:
            for i in range(0, len(user_ratings[user_id][0])):
                user_ids.append(user_id)
                movie_ids.append(user_ratings[user_id][0][i])
                ratings.append(user_ratings[user_id][1][i])

        # Shrink lists to satisfy "factors"
        user_ids, movie_ids, ratings = \
            self.shrink_dataset(user_ids, movie_ids, ratings, factors)

        # Prepare (user_ids, movie_ids, ratings) for ALS.
        # Subtract median from ratings
        for i in range(0, len(movie_ids)):
            ratings[i] -= self.movie_medians[movie_ids[i]]

        # Remap IDs so they are zero based
        user_ids, self.user_id_lookup = self.remap_ids(user_ids)
        movie_ids, self.movie_id_lookup = self.remap_ids(movie_ids)

        # Print model statistics
        self.num_movies = len(self.movie_id_lookup)
        self.num_users = len(self.user_id_lookup)
        print("Model uses", factors, "factors,",
              self.num_users, "users,",
              self.num_movies, "movies, and",
              len(ratings), "ratings.")

        # Fit model to data using ALS
        self.users, self.movies = self.fit_to_data(user_ids, movie_ids, ratings)


    def count_id_appearance(self, id_list : list):
        """Return a dictionary of how many times each id appeared."""
        id_appearance = {}

        for id in id_list:
            if id not in id_appearance:
                id_appearance[id] = 0

            id_appearance[id] += 1

        return id_appearance


    def shrink_dataset(self, user_ids : list, movie_ids : list,
                       ratings : list, factors : int):
        """Reduce the training set "user_ids, movie_ids, ratings"
        dataset so to satisfy the "factors" constraint. Only users
        with "factors + 1" or more appearances, and movies with
        "factors" or more appearances will remain in the
        return dataset.

        return new user_ids, movie_ids, ratings
        """

        while len(user_ids) > 0:
            # count appearance of user_id and movie_id
            user_id_appearance_dict = self.count_id_appearance(user_ids)
            movie_id_appearance_dict = self.count_id_appearance(movie_ids)

            # go through the data and create new lists that satisfy the
            # factors constraint
            changes = False
            new_user_ids = []
            new_movie_ids = []
            new_ratings = []

            for i in range(0, len(user_ids)):
                user_id = user_ids[i]
                movie_id = movie_ids[i]

                if user_id_appearance_dict[user_id] >= factors + 1 \
                    and movie_id_appearance_dict[movie_id] >= factors:

                    new_user_ids.append(user_ids[i])
                    new_movie_ids.append(movie_ids[i])
                    new_ratings.append(ratings[i])

                else:
                    changes = True

            if changes:
                # need to go through all the data again
                user_ids = new_user_ids
                movie_ids = new_movie_ids
                ratings = new_ratings

                # keep looping

            else:
                # no changes found means done.
                return new_user_ids, new_movie_ids, new_ratings


    def remap_ids(self, id_list: list):
        """Remap ids in "id_list" to zero based values.

        return new_id_list, zero_based_id_lookup
        """
        zero_based_id_lookup = {} # {standard id: zero based id}
        new_id_list = []
        next_free_id = 0

        for i in range(0, len(id_list)):
            id = id_list[i]

            if id not in zero_based_id_lookup:
                zero_based_id_lookup[id] = next_free_id
                next_free_id += 1

            new_id_list.append(zero_based_id_lookup[id])

        return new_id_list, zero_based_id_lookup


    def can_predict(self, user_id, movie_id=None):
        """Returns True if the model can be used for "user_id" and "movie_id"."""

        if movie_id is None:
            if user_id in self.user_id_lookup: return True
            else: return False

        if user_id in self.user_id_lookup and movie_id in self.movie_id_lookup:
            return True
        else:
            return False


    def solve_for_users(self, movies : numpy.ndarray,
                        user_ids : list, movie_ids : list,
                        ratings : numpy.ndarray):
        """Given "movies", solve for "users". Return users."""

        num_users = self.num_users
        factors = self.factors

        # create the "A" matrix in "Ax=b"
        # It's a sparse matrix in (data, (row, col)) format
        ratings_length = ratings.shape[0]

        data = numpy.zeros(ratings_length * (factors + 1),
                           dtype=numpy.double)
        row = numpy.zeros(ratings_length * (factors + 1),
                          dtype=numpy.int)
        col = numpy.zeros(ratings_length * (factors + 1),
                          dtype=numpy.int)

        for i in range(0, ratings_length):
            movie_id = movie_ids[i]
            user_id = user_ids[i]

            for j in range(0, factors):
                # row is always the same
                row[i * (factors + 1) + j] = i

                # column[...] is user_id
                col[i * (factors + 1) + j] = user_id * (factors + 1) + j

                # data[...] = movie factors
                data[i * (factors + 1) + j] = movies[movie_id * factors + j][0]

            # bias term = 1
            row[i * (factors + 1) + factors] = i
            col[i * (factors + 1) + factors] = user_id * (factors + 1) + factors
            data[i * (factors + 1) + factors] = 1

        m = sparse.coo_matrix((data, (row, col)),
                              shape=(ratings_length, num_users * (factors + 1)),
                              dtype=numpy.double)
        A = m.tocsr()

        result = linalg.lsqr(A, ratings, iter_lim=100)
        users = numpy.array([result[0]]).T

        # print("Size of A =", A.shape)
        # print("Number of iterations =", result[2])

        return users


    def solve_for_movies(self, users : numpy.ndarray,
                         user_ids : list, movie_ids : list,
                         ratings : numpy.ndarray):
        """Given "users", solve for "movies". Return movies"""

        num_movies = self.num_movies
        factors = self.factors

        # create the "A" matrix in "Ax=b"
        # It's a sparse matrix in (data, (row, col)) format
        ratings_length = ratings.shape[0]

        data = numpy.zeros(ratings_length * factors, dtype=numpy.double)
        row = numpy.zeros(ratings_length * factors, dtype=numpy.int)
        col = numpy.zeros(ratings_length * factors, dtype=numpy.int)

        for i in range(0, ratings_length):
            movie_id = movie_ids[i]
            user_id = user_ids[i]

            for j in range(0, factors):
                # row is always the same
                row[i * factors + j] = i

                # column[...] is movie_id
                col[i * factors + j] = movie_id * factors + j

                # data[...] = user factors
                data[i * factors + j] = users[user_id * (factors + 1) + j][0]

        m = sparse.coo_matrix((data, (row, col)),
                              shape=(ratings_length, num_movies * factors),
                              dtype=numpy.double)
        A = m.tocsr()

        # the user bias needs to be subtracted from "ratings"
        bias = users[3::4]
        ratings_minus_bias = numpy.array(ratings)

        for i in range(0, ratings_minus_bias.shape[0]):
            user_id = user_ids[i]
            ratings_minus_bias[i] -= bias[user_id]

        result = linalg.lsqr(A, ratings_minus_bias, iter_lim=100)
        movies = numpy.array([result[0]]).T

        # print("Size of A =", A.shape)
        # print("Number of iterations =", result[2])

        return movies


    def training_predict(self, user_id : int, movie_id : int,
                         users : numpy.ndarray, movies : numpy.ndarray):
        """Predicts how user_id will rate movie_id. This version does
        not add on the movie median. The "users" and "movies" factor
        vectors has to be supplied by the caller."""
        factors = self.factors

        user_vec = users[user_id * (factors + 1): user_id * (factors + 1) + factors]
        user_bias = users[user_id * (factors + 1) + factors][0]
        movie_vec = movies[movie_id * factors: movie_id * factors + factors]

        prediction = user_vec.T.dot(movie_vec)[0][0] + user_bias
        return prediction


    def predict(self, user_id, movie_id):
        """Predict rating for the given user_id and movie_id.

        Returns prediction"""
        movie_median = self.movie_medians[movie_id]

        # the model uses zero based user and movie IDs
        user_id = self.user_id_lookup[user_id]
        movie_id = self.movie_id_lookup[movie_id]

        prediction = self.training_predict(user_id, movie_id, self.users, self.movies)
        prediction += movie_median

        return prediction


    def compute_training_error(self, users : numpy.ndarray,
                               movies : numpy.ndarray, user_ids : list,
                               movie_ids : list, ratings : numpy.ndarray):
        """ Compute the average error of the current "users" and "movies" estimates.
        This version has no movie median information, so it's for use with
        the training set only.

        Return average error."""
        error = 0.0

        for i in range(0, ratings.shape[0]):
            user_id = user_ids[i]
            movie_id = movie_ids[i]

            prediction = self.training_predict(user_id, movie_id, users, movies)
            error += math.fabs(prediction - ratings[i][0])

        return error / ratings.shape[0]


    def fit_to_data(self, user_ids : list, movie_ids : list,
                    ratings : list):
        """Use ALS to find "users" and "movies" factors.

        Return users, movies."""

        ratings = numpy.array([ratings]).T

        # start off with a random "movies" vector
        movies = numpy.random.uniform(-1, 1, size=(self.num_movies * self.factors, 1))

        old_error = None
        print("iteration".center(15), "average training error".center(25))

        for i in range(0, 30):
            users = self.solve_for_users(movies, user_ids, movie_ids, ratings)
            movies = self.solve_for_movies(users, user_ids, movie_ids, ratings)
            error = self.compute_training_error(users, movies, user_ids,
                                                movie_ids, ratings)
            print(str(i).center(15), str(error).center(25))

            # stop the optimization process if incremental improvement is less than 1%
            if old_error is not None:
                if error > 0.99 * old_error: break

            old_error = error

        return users, movies



class Tester:
    def __init__(self, model_data : ModelData, als_model : ALS_Model):
        self.model_data = model_data
        self.als_model = als_model


    def test(self, user_id : int):
        """Returns ranking agreement percentage for "user_id".

        Returns None if none of the test set's movies are
        covered by the ALS model."""

        movie_ids = self.model_data.user_ratings_test[user_id][0]
        real_ratings = self.model_data.user_ratings_test[user_id][1]

        # test only on movies that are covered by the ALS model
        movie_ids2 = []
        real_ratings2 = []

        for (movie_id, rating) in zip(movie_ids, real_ratings):
            if self.als_model.can_predict(user_id, movie_id):
                movie_ids2.append(movie_id)
                real_ratings2.append(rating)

        if len(movie_ids2) == 0: return None

        # predict ratings for "movie_ids2"
        predicted_ratings = {}
        for movie_id in movie_ids2:
            predicted_ratings[movie_id] = self.als_model.predict(user_id, movie_id)

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

    # some data exploration:
    # csv_data.plot_user_reviews_histogram()
    # csv_data.plot_movie_reviews_histogram()

    model_data = ModelData(csv_data)
    als_model = ALS_Model(model_data, factors=3)

    tester = Tester(model_data, als_model)

    agreements = []
    for user_id in model_data.user_ratings_test.keys():
        # print(user_id) - to see where things crashed

        if als_model.can_predict(user_id):
            agreement = tester.test(user_id)

            if agreement is not None:
                agreements.append(agreement)

    pyplot.hist(agreements, bins=20)
    pyplot.xlabel("ranking agreement")
    pyplot.ylabel("frequency")
    pyplot.title("Prediction Ranking Agreement (ALS factors model)", fontsize=14)

    average_agreement = sum(agreements) / len(agreements)
    print("Average agreement:", average_agreement)
    pyplot.show()


if __name__ == "__main__":
    main()


