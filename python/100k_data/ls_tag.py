"""
This module investigates making recommendations by
building a profile for each user. Each user's rating
is a linear function of the genres and tags. The
genre and tag coefficients are calculated using
least squares.
"""

import csv, numpy, math, random, os
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as pyplot


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

        return movie_medians


    def compute_genre_count(self, movie_genres):
        """ Returns genre count.

        :param movie_genres: {movie_id : set of genre ids}
        :return: {genre_id : count}
        """
        genre_count = {} # {genre_id : genre count}

        for movie_id in movie_genres.keys():
            for genre_id in movie_genres[movie_id]:

                if genre_id not in genre_count:
                    genre_count[genre_id] = 0

                genre_count[genre_id] += 1

        return genre_count


    def compute_new_movie_tags(self, movie_tags):
        """Returns a movie tags data structure that uses fewer
        tags and keeps track of number of appearances of tags.

        :param movie_tags: {movie id: list of tag ids}
        :return: {movie id: tag id: ln(count) + 1}, {tag_id: count}
        """
        # count all tags
        tag_count = {}  # {tag_id: count}

        for movie_id in movie_tags.keys():
            for tag_id in movie_tags[movie_id]:

                if tag_id not in tag_count:
                    tag_count[tag_id] = 0

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
            for tag_id in movie_tags[movie_id]:
                if tag_id in tag_id_set:
                    if movie_id not in new_movie_tags:
                        new_movie_tags[movie_id] = {}

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
        profile - [list of genre and tag ids]
        x_factors - vector of coefficients
    """

    def __init__(self, user_id, csv_data : CsvData, model_data :ModelData):

        movie_genres = csv_data.movie_genres

        movie_ids = model_data.user_ratings_train[user_id][0]
        ratings = model_data.user_ratings_train[user_id][1]

        movie_tags = model_data.movie_tags
        tag_count = model_data.tag_count
        movie_medians = model_data.movie_medians

        id_count = self.count_ids(movie_ids, movie_genres, movie_tags)
        num_factors = self.decide_num_factors(len(movie_ids))
        self.profile = self.decide_profile(id_count, num_factors)

        self.x_factors = self.compute_x_factors(movie_ids, ratings,
                                                movie_genres, movie_tags,
                                                movie_medians, tag_count)


    def count_ids(self, movie_ids : list, movie_genres, movie_tags):
        """ Counts the genres and tags encountered in this
        user's movie reviews.

        :param movie_ids: list of movie ids
        :param movie_genres: {movie_id: set of genre ids}
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1}}
        :return: {id: count}. The tag id is offset by 1000.
            So a tag id of 10 in the data set will show up as
            a tag id of 1010.
        """
        id_count = {}
        for movie_id in movie_ids:

            if movie_id in movie_genres:
                for genre_id in movie_genres[movie_id]:

                    if genre_id not in id_count:
                        id_count[genre_id] = 0

                    id_count[genre_id] += 1

            if movie_id in movie_tags:
                for tag_id in movie_tags[movie_id].keys():

                    if tag_id+1000 not in id_count:
                        id_count[tag_id+1000] = 0

                    id_count[tag_id + 1000] += 1

        return id_count


    def decide_num_factors(self, num_ratings):
        """Based on the number of ratings, return number of factors
        to use."""
        # factor profile
        # (number of factors, number of equations(ratings))
        factor_profile = [
            [5, 10], # 5 factors for the first 10 ratings
            [5, 5 * 4],
            [5, 5 * 8],
            [5, 5 * 16],
            [5, 5 * 32],
        ]

        num_factors = 0

        for factors, ratings in factor_profile:
            # determine the multiplier (x2, x4, x8, ...)
            multiplier = int(ratings / factors)

            if num_ratings <= ratings:
                return num_factors + int(num_ratings / multiplier)
            else:
                num_factors += factors
                num_ratings -= ratings

        # max limit reached
        return num_factors


    def decide_profile(self, id_count, num_factors):
        """ Decide on what factors (ids) to use for the profile.
        The most common factors (ids) are used.

        :param id_count: {id: count} shows how often was each
            id encountered.
        :param num_factors: number of factors to use
        :return: list of ids
        """
        # build id_count into a list and sort
        count_id_list = [] # list of (count, id)

        for id in id_count:
            count_id_list.append((id_count[id], id))

        # Keep just the IDs necessary. There is a bias factor
        # that is not part of the profile, that's why there is
        # a "-1" below.
        count_id_list.sort(reverse=True)
        count_id_list = count_id_list[:num_factors-1]

        id_list = [x[1] for x in count_id_list]
        return id_list


    def compute_x_factors(self, movie_ids : list, ratings,
                          movie_genres, movie_tags,
                          movie_medians, tag_count):
        """ Compute x_factors using least square fit.

        :param movie_ids: list of movie ids
        :param ratings: list of ratings
        :param movie_genres: {movie id: set of genre ids}
        :param movie_tags: {movie id: {tag id: ln(count)+1} }
        :param movie_medians: {movie id: movie median rating}
        :param tag_count: {tag id: count}
        :return: x_factors vector
        """
        # tag_count is optional - but it does make x_factors
        # more standardized

        # build up matrix A
        profile = self.profile
        num_factors = len(profile) + 1
        row = []
        col = []
        data = []

        for i in range(0, len(movie_ids)):
            movie_id = movie_ids[i]

            for j in range(0, len(profile)):
                factor_id = profile[j]

                if factor_id < 1000:
                    # factor_id is a genre id
                    if movie_id in movie_genres:
                        if factor_id in movie_genres[movie_id]:
                            row.append(i)
                            col.append(j)
                            data.append(1.0)

                else:
                    # factor_id is a tag id
                    tag_id = factor_id - 1000

                    if movie_id in movie_tags:
                        if tag_id in movie_tags[movie_id]:
                            tag_val =  movie_tags[movie_id][tag_id] / tag_count[tag_id]
                            row.append(i)
                            col.append(j)
                            data.append(tag_val)

            # there's always bias as the last factor
            row.append(i)
            col.append(num_factors - 1)
            data.append(1.0)

        m = sparse.coo_matrix((data, (row, col)),
                              shape=(len(movie_ids), num_factors),
                              dtype=numpy.double)
        A = m.tocsr()

        # build up matrix b
        b = numpy.zeros((len(movie_ids), 1), dtype=numpy.double)

        for i in range(0, len(movie_ids)):
            b[i][0] = ratings[i] - movie_medians[movie_ids[i]]

        # solve for Ax = b
        result = linalg.lsqr(A, b, iter_lim=1000)
        x = numpy.array([result[0]]).T

        return x


    def predict(self, movie_tag_scores, tag_count, movie_genre_ids,
                movie_median):
        """ Predicts a rating for the movie

        :param movie_tag_scores: {tag_id: ln(count) + 1}
        :param tag_count: {tag id: count}
        :param movie_genre_ids: a set of genre ids
        :param movie_median: median score for the movie
        :return: a rating for the movie
        """
        profile = self.profile

        # build up movie attributes vector
        attr = numpy.zeros((1, len(profile) + 1), dtype=numpy.double)

        for i in range(0, len(profile)):
            id = profile[i]

            if id < 1000:
                # id is genre id
                if movie_genre_ids is not None:
                    if id in movie_genre_ids:
                        attr[0][i] = 1.0
            else:
                # id is tag id
                tag_id = id - 1000
                if movie_tag_scores is not None:
                    if tag_id in movie_tag_scores:
                        attr[0][i] = movie_tag_scores[tag_id] / tag_count[tag_id]

        # the last term is always 1
        attr[0][len(profile)] = 1.0

        # prediction = (attr) dot (x_factors) + median
        return attr.dot(self.x_factors)[0][0] + movie_median



class Tester:
    def __init__(self, csv_data : CsvData, model_data : ModelData):
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


    def predict(self, user_profile : UserProfile, movie_id):
        movie_tag_scores = None
        if movie_id in self.model_data.movie_tags:
            movie_tag_scores = self.model_data.movie_tags[movie_id]

        tag_count = self.model_data.tag_count

        movie_genre_ids = None
        if movie_id in self.csv_data.movie_genres:
            movie_genre_ids = self.csv_data.movie_genres[movie_id]

        movie_median = self.model_data.movie_medians[movie_id]
        return user_profile.predict(movie_tag_scores, tag_count,
                                    movie_genre_ids, movie_median)


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
    pyplot.title("Prediction Ranking Agreement (LS tag model)", fontsize=14)

    average_agreement = sum(agreements) / len(agreements)
    print("Average agreement:", average_agreement)
    pyplot.show()


if __name__ == "__main__":
    main()