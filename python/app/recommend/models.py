import math, numpy, numpy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg



class Model:
    """Interface for movie prediction models."""
    def predict(self, movie_id : int):
        """Predicts a rating for the given movie_id. Returns None if
        no prediction can be made."""
        return None


    def get_param_list(self) -> list:
        """Returns a list of (param_name, param_value) tutples,
        one for each model parameter."""
        return []




# mostly same as tag_count_predictor.py
class TagCount_UserProfile(Model):
    """Model for a single user.
    ::
        User profile:
            tag_profile - {tag_id: score}
            genre_profile - {genre_id: score}
            x_factors0 - 3 variable vector for movies with tags
            x_factors1 - 2 variable vector for movies without tags

        Movie data:
            _movie_tags - {movie_id: {tag id: ln(count) + 1} }
            _movie_genres - {movie id : set of genre ids}
            _movie_medians - {movie id : movie median rating}
    """

    def __init__(self, movie_genres, movie_ratings, movie_tags,
                 tag_counts, genre_counts, movie_medians,
                 genre_ids = None, tag_ids = None):
        """
        :param movie_genres: {movie id : set of genre ids}
        :param movie_ratings: [(movie id, rating)]
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1} }
        :param tag_counts: {tag_id: count}
        :param genre_counts: {genre id : genre count}
        :param movie_medians: {movie id : movie median rating}
        :param genre_ids: {genre string : genre id} - for get_param_list(...) only
        :param tag_ids: {tag string : tag id} - for get_param_list(...) only
        """
        self._movie_tags = movie_tags
        self._movie_genres = movie_genres
        self._movie_medians = movie_medians
        self._genre_ids = genre_ids
        self._tag_ids = tag_ids

        self.tag_profile = self._compute_tag_profile(movie_ratings, movie_tags,
                                                     tag_counts)
        self.genre_profile = self._compute_genre_profile(movie_ratings,
                                                         movie_genres, genre_counts)
        self.x_factors0, self.x_factors1 = \
            self._compute_x_factors(movie_ratings, movie_tags,
                                    movie_genres, movie_medians)


    def _compute_tag_profile(self, movie_ratings, movie_tags, tag_counts):
        """Returns a tag_profile object.

        :param movie_ratings: [(movie id, rating)]
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1}}
        :param tag_counts: {tag_id: count}
        :return: {tag_id: score}
        """
        tag_profile = {} # {tag_id: score}

        # for each movie review, add to "tag_profile"
        for movie_id, rating in movie_ratings:
            offset = rating - 3

            # it's possible for a movie to be untagged
            if (offset != 0) and (movie_id in movie_tags):
                for tag_id in movie_tags[movie_id]:
                    if tag_id not in tag_profile:
                        tag_profile[tag_id] = 0

                    tag_profile[tag_id] += offset * movie_tags[movie_id][tag_id]

        # normalize the values in "tag_profile"
        for tag_id in tag_profile:
            tag_profile[tag_id] /= tag_counts[tag_id]

        return tag_profile


    def _compute_genre_profile(self, movie_ratings, movie_genres, genre_counts):
        """Returns a genre_profile object.

        :param movie_ratings: [(movie id, rating)]
        :param movie_genres: {movie_id: set of genre ids}
        :param genre_counts: {genre_id: genre count}
        :return: {genre_id: score}
        """
        genre_profile = {} # {genre_id: score}

        # for each movie review, add to "genre_profile"
        for movie_id, rating in movie_ratings:
            offset = rating - 3

            # it's possible for a movie to have no genre classified
            if (offset != 0) and (movie_id in movie_genres):
                for genre_id in movie_genres[movie_id]:
                    if genre_id not in genre_profile:
                        genre_profile[genre_id] = 0

                    genre_profile[genre_id] += offset

        # normalize the values in "genre_profile"
        for genre_id in genre_profile.keys():
            genre_profile[genre_id] /= genre_counts[genre_id]

        return genre_profile


    def _dot_product_with_movie_genre_ids(self, movie_genre_ids):
        """
        :param movie_genre_ids: a set of genre ids
        """
        dot_product = 0
        genre_profile = self.genre_profile

        for genre_id in movie_genre_ids:
            if genre_id in genre_profile:
                dot_product += genre_profile[genre_id]

        return dot_product


    def _dot_product_with_movie_tag_scores(self, movie_tag_scores):
        """
        :param movie_tag_scores: {tag_id: score}
        """
        dot_product = 0
        tag_profile = self.tag_profile

        for tag_id in movie_tag_scores:
            if tag_id in tag_profile:
                dot_product += tag_profile[tag_id] * movie_tag_scores[tag_id]

        return dot_product


    def _compute_x_factors(self, movie_ratings, movie_tags,
                           movie_genres, movie_medians):
        """ Fit this user to the Ax = b model using least squares.

        :param movie_ratings: [(movie id, rating)]
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1} }
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

        for movie_id, rating in movie_ratings:
            # each movie rating is a single row
            # the b value is always the same
            b = rating - movie_medians[movie_id]

            # tag and genre dot products
            tag_dot_product = 0
            genre_dot_product = 0

            if movie_id in movie_tags.keys():
                tag_dot_product = self._dot_product_with_movie_tag_scores(movie_tags[movie_id])

            if movie_id in movie_genres.keys():
                genre_dot_product = self._dot_product_with_movie_genre_ids(movie_genres[movie_id])

            if math.fabs(tag_dot_product > 0.1):
                # tags are present
                A0.append([tag_dot_product, genre_dot_product, 1.0])
                b0.append([b])

            # Always contribute data to the alternative model, where
            # "tag_dot_product" is not needed
            A1.append([genre_dot_product, 1.0])
            b1.append([b])

            # An alternative is to not always contribute data to A1. The
            # problem here is that if the user reviews very few movies,
            # then the lack of A1 means no "x_factors1", which then
            # means there is no fall back model for movies that have
            # no tag information.

        # least square solutions
        x_factors0 = None
        if len(A0) >= 3:
            result = numpy.linalg.lstsq(A0, b0, rcond=None)
            x_factors0 = result[0]

        x_factors1 = None
        if len(A1) >= 2:
            result = numpy.linalg.lstsq(A1, b1, rcond=None)
            x_factors1 = result[0]

        return x_factors0, x_factors1


    def predict(self, movie_id):
        """ Predicts a rating for the given movie_id. Returns None if
        no prediction can be made.
        """
        # movie median is a required value
        if movie_id not in self._movie_medians:
            return None

        movie_median = self._movie_medians[movie_id]

        # compute "tag_product" and "genre_product"
        movie_tag_scores = None # {tag_id: movie tag score}
        movie_genre_ids = None # a set of genre ids

        if movie_id in self._movie_tags:
            movie_tag_scores = self._movie_tags[movie_id]

        if movie_id in self._movie_genres:
            movie_genre_ids = self._movie_genres[movie_id]

        tag_product = 0
        genre_product = 0

        if movie_tag_scores is not None:
            tag_product = self._dot_product_with_movie_tag_scores(movie_tag_scores)

        if movie_genre_ids is not None:
            genre_product = self._dot_product_with_movie_genre_ids(movie_genre_ids)

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


    def get_param_list(self):
        genre_ids = self._genre_ids # {genre string : genre id}
        tag_ids = self._tag_ids # {tag string : tag id}

        # return empty list if "genre_ids" or "tag_ids" are not provided
        if genre_ids is None: return []
        if tag_ids is None: return []

        # generate reverse look up from "genre_ids" and "tag_ids"
        genre_names = {} # id to name look up
        for name, genre_id in genre_ids.items():
            genre_names[genre_id] = name

        tag_names = {}  # id to name look up
        for name, tag_id in tag_ids.items():
            tag_names[tag_id] = name

        param_list = []

        # add "tag_profile" to "param_list"
        tag_profile = self.tag_profile
        tag_param_list = []

        for tag_id, score in tag_profile.items():
            tag_param_list.append((tag_names[tag_id], score))

        if len(tag_param_list) > 0:
            tag_param_list.sort(key=lambda e: e[1], reverse=True)
            param_list.append(("__comment", "Tag Profile (Top 100)"))
            param_list += tag_param_list[:100]

        # add "genre_profile" to "param_list"
        genre_profile = self.genre_profile
        genre_param_list = []

        for genre_id, score in genre_profile.items():
            genre_param_list.append((genre_names[genre_id], score))

        if len(genre_param_list) > 0:
            genre_param_list.sort(key=lambda e: e[1], reverse=True)
            param_list.append(("__comment", "Genre Profile"))
            param_list += genre_param_list

        # add "x_factors0" to param_list
        x_factors0 = self.x_factors0

        # Check for x_factors0 being None, which can happen if the
        # user has only reviewed a few movies, and none of them are
        # tagged, causing inability to build the "A0" matrix in
        # "_compute_x_factors(...)".
        if x_factors0 is not None:
            param_list.append(("__comment", "For Movies with Tags"))
            param_list.append(("tag importance", x_factors0[0][0]))
            param_list.append(("genre importance", x_factors0[1][0]))
            param_list.append(("user bias", x_factors0[2][0]))

        # add "x_factors1" to param_list
        x_factors1 = self.x_factors1

        # Check for x_factors1 being None. This can only happen if
        # an alternative approach to training x_factors1 is used.
        # Right now all data is used to train x_factors1. But if
        # only movies without tags are used to train x_factors1, and
        # all movies in the training set has tags, then
        # "_compute_x_factors(...)" cannot build the "A1" matrix,
        # causing x_factors1 to be None.
        if x_factors1 is not None:
            param_list.append(("__comment", "For Movies without Tags"))
            param_list.append(("genre importance", x_factors1[0][0]))
            param_list.append(("user bias", x_factors1[1][0]))

        return param_list




# mostly same as tag_ls_predictor.py
class TagLS_UserProfile(Model):
    """Model for a single user.
    ::
        User profile:
            profile - [list of genre and tag ids]
            x_factors - vector of coefficients
        Movie data:
            _movie_tags - {movie_id: {tag_id: ln(count) + 1} }
            _tag_counts - {tag id: count}
            _movie_genres - {movie id : set of genre ids}
            _movie_medians - {movie id : movie median rating}
    """

    def __init__(self, movie_genres, movie_ratings, movie_tags, tag_counts,
                 movie_medians, genre_ids=None, tag_ids=None):
        """Constructs a user profile.

        :param movie_genres: {movie id : set of genre ids}
        :param movie_ratings: [(movie id, rating)]
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1} }
        :param tag_counts: {tag id: count}
        :param movie_medians: {movie id : movie median rating}
        :param genre_ids: {genre string : genre id} - for get_param_list(...) only
        :param tag_ids: {tag string : tag id} - for get_param_list(...) only
        """
        self._movie_tags = movie_tags
        self._tag_counts = tag_counts
        self._movie_genres = movie_genres
        self._movie_medians = movie_medians
        self._genre_ids = genre_ids
        self._tag_ids = tag_ids

        id_counts = self._count_ids(movie_ratings, movie_genres, movie_tags)
        num_factors = self._decide_num_factors(len(movie_ratings))
        self.profile = self._decide_profile(id_counts, num_factors)

        self.x_factors = self._compute_x_factors(
            movie_ratings, movie_genres, movie_tags, movie_medians,
            tag_counts)


    def _count_ids(self, movie_ratings : list, movie_genres, movie_tags):
        """ Counts the genres and tags encountered in this
        user's movie ratings.

        :param movie_ratings: [(movie id, rating)]
        :param movie_genres: {movie_id: set of genre ids}
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1}}
        :return: {id: count}. The tag id is offset by 1000.
            So a tag id of 10 in the data set will show up as
            a tag id of 1010.
        """
        id_counts = {}
        for movie_id, _ in movie_ratings:

            if movie_id in movie_genres:
                for genre_id in movie_genres[movie_id]:

                    if genre_id not in id_counts:
                        id_counts[genre_id] = 0

                    id_counts[genre_id] += 1

            if movie_id in movie_tags:
                for tag_id in movie_tags[movie_id].keys():

                    if tag_id+1000 not in id_counts:
                        id_counts[tag_id+1000] = 0

                    id_counts[tag_id + 1000] += 1

        return id_counts


    def _decide_num_factors(self, num_ratings):
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


    def _decide_profile(self, id_counts, num_factors):
        """ Decide on which factors (ids) to use for the profile.
        The most common factors (ids) are used.

        :param id_counts: {id: count} shows how often was each
            genre or tag encountered. If the id is < 1000, it's a genre id.
            If id is >= 1000, it's a tag id.
        :param num_factors: number of factors to use
        :return: list of ids
        """
        # build id_count into a list and sort
        count_id_list = [] # list of (count, id)

        for id in id_counts:
            count_id_list.append((id_counts[id], id))

        # Keep just the IDs necessary. There is a bias factor
        # that is not part of the profile, that's why there is
        # a "-1" below.
        count_id_list.sort(reverse=True)
        count_id_list = count_id_list[:num_factors]

        id_list = [x[1] for x in count_id_list]
        return id_list


    def _compute_x_factors(self, movie_ratings, movie_genres,
                           movie_tags, movie_medians, tag_counts):
        """ Compute x_factors using least square fit.

        :param movie_ratings: [(movie id, rating)]
        :param movie_genres: {movie id: set of genre ids}
        :param movie_tags: {movie id: {tag id: ln(count)+1} }
        :param movie_medians: {movie id: movie median rating}
        :param tag_counts: {tag id: count}
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

        for i in range(0, len(movie_ratings)):
            movie_id = movie_ratings[i][0]

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
                            tag_val = movie_tags[movie_id][tag_id] / tag_counts[tag_id]
                            row.append(i)
                            col.append(j)
                            data.append(tag_val)

            # there's always bias as the last factor
            row.append(i)
            col.append(num_factors - 1)
            data.append(1.0)

        m = sparse.coo_matrix((data, (row, col)),
                              shape=(len(movie_ratings), num_factors),
                              dtype=numpy.double)
        A = m.tocsr()

        # build up matrix b
        b = numpy.zeros((len(movie_ratings), 1), dtype=numpy.double)

        for i in range(0, len(movie_ratings)):
            movie_id, rating = movie_ratings[i]
            b[i][0] = rating - movie_medians[movie_id]

        # solve for Ax = b
        result = linalg.lsqr(A, b, iter_lim=1000)
        x = numpy.array([result[0]]).T

        # To detect failure to converge to a solution:
        # if numpy.sum(numpy.abs(x)) < 0.01:

        return x


    def predict(self, movie_id : int):
        """Predicts a rating for the given movie_id. Returns None if
        no prediction can be made."""
        # param movie_tag_scores: {tag id: ln(count) + 1}
        #         :param tag_counts: {tag id: count}
        #         :param movie_genre_ids: a set of genre ids
        #         :param movie_median: median score for the movie

        # movie median is required
        if movie_id not in self._movie_medians:
            return None
        movie_median = self._movie_medians[movie_id]

        # "movie_tag_scores" and "movie_genre_ids" are optional
        movie_tag_scores = None
        if movie_id in self._movie_tags:
            movie_tag_scores = self._movie_tags[movie_id]

        movie_genre_ids = None
        if movie_id in self._movie_genres:
            movie_genre_ids = self._movie_genres[movie_id]

        profile = self.profile # [list of genre and tag ids]

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
                        attr[0][i] = movie_tag_scores[tag_id] / self._tag_counts[tag_id]

        # the last term is always 1
        attr[0][len(profile)] = 1.0

        # prediction = (attr) dot (x_factors) + median
        return attr.dot(self.x_factors)[0][0] + movie_median


    def get_param_list(self):
        genre_ids = self._genre_ids # {genre string : genre id}
        tag_ids = self._tag_ids # {tag string : tag id}

        # return empty list if "genre_ids" or "tag_ids" are not provided
        if genre_ids is None: return []
        if tag_ids is None: return []

        # generate reverse look up from "genre_ids" and "tag_ids"
        genre_names = {} # id to name look up
        for name, genre_id in genre_ids.items():
            genre_names[genre_id] = name

        tag_names = {}  # id to name look up
        for name, tag_id in tag_ids.items():
            tag_names[tag_id] = name

        profile = self.profile
        x_factors = self.x_factors

        # build up genres and tags as separate lists
        genre_param_list = []
        tag_param_list = []

        for i in range(0, len(profile)):
            profile_id = profile[i]
            factor = x_factors[i][0]

            if profile[i] < 1000:
                # profile[i] is genre id
                genre_param_list.append((genre_names[profile_id], factor))
            else:
                tag_param_list.append((tag_names[profile_id - 1000], factor))

        param_list = []

        if len(genre_param_list) > 0:
            genre_param_list.sort(key=lambda e: e[1], reverse=True)
            param_list.append(("__comment", "Genre Parameters"))
            param_list += genre_param_list

        if len(tag_param_list) > 0:
            tag_param_list.sort(key=lambda e: e[1], reverse=True)
            param_list.append(("__comment", "Tag Parameters"))
            param_list += tag_param_list

        return param_list




# Not the same as "als_predictor.py"
class ALS_Model(Model):
    """ALS Model for a single user.
    ::
        User profile:
            user_factors - numpy array of factors for this user only
            _valid - False if the user did not rate enough movies
        Movie information:
            _movie_medians - {movie id : movie median rating}
            _als_movie_factors - numpy array of movie factors for all movies
            _als_movie_ids - {standard movie id : zero based movie id}
    """

    def __init__(self, num_factors : int, movie_ratings : list,
                 movie_medians : dict, als_movie_factors : numpy.ndarray,
                 als_movie_ids : dict):
        """Create an ALS_Model for a particular user.

        :param num_factors: number of factors per movie
        :param movie_ratings: [(movie id, rating)]
        :param movie_medians: {movie id : movie median rating}
        :param als_movie_factors: numpy array of type double
        :param als_movie_ids: {standard movie id : zero based movie id}
        """
        self._valid = False

        # basic check - you need at least num_factor + 1 movies
        if len(movie_ratings) < num_factors + 1: return

        # build up "A" and "b" in "Ax=b"
        A = []
        b = []
        for movie_id, rating in movie_ratings:
            # The (movie_id, rating) can only be used if "movie_id" is in
            # "als_movie_ids"
            if movie_id in als_movie_ids:
                als_movie_id = als_movie_ids[movie_id]

                # retrieve the movie factors
                index = num_factors * als_movie_id
                movie_factors = als_movie_factors[index : index + num_factors]

                # add a row to A and to b
                row = list(movie_factors)
                row.append(1)

                A.append(row)
                b.append(rating)

        # check that A has enough rows
        if len(A) < num_factors + 1: return

        # A(user_factors) = b
        self.user_factors = numpy.linalg.lstsq(A, b, rcond=None)[0]

        self._movie_medians = movie_medians
        self._als_movie_factors = als_movie_factors
        self._als_movie_ids = als_movie_ids

        self._valid = True


    def is_valid(self):
        return self._valid


    def predict(self, movie_id : int):
        """Predicts a rating for the given movie_id. Returns None if
        no prediction can be made."""

        if self._valid == False: return None

        # Information about "movie_id" must exist
        if (movie_id not in self._movie_medians) \
            or (movie_id not in self._als_movie_ids):
            return None

        # get "movie_median" and "movie_factors"
        movie_median = self._movie_medians[movie_id]

        num_item_factors = len(self.user_factors) - 1
        als_movie_id = self._als_movie_ids[movie_id]
        movie_factors = self._als_movie_factors[num_item_factors * als_movie_id:
                                                num_item_factors * (als_movie_id + 1)]

        # compute rating
        rating = 0
        for i in range(0, num_item_factors):
            rating += self.user_factors[i] * movie_factors[i]

        rating += self.user_factors[num_item_factors]
        rating += movie_median

        return rating


    def get_param_list(self):
        user_factors = self.user_factors
        param_list = []

        for i in range(0, len(user_factors) - 1):
            param_list.append(("factor " + str(i), user_factors[i]))

        # the very last parameter is user bias
        param_list.append(("user bias", user_factors[-1]))

        return param_list


