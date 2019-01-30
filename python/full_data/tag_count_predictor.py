import math, numpy
import cluster, movie_lens_data, my_util



class UserProfile(my_util.Model):
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
                 tag_counts, genre_counts, movie_medians):
        """
        :param movie_genres: {movie id : set of genre ids}
        :param movie_ratings: [(movie id, rating)]
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1} }
        :param tag_counts: {tag_id: count}
        :param genre_counts: {genre id : genre count}
        :param movie_medians: {movie id : movie median rating}
        """
        self._movie_tags = movie_tags
        self._movie_genres = movie_genres
        self._movie_medians = movie_medians

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



def main():
    length = movie_lens_data.get_input_obj("user_ratings_test_length")

    # Sends a "tag_count_eval" command to cluster nodes.
    if cluster.cluster_info is None:
        print("Cannot connect to cluster.")
        return

    cluster.send_command({
        "op": "distribute",
        "worker_op": "tag_count_eval",
        "length": length
    })

    cluster.wait_for_completion()
    cluster.print_status()
    print()

    file_name = "tag_count_eval_results.bin"
    results = cluster.merge_list_results(file_name)
    user_ids, agreements = zip(*results)

    my_util.print_rank_agreement_results(agreements, "tag counting")
    print('\a')


if __name__ == "__main__":
    main()

