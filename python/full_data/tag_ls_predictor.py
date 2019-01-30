import numpy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

import cluster, movie_lens_data, my_util



class UserProfile(my_util.Model):
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
                 movie_medians):
        """Constructs a user profile.

        :param movie_genres: {movie id : set of genre ids}
        :param movie_ratings: [(movie id, rating)]
        :param movie_tags: {movie_id: {tag_id: ln(count) + 1} }
        :param tag_counts: {tag id: count}
        :param movie_medians: {movie id : movie median rating}
        """
        self._movie_tags = movie_tags
        self._tag_counts = tag_counts
        self._movie_genres = movie_genres
        self._movie_medians = movie_medians

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




def main():
    length = movie_lens_data.get_input_obj("user_ratings_test_length")

    # Sends a "tag_ls_eval" command to cluster nodes.
    if cluster.cluster_info is None:
        print("Cannot connect to cluster.")
        return

    cluster.send_command({
        "op": "distribute",
        "worker_op": "tag_ls_eval",
        "length": length
    })

    cluster.wait_for_completion()
    cluster.print_status()
    print()

    file_name = "tag_ls_eval_results.bin"
    results = cluster.merge_list_results(file_name)
    user_ids, agreements = zip(*results)

    my_util.print_rank_agreement_results(agreements, "tag least squares")
    print('\a')


if __name__ == "__main__":
    main()
