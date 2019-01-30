import numpy
import cluster, movie_lens_data, my_util



class ALS_Model(my_util.Model):
    """ALS Model for a single user.
    ::
        User profile:
            user_factors - numpy array of factors for this user only
        Movie information:
            _movie_medians - {movie id : movie median rating}
            _als_movie_factors - numpy array of movie factors for all movies
            _als_movie_ids - {standard movie id : zero based movie id}

    """

    def __init__(self, user_factors : numpy.ndarray, movie_medians,
                 als_movie_factors : numpy.ndarray, als_movie_ids
                 ):
        """Create an ALS_Model for a particular user.

        :param user_factors: numpy array of type double
        :param movie_medians: {movie id : movie median rating}
        :param als_movie_factors: numpy array of type double
        :param als_movie_ids: {standard movie id : zero based movie id}
        """
        self.user_factors = user_factors
        self._movie_medians = movie_medians
        self._als_movie_factors = als_movie_factors
        self._als_movie_ids = als_movie_ids



    def predict(self, movie_id : int):
        """Predicts a rating for the given movie_id. Returns None if
        no prediction can be made."""

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




def main():
    # file name will start with "als_prefix", which looks like "als3_"
    file_name_suffix = "eval_results.bin"

    # keep track of average and number of data points for each test
    average_agreement = []
    num_data_points = []

    factor_list = [3, 5, 7, 9, 11]

    for factor in factor_list:
        als_prefix = "als" + str(factor) + "_"

        # issue "als_eval" test for the current factor and print results
        length = movie_lens_data.get_als_obj(als_prefix + "user_ratings_test_length")

        # sends an "als_eval" command to cluster nodes.
        if cluster.cluster_info is None:
            print("Cannot connect to cluster.")
            return

        cluster.send_command({
            "op": "distribute",
            "worker_op": "als_eval",
            "setup_param": factor,
            "length": length
        })

        cluster.wait_for_completion()
        cluster.print_status()
        print()

        results = cluster.merge_list_results(als_prefix + file_name_suffix)
        user_ids, agreements = zip(*results)

        my_util.print_rank_agreement_results(agreements, "ALS " + str(factor))
        print()

        # update "average_agreement" and "num_data_points"
        average_agreement.append(sum(agreements) / len(agreements))
        num_data_points.append(len(agreements))

    # print a summary table for all factors
    print("factor".center(10), "average agreement".center(25),
          "number of values".center(20))

    for i in range(0, len(factor_list)):
        print(str(factor_list[i]).center(10),
              str(average_agreement[i]).center(25),
              str(num_data_points[i]).center(20))

    print('\a')


if __name__ == "__main__":
    main()

