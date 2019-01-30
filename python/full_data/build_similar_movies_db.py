"""
Command line flags:
    overwrite - sets the overwrite flag, overwriting existing .bin files
        default is False

    cpu_count=4 - sets number of processes to use. This only applies to
        building the similarity database locally, using multiprocessing.
        If using cluster, then the number of CPU is the number of
        processes spawned by the worker_server.py, and this cpu_count
        parameter has no effect. Default is None.

Output file:
    similar_movies.bin - {movie_id: [similar_movie_ids]}
"""

import datetime, math, numpy, os, pickle, sys, time
import cluster, config, movie_lens_data, my_util
import movie_lens_data_proc as _proc


class SimilarMovieFinder:
    """ Class for finding similar movies.
    ::
        movie_genres - {movie id: set of genre ids}
        movie_ratings - [(movie_id, {user_id: rating})]
        buff_limit - maximum boost to score
        buff_point - number of common reviewers at max boost
    """

    def __init__(self, movie_genres, movie_ratings, buff_limit = 0.05,
                 buff_point = 100):
        """
        :param movie_genres: {movie id: set of genre ids}
        :param movie_ratings: [(movie_id, {user_id: rating})]
        :param buff_limit: maximum boost to score
        :param buff_point: number of common reviewers at max boost
        """
        self.movie_genres = movie_genres
        self.movie_ratings = movie_ratings
        self.buff_limit = buff_limit
        self.buff_point = buff_point


    def _genres_similar(self, movie_id1, movie_id2):
        """
        :return: True if the genres are similar.
        """
        movie_genres = self.movie_genres

        if movie_id1 not in movie_genres: return False
        if movie_id2 not in movie_genres: return False

        genre1 = movie_genres[movie_id1]
        genre2 = movie_genres[movie_id2]

        # have genre1 be the shorter genre set
        if len(genre1) > len(genre2):
            genre1, genre2 = genre2, genre1

        length = len(genre1)
        matches = 0

        # count genre matches
        for genre_id in genre1:
            if genre_id in genre2: matches += 1

        # genre match is True if 50% or higher match
        if matches / length >= 0.5: return True
        else: return False


    def _scaled_dot_product(self, movie_id1_index, movie_id2_index):
        """Computes (dot product) similarity of two
        movies' user reviews. The dot product is
        scaled for movies that have large number
        of common users.
        ::
            returns final_score, common_reviewers, pre_boost_score
        """
        ratings1 = self.movie_ratings[movie_id1_index][1]
        ratings2 = self.movie_ratings[movie_id2_index][1]

        # have ratings1 be the movie with fewer reviews
        if len(ratings1) > len(ratings2):
            ratings1, ratings2 = ratings2, ratings1

        # look for common users first
        r1 = []
        r2 = []

        for user_id in ratings1:
            if user_id in ratings2:
                r1.append(ratings1[user_id])
                r2.append(ratings2[user_id])

        # if there are too few common users, return 0 (no similarity)
        if len(r1) < 3: return 0.0, len(r1), 0.0

        r1 = numpy.array(r1)
        r2 = numpy.array(r2)

        norm1 = numpy.linalg.norm(r1)
        norm2 = numpy.linalg.norm(r2)

        similarity = r1.dot(r2) / (norm1 * norm2)

        # Scale output due to number of common users.
        buff_limit = self.buff_limit
        buff_point = self.buff_point
        n = len(r1) # number of common users

        x_limit = 3 * math.exp(buff_limit)
        x = 3 + (x_limit - 3) * (n - 3) / (buff_point - 3)
        buff = math.log(x) - math.log(3)

        if buff > buff_limit: buff = buff_limit # for input > buff_point
        if buff < 0: buff = 0 # for input < 3, which shouldn't happen

        return similarity * (1.0 + buff), n, similarity


    def _compare_two_movies(self, movie_id1_index, movie_id2_index):
        """Returns score, common reviewers
        ::
            score - a value that represents the similarity
                between movie_id1 and movie_id2.
        """
        movie_id1 = self.movie_ratings[movie_id1_index][0]
        movie_id2 = self.movie_ratings[movie_id2_index][0]

        if self._genres_similar(movie_id1, movie_id2) == False:
            return 0.0, 0

        score, common_reviewers, _ = self._scaled_dot_product(movie_id1_index,
                                                              movie_id2_index)
        return score, common_reviewers


    def find_movie_index(self, movie_id: int):
        """Return the "movie_ratings" list index for movie_id. If
        the movie_id cannot be found in "movie_ratings", return -1."""
        movie_ratings = self.movie_ratings
        for i in range(0, len(movie_ratings)):
            if movie_id == movie_ratings[i][0]:
                return i

        return -1



    def find_similar_movie(self, movie_id_index : int, num_results = 20):
        """Returns movie_ids, similarity_scores."""

        sim_scores = [] # list of (movie_id, score, num_reviewers)
        movie_ratings = self.movie_ratings

        for i in range(0, len(movie_ratings)):
            if i != movie_id_index:
                score, num_reviewers = self._compare_two_movies(movie_id_index, i)

                if score > 0.3:
                    sim_scores.append((movie_ratings[i][0], score, num_reviewers))

        # keep only top (num_results * 20) movies, sorted by num_reviewers
        if len(sim_scores) > num_results * 20:
            sim_scores.sort(key=lambda e: e[2], reverse=True)
            sim_scores = sim_scores[:num_results * 20]

        # sort according to score
        sim_scores.sort(key = lambda e: e[1], reverse=True)

        # return just the "movie_ids" and "scores"
        if len(sim_scores) > 0:
            movie_ids, scores, _ = zip(*sim_scores)
            movie_ids = movie_ids[:num_results]
            scores = scores[:num_results]

            return movie_ids, scores
        else:
            return [], []


    def tune(self, movie_id1, movie_id2, top_n, expected_search_size):
        """Tune "buff_limit" and "buff_point", such that when
        searching for movies similar to "movie_id1", "movie_id2"
        will show up in "top_n".
        ::
            expected_search_size - the application's usual "num_results"
                parameter in the find_similar_movie(...) call
        """

        # buff_point is the number of reviewers common to id1 and id2
        index1 = self.find_movie_index(movie_id1)
        index2 = self.find_movie_index(movie_id2)

        final_score, common_reviewers, pre_boost_score = \
            self._scaled_dot_product(index1, index2)

        self.buff_point = common_reviewers

        # reset buff_limit, then increase it until it "movie_id2" is in "top_n"
        self.buff_limit = 0

        while self.buff_limit < 2:
            movie_ids, scores = self.find_similar_movie(
                index1, num_results = expected_search_size * 2)

            movie_ids = movie_ids[:top_n]

            # look for movie_id2 in results, exit if found
            for movie_id in movie_ids:
                if movie_id == movie_id2: return

            # movie_id2 not found in results; boost movie_id2 relative to
            # the top result
            top_score = scores[0]

            score, common_reviewers, pre_boost_score = \
                self._scaled_dot_product(index1, index2)

            self.buff_limit = self.buff_limit * top_score / score + 0.01


def print_similar_movies(movie_id : int, similar_movies, movie_titles):
    """
    ::
        similar_movies - the "similar_movies.bin" from out directory
        movie_titles - the "movie_titles.bin" from in directory
    """
    movie_ids = similar_movies[movie_id]
    for movie_id in movie_ids:
        print(movie_titles[movie_id])


class SimilarMovieDB:
    """A class for testing "similar_movies.bin" after it has been generated.
    This class assumes the existence of "similar_movies.bin" and
    "movie_titles.bin"."""
    def __init__(self):
        self.similar_movies = movie_lens_data.get_output_obj("similar_movies")
        movie_lens_data.read_movies_csv()
        self.movie_titles = movie_lens_data.get_input_obj("movie_titles")

    def print_similar_movies(self, movie_id):
        if movie_id in self.similar_movies:
            print()
            print('Movies similar to "' + self.movie_titles[movie_id] + '":')

            for similar_movie_id in self.similar_movies[movie_id]:
                print(self.movie_titles[similar_movie_id])
        else:
            print("No similar movie found")


def build_locally(movie_genres, movie_ratings, buff_point, buff_limit,
                  similar_movies_file_name, cpu_count):
    _proc.start_processes(cpu_count)

    # "movie_ratings" should have been shuffled when it's created
    # _proc.shuffle_list(movie_ratings)

    # distribute data to all processes
    _proc.send_same_data({
        "movie_genres": movie_genres,
        "movie_ratings": movie_ratings,
        "buff_point": buff_point,
        "buff_limit": buff_limit
    })

    # set up the indices that each process will work on
    _proc.split_range_and_send(0, len(movie_ratings), "movie_ratings")

    print("Current time", datetime.datetime.now())
    _proc.run_function("_find_similar_movies", {})

    # merge results
    similar_movies = _proc.update_var_into_dict("similar_movies")

    print("Saving", similar_movies_file_name)
    with open(similar_movies_file_name, mode="wb") as file:
        pickle.dump(similar_movies, file)

    _proc.end_processes()

    # print run time
    run_time = int(time.time() - movie_lens_data.start_time)
    print("Total run time", datetime.timedelta(seconds=run_time))
    print('\a')


def build_with_cluster(buff_point, buff_limit, length,
                       output_file_name):

    setup_param = {"buff_point": buff_point,
                   "buff_limit": buff_limit}

    cluster.send_command({
        "op": "distribute",
        "worker_op": "build_similar_movies_db",
        "setup_param": setup_param,
        "length": length
    })

    cluster.wait_for_completion()
    cluster.print_status()
    print()

    cluster.merge_list_results_into_dict(output_file_name)
    print("similar_movies.bin has been built")
    print('\a')


def main():
    # process command line arguments
    overwrite = False
    cpu_count = None

    for arg in sys.argv:
        if arg == "overwrite":
            overwrite = True
        elif arg.startswith("cpu_count="):
            cpu_count = int(arg.split(sep='=')[1])

    similar_movies_file_name = config.out_dir + "similar_movies.bin"

    # exit if object already exists
    if overwrite == False and os.path.exists(similar_movies_file_name):
        print(similar_movies_file_name, "already exists")
        return

    movie_lens_data.start_time = time.time()

    # The SimilarMovieFinder class is tuned using:
    # movie id 1196 - Star Wars: Episode V - The Empire Strikes Back
    # movie id 1210 - Star Wars: Episode VI - Return of the Jedi
    movie_id1 = 1196
    movie_id2 = 1210

    # check movie_id titles - to be sure that the movie_id is still valid
    movie_lens_data.read_movies_csv()
    movie_titles = movie_lens_data.get_input_obj("movie_titles")

    if movie_titles[movie_id1].lower().find("empire strikes back") < 0:
        print('Movie ID', movie_id1, 'no longer "empire strikes back",',
              "cannot continue")
        return

    if movie_titles[movie_id2].lower().find("return of the jedi") < 0:
        print('Movie ID', movie_id2, 'no longer "return of the jedi",',
              "cannot continue")
        return

    # create SimilarMovieFinder and tune
    movie_ratings = movie_lens_data.create_movie_ratings(overwrite)
    movie_genres = movie_lens_data.get_input_obj("movie_genres")
    movie_finder = SimilarMovieFinder(movie_genres, movie_ratings)
    movie_finder.tune(movie_id1, movie_id2, 2, 20)

    print("SimilarMovieFinder tuning results in buff_point =",
          movie_finder.buff_point, "buff_limit =",
          movie_finder.buff_limit)

    print()
    print("Movies similar to \"" + movie_titles[movie_id1] + "\":")

    movie_id1_index = movie_finder.find_movie_index(movie_id1)
    movie_ids, _ = movie_finder.find_similar_movie(movie_id1_index)

    for movie_id in movie_ids:
        print(movie_titles[movie_id])

    # build database of similar movies
    print()
    print("Starting to build database of similar movies")

    if cluster.cluster_info is None:
        build_locally(
            movie_genres, movie_ratings, movie_finder.buff_point,
            movie_finder.buff_limit, similar_movies_file_name, cpu_count)
    else:
        build_with_cluster(movie_finder.buff_point, movie_finder.buff_limit,
                           len(movie_ratings), "similar_movies.bin")




if __name__ == "__main__":
    main()


