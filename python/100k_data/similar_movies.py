"""
This module computes similarity between two movies.
"""
import csv, numpy, math, os


data_dir = "data" + os.sep



class CsvData:
    """This class contains data read from csv files.
    ::
        genre_ids - {genre string: genre id}
        movie_genres - {movie id: set of genre ids}
        movie_ratings - {movie id: user id: rating}
        movie_titles - {movie id: title string}
    """

    def __init__(self):
        self.genre_ids, self.movie_genres, self.movie_titles = self.read_movies_csv()
        self.movie_ratings = self.read_ratings_csv()


    def read_movies_csv(self):
        """ Read "movies.csv". Returns genre_str_to_id,
        movie_id_to_genre_ids, movie_id_to_title.
        ::
            genre_str_to_id - {genre string:  genre id}
            movie_id_to_genre_ids - {movie id: set of genre ids}
            movie_id_to_title - {movie id: title string}
        """
        genre_str_to_id = {} # {genre string: genre id}
        next_genre_id = 0

        movie_id_to_genre_ids = {} # {movie id: set of genre ids}

        movie_id_to_title = {} # # {movie_id: title string}

        # read from "movies.csv" to collect genres
        with open(data_dir + "movies.csv", "r", encoding="utf-8", newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)  # skip the first row

            for row in csv_reader:
                movie_id = int(row[0])
                title = row[1]
                genre_strings = row[2].lower().split('|')

                movie_id_to_title[movie_id] = title

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

        return genre_str_to_id, movie_id_to_genre_ids, movie_id_to_title


    def read_ratings_csv(self):
        """ Reads the "ratings.csv" file. Returns movie_ratings.
        ::
            movie_ratings - {movie_id: user_id: rating}
        """

        movie_ratings = {}

        with open(data_dir + "ratings.csv", "r", encoding="utf-8", newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)  # skip the first row

            for row in csv_reader:
                user_id = int(row[0])
                movie_id = int(row[1])
                rating = float(row[2])

                if movie_id not in movie_ratings:
                    movie_ratings[movie_id]  = {}

                movie_ratings[movie_id][user_id] = rating

        return movie_ratings



class SimilarMovieFinder:
    """ Class for finding similar movies.
    ::
        csv_data
    """

    def __init__(self, csv_data : CsvData):
        self.csv_data = csv_data


    def genres_similar(self, movie_id1, movie_id2):
        """
        :return: True if the genres are similar.
        """
        movie_genres = self.csv_data.movie_genres

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
        for id in genre1:
            if id in genre2: matches += 1

        # genre match is True if 50% or higher match
        if matches / length >= 0.5: return True
        else: return False


    def scaled_dot_product(self, movie_id1, movie_id2, verbose=False):
        """Computes (dot product) similarity of two
        movies' user reviews. The dot product is
        scaled for movies that have large number
        of common users.
        """
        ratings1 = self.csv_data.movie_ratings[movie_id1]
        ratings2 = self.csv_data.movie_ratings[movie_id2]

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
        if len(r1) < 3: return 0.0

        r1 = numpy.array(r1)
        r2 = numpy.array(r2)

        norm1 = numpy.linalg.norm(r1)
        norm2 = numpy.linalg.norm(r2)

        similarity = r1.dot(r2) / (norm1 * norm2)

        # Scale output due to number of common users.
        # The settings below buff score by 21.6% for having 162 users in common
        buff_limit = 0.216
        buff_point = 162.0
        n = len(r1) # number of common users

        # for tuning the above parameters, you need to know common users "n"
        if verbose:
            print("similarity =", similarity,
                  "common reviewers =", n)

        x_limit = 3 * math.exp(buff_limit)
        x = 3 + (x_limit - 3) * (n - 3) / (buff_point - 3)
        buff = math.log(x) - math.log(3)

        if buff > buff_limit: buff = buff_limit # for input > buff_point
        if buff < 0: buff = 0 # for input < 3, which shouldn't happen

        return similarity * (1.0 + buff)


    def compare_two_movies(self, movie_id1, movie_id2):
        """
        :return: A value that represents the similarity
        between movie_id1 and movie_id2.
        """
        if self.genres_similar(movie_id1, movie_id2) == False:
            return 0.0

        return self.scaled_dot_product(movie_id1, movie_id2)


    def find_similar_movie(self, movie_id, num_results = 10):
        """Returns a list of (movie id, similarity score) pairs,
        containing movies that are similar to the given
        movie_id."""

        sim_scores = []

        for id2 in self.csv_data.movie_ratings:
            if id2 != movie_id:
                score = self.compare_two_movies(movie_id, id2)

                if score > 0.3:
                    sim_scores.append((id2, score))

        sim_scores.sort(key = lambda e: e[1], reverse=True)

        return sim_scores[:num_results]





def main():
    csv_data = CsvData()
    movie_titles = csv_data.movie_titles

    movie_finder = SimilarMovieFinder(csv_data)

    movie_id = 1196 # Star Wars: Episode V - The Empire Strikes Back (1980)

    # id 1210 is Star Wars: Episode VI - Return of the Jedi (1983)
    movie_finder.scaled_dot_product(movie_id, 1210, verbose=True)

    print("Searching for movies similar to id #",
          movie_id, ", ", movie_titles[movie_id])

    sim_movies = movie_finder.find_similar_movie(movie_id, num_results=100)

    print("Score".center(10))
    for movie_id, score in sim_movies:
        print("{0:.4g}".format(score).center(10), movie_titles[movie_id])




if __name__ == "__main__":
    main()
