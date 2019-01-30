import csv, datetime, gc, math, os, pickle, random, time

import config, cpp_ls
import movie_lens_data_proc as _proc


shared_directory = config.shared_directory

txt_dir = config.txt_dir
in_dir = config.in_dir
als_dir = config.als_dir
out_dir = config.out_dir
temp_out_dir = config.temp_out_dir

start_time = time.time()



def get_input_obj(object_name):
    """Use pickle to load "object_name" from "in_dir"."""
    file_name = in_dir + object_name + ".bin"

    with open(file_name, mode="rb") as file:
        obj = pickle.load(file)

    return obj


def get_als_obj(object_name):
    """Use pickle to load "object_name" from "als_dir"."""
    file_name = als_dir + object_name + ".bin"

    with open(file_name, mode="rb") as file:
        obj = pickle.load(file)

    return obj


def get_output_obj(object_name):
    """Use pickle to load "object_name" from "out_dir"."""
    file_name = out_dir + object_name + ".bin"

    with open(file_name, mode="rb") as file:
        obj = pickle.load(file)

    return obj


def _current_time():
    """Returns the time since "start_time"."""
    run_time = int(time.time() - start_time)
    return datetime.timedelta(seconds=run_time)


def _create_dirs():
    """Create various directories if they don't already exist"""
    if os.path.exists(in_dir) == False: os.mkdir(in_dir)
    if os.path.exists(als_dir) == False: os.mkdir(als_dir)
    if os.path.exists(out_dir) == False: os.mkdir(out_dir)
    if os.path.exists(temp_out_dir) == False: os.mkdir(temp_out_dir)


def read_links_csv(overwrite = False):
    """Reads "links.csv" and creates "links.bin". Returns links.
    ::
        overwrite - if set to True, overwrite existing .bin files
        links.bin - {movie_id: [imdb_str, tmdb_str] }
    """
    links_file_name = in_dir + "links.bin"
    if overwrite == False:
        if os.path.exists(links_file_name):
            return get_input_obj("links")

    print(_current_time(), "Reading links.csv")
    links = {}

    # read from "links.csv" to collect link information
    with open(txt_dir + "links.csv", "r", encoding="utf-8", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip the first row

        for row in csv_reader:
            movie_id = int(row[0])

            imdb_str = None
            if len(row) >= 2 and len(row[1]) > 0:
                imdb_str = row[1]

            tmdb_str = None
            if len(row) >= 3 and len(row[2]) > 0:
                tmdb_str = row[2]

            links[movie_id] = [imdb_str, tmdb_str]

    print(_current_time(), "Saving", links_file_name)
    with open(links_file_name, mode="wb") as file:
        pickle.dump(links, file)

    return links


def read_movies_csv(overwrite = False):
    """ Reads "movies.csv" and creates "genre_counts.bin",
    "genre_ids.bin",  "movie_genres.bin", and "movie_titles.bin".
    ::
        overwrite - if set to True, overwrite existing .bin files
        genre_counts.bin - {genre id : genre count}
        genre_ids.bin - {genre string : genre id}
        movie_genres.bin - {movie id : set of genre ids}
        movie_titles.bin - {movie_id: title}
    """
    genre_counts_file_name = in_dir + "genre_counts.bin"
    genre_ids_file_name = in_dir + "genre_ids.bin"
    movie_genres_file_name = in_dir + "movie_genres.bin"
    movie_titles_file_name = in_dir + "movie_titles.bin"

    # early exit if objects already exist
    if overwrite == False:
        if os.path.exists(genre_counts_file_name) and \
            os.path.exists(genre_ids_file_name) and \
            os.path.exists(movie_genres_file_name) and \
            os.path.exists(movie_titles_file_name):
            return

    print(_current_time(), "Reading movies.csv")

    genre_ids = {} # {genre string : genre id}
    next_genre_id = 0

    movie_genres = {} # {movie id : set of genre ids}
    movie_titles = {} # {movie_id: title}

    # read from "movies.csv" to collect genres
    with open(txt_dir + "movies.csv", "r", encoding="utf-8", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip the first row

        for row in csv_reader:
            movie_id = int(row[0])
            genre_strings = row[2].lower().split('|')

            movie_titles[movie_id] = row[1]

            # go through each genre found
            for genre_str in genre_strings:
                if genre_str != "(no genres listed)":

                    # add "genre_str" to "genre_ids"
                    if genre_str not in genre_ids:
                        genre_ids[genre_str] = next_genre_id
                        next_genre_id += 1

                    genre_id = genre_ids[genre_str]

                    # add genre_id to "movie_genres"
                    if movie_id not in movie_genres:
                        movie_genres[movie_id] = set()

                    movie_genres[movie_id].add(genre_id)

    print(_current_time(), "Saving", genre_ids_file_name, movie_genres_file_name,
          movie_titles_file_name)

    with open(genre_ids_file_name, mode="wb") as file:
        pickle.dump(genre_ids, file)

    with open(movie_genres_file_name, mode="wb") as file:
        pickle.dump(movie_genres, file)

    with open(movie_titles_file_name, mode="wb") as file:
        pickle.dump(movie_titles, file)

    # build up genre counts
    genre_counts = {} # {genre id : genre count}
    for movie_id in movie_genres:
        for genre_id in movie_genres[movie_id]:
            if genre_id not in genre_counts: genre_counts[genre_id] = 0
            genre_counts[genre_id] += 1

    print(_current_time(), "Saving", genre_counts_file_name)
    with open(genre_counts_file_name, mode="wb") as file:
        pickle.dump(genre_counts, file)

    return


def create_movie_ratings(overwrite = False):
    """Reads the "ratings.csv" file and creates "movie_ratings.bin".
    Returns movie_ratings.
    ::
        overwrite - if set to True, overwrite existing .bin files
        movie_ratings.bin - [ (movie_id, {user_id: rating}) ]
    """
    movie_ratings_file_name = in_dir + "movie_ratings.bin"

    # check for file existence
    if overwrite == False:
        if os.path.exists(movie_ratings_file_name):
            print(_current_time(), 'Reading in "movie_ratings.bin"')
            return get_input_obj("movie_ratings")

    movie_ratings = []

    print(_current_time(), "Reading ratings.csv")

    with open(txt_dir + "ratings.csv", "r", encoding="utf-8", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip the first row

        movie_id_to_list_index = {}

        for row in csv_reader:
            user_id = int(row[0])
            movie_id = int(row[1])
            rating = float(row[2])

            # add (user_id, movie_id, rating) to "movie_ratings"
            if movie_id in movie_id_to_list_index:
                index = movie_id_to_list_index[movie_id]
                movie_ratings[index][1][user_id] = rating
            else:
                movie_id_to_list_index[movie_id] = len(movie_ratings)
                movie_ratings.append((movie_id, {user_id: rating}))

    # The structure of "movie_ratings" by default is that older movies
    # have lower IDs, and they have more reviews.
    # Shuffle "movie_ratings" so its distributed work load is more even.
    for i in range(0, len(movie_ratings)):
        swap_index = random.randint(0, len(movie_ratings) - 1)
        if swap_index != i:
            movie_ratings[i], movie_ratings[swap_index] = \
                movie_ratings[swap_index], movie_ratings[i]

    print(_current_time(), "Saving", movie_ratings_file_name)
    with open(movie_ratings_file_name, mode="wb") as file:
        pickle.dump(movie_ratings, file)

    return movie_ratings


def create_user_ratings(overwrite = False):
    """Reads the "ratings.csv" file and creates "user_ratings.bin".
    Returns user_ratings.
    ::
        overwrite - if set to True, overwrite existing .bin files
        user_ratings.bin - [ (user_id, [(movie_id, rating)] ) ]
    """
    user_ratings_file_name = in_dir + "user_ratings.bin"

    # check for file existence
    if overwrite == False:
        if os.path.exists(user_ratings_file_name):
            print(_current_time(), 'Reading in "user_ratings.bin"')
            return get_input_obj("user_ratings")

    user_ratings = []

    print(_current_time(), "Reading ratings.csv")

    with open(txt_dir + "ratings.csv", "r", encoding="utf-8", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip the first row

        user_id_to_list_index = {}
        current_list_index = None
        current_user_id = None  # data tends to have same user ids together

        for row in csv_reader:
            user_id = int(row[0])
            movie_id = int(row[1])
            rating = float(row[2])

            # add (user_id, movie_id, rating) to "user_ratings"
            if user_id == current_user_id:
                # usual case - user ids seem clustered together
                user_ratings[current_list_index][1].append((movie_id, rating))

            elif user_id in user_id_to_list_index:
                # look up index
                current_list_index = user_id_to_list_index[user_id]
                current_user_id = user_id

                user_ratings[current_list_index][1].append((movie_id, rating))

            else:
                # new user id encountered
                current_list_index = len(user_ratings)
                current_user_id = user_id
                user_id_to_list_index[user_id] = current_list_index

                user_ratings.append((user_id, [(movie_id, rating)]))

    print(_current_time(), "Saving", user_ratings_file_name)
    with open(user_ratings_file_name, mode="wb") as file:
        pickle.dump(user_ratings, file)

    return user_ratings


def read_tags_csv(tag_min_appearance : int, overwrite = False):
    """Reads "tags.csv" and creates "movie_tags.bin",
    "tag_counts.bin", "tag_names.bin", and "tag_ids.bin".
    Only tags that are used at least "tag_min_appearance"
    times are used.
    ::
        movie_tags.bin - {movie_id: {tag_id: ln(count) + 1} }
        tag_counts.bin - {tag_id: count}
        tag_names.bin - {tag id: tag string}
        tag_ids.bin - {tag string : tag id}
    """
    movie_tags_file_name = in_dir + "movie_tags.bin"
    tag_counts_file_name = in_dir + "tag_counts.bin"
    tag_names_file_name = in_dir + "tag_names.bin"
    tag_ids_file_name = in_dir + "tag_ids.bin"

    # check for file existence
    if overwrite == False:
        if os.path.exists(movie_tags_file_name) and \
            os.path.exists(tag_counts_file_name) and \
            os.path.exists(tag_names_file_name) and \
            os.path.exists(tag_ids_file_name): return

    print(_current_time(), "Reading tags.csv")

    tag_ids = {} # {tag string : tag id}
    next_tag_id = 0

    data = [] # [(movie id, tag id)]

    tag_counts = {} # {tag id: count}

    # read from "tags.csv" to collect tags
    with open(txt_dir + "tags.csv", "r", encoding="utf-8", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip the first row

        for row in csv_reader:
            movie_id = int(row[1])
            tag = row[2].lower()

            # detect new tags
            if tag not in tag_ids:
                tag_ids[tag] = next_tag_id
                tag_counts[next_tag_id] = 0
                next_tag_id += 1

            # data, tag_counts
            tag_id = tag_ids[tag]
            tag_counts[tag_id] += 1
            data.append((movie_id, tag_id))

    # filter out tags that do not satisfy "tag_min_appearance"
    # reduce entries in "tag_counts"
    old_tag_counts = tag_counts
    tag_counts = {}

    for tag_id in old_tag_counts:
        count = old_tag_counts[tag_id]
        if count >= tag_min_appearance:
            tag_counts[tag_id] = count

    # reduce entries in "tag_ids", build "tag_names"
    old_tag_ids = tag_ids
    tag_ids = {}
    tag_names = {} # {tag id: tag string}

    for tag_name in old_tag_ids:
        tag_id = old_tag_ids[tag_name]
        if tag_id in tag_counts:
            tag_ids[tag_name] = tag_id
            tag_names[tag_id] = tag_name

    # build first version of "movie_tags"
    movie_tags = {} # {movie_id: {tag_id: count} }
    for movie_id, tag_id in data:
        if tag_id in tag_counts:
            # add (movie_id, tag_id) to movie_tags
            if movie_id not in movie_tags:
                movie_tags[movie_id] = {tag_id: 0}
            if tag_id not in movie_tags[movie_id]:
                movie_tags[movie_id][tag_id] = 0

            movie_tags[movie_id][tag_id] += 1

    # change "movie_tags" count to "ln(count) + 1"
    for movie_id in movie_tags:
        for tag_id in movie_tags[movie_id]:
            count = movie_tags[movie_id][tag_id]
            movie_tags[movie_id][tag_id] = math.log(count) + 1

    # save objects to file
    print(_current_time(), "Saving", movie_tags_file_name)
    with open(movie_tags_file_name, mode="wb") as file:
        pickle.dump(movie_tags, file)

    print(_current_time(), "Saving", tag_counts_file_name)
    with open(tag_counts_file_name, mode="wb") as file:
        pickle.dump(tag_counts, file)

    print(_current_time(), "Saving", tag_names_file_name)
    with open(tag_names_file_name, mode="wb") as file:
        pickle.dump(tag_names, file)

    print(_current_time(), "Saving", tag_ids_file_name)
    with open(tag_ids_file_name, mode="wb") as file:
        pickle.dump(tag_ids, file)


def refresh_training_sets_mp(user_ratings, training_set_ratio = 0.80):
    """Refresh training and testing sets. Also computes movie medians
    from the training set. Returns movie_medians_train.
    ::
        File Outputs:
            user_ratings_train.bin - [ (user id, [(movie id, rating)] ) ]
            user_ratings_test.bin - [ (user id, [(movie id, rating)] ) ]
            user_ratings_test_length.bin - length of "user_ratings_test"
            movie_medians_train.bin - {movie id : movie median rating}

        In process memory upon exit:
            user_ratings_train - [ (user id, [(movie id, rating)] ) ]
            user_ratings_test - [ (user id, [(movie id, rating)] ) ]

        Function return:
            movie_medians_train - {movie id : movie median rating}
    """
    print(_current_time(), "Splitting data into training and testing sets.")

    # Split "user_ratings" to different processes. Run "_compute_training_set".
    # Retrieve "user_ratings_train" and "user_ratings_test".
    _proc.split_list_and_send(user_ratings, "user_ratings")

    _proc.run_function("_compute_training_set",
                       {"training_set_ratio": training_set_ratio})

    user_ratings_train = _proc.concat_var_into_list("user_ratings_train")
    user_ratings_test = _proc.concat_var_into_list("user_ratings_test")

    # save "user_ratings_train" and "user_ratings_test" to file
    print(_current_time(), 'Saving "user_ratings_train" and "user_ratings_test"')

    file_name = in_dir + "user_ratings_train.bin"
    with open(file_name, mode="wb") as file:
        pickle.dump(user_ratings_train, file)

    file_name = in_dir + "user_ratings_test.bin"
    with open(file_name, mode="wb") as file:
        pickle.dump(user_ratings_test, file)

    file_name = in_dir + "user_ratings_test_length.bin"
    with open(file_name, mode="wb") as file:
        pickle.dump(len(user_ratings_test), file)

    # The next goal is to get movie medians from the training set
    # Run "_extract_movie_ratings" on each training set. Then merge
    # the movie ratings from the different processes. Split the
    # movie ratings to different processes and have them compute
    # the median.
    _proc.run_function("_extract_movie_ratings", {})
    movie_ratings_lists = _proc.append_var_into_list("movie_ratings")
    movie_ratings = _merge_movie_ratings_lists(movie_ratings_lists)
    _proc.split_list_and_send(movie_ratings, "movie_ratings")
    _proc.run_function("_compute_medians", {})

    movie_medians = _proc.update_var_into_dict("movie_medians")

    # save "movie_medians" to file
    print(_current_time(), 'Saving "movie_medians_train"')

    file_name = in_dir + "movie_medians_train.bin"
    with open(file_name, mode="wb") as file:
        pickle.dump(movie_medians, file)

    # free memory
    _proc.delete_variable("user_ratings") # probably not needed in future
    _proc.delete_variable("movie_ratings") # intermediate result only
    _proc.delete_variable("movie_medians") # intermediate result only

    # user_ratings_train, user_ratings_test remain in memory

    return movie_medians


def _merge_movie_ratings_lists(movie_ratings_lists):
    """Merge multiple movie_ratings_lists produced by
    different processors. Side effect - this call
    modifies some of the data in "movie_ratings_lists".

    :param movie_ratings_lists: Multiple list, where each list
        is [(movie_id, [movie ratings])].
    :return: A single list of [(movie_id, [movie ratings])].
    """
    # Side effect - this call modifies some of the data
    # in "movie_ratings_lists".
    # Suppose movie 1 appears in movie_ratings_lists[0]
    # first. That particular "ratings" structure is being
    # appended to include all movie ratings.

    # merge the movie_ratings_lists
    indices = [0 for _ in range(0, len(movie_ratings_lists))]
    # one index per list, it starts at zero and goes to the
    # length of the list

    movie_ratings = []  # (movie_id, [ratings list])

    while True:
        # look for the smallest movie_id
        smallest_movie_id = None
        smallest_list = None  # list index with the smallest movie id

        for i in range(0, len(movie_ratings_lists)):
            j = indices[i]
            # i picks which list, j is how far into that list

            if j < len(movie_ratings_lists[i]):
                movie_id = movie_ratings_lists[i][j][0]

                if smallest_list is None:
                    smallest_movie_id = movie_id
                    smallest_list = i

                elif smallest_movie_id > movie_id:
                    smallest_movie_id = movie_id
                    smallest_list = i

        if smallest_list is None:
            break

        j = indices[smallest_list]
        ratings = movie_ratings_lists[smallest_list][j][1]
        indices[smallest_list] += 1

        # merge (smallest_movie_id, ratings) into movie_ratings
        if len(movie_ratings) == 0:
            movie_ratings.append((smallest_movie_id, ratings))

        elif movie_ratings[-1][0] == smallest_movie_id:
            # movie_ratings[-1][1] = movie_ratings[-1][1] + ratings
            x = movie_ratings[-1][1]
            x += ratings

        else:
            movie_ratings.append((smallest_movie_id, ratings))

    return movie_ratings


def als_data_set_shrink_mp(movie_medians_train, factors_list,
                           no_test_set = False):
    """The "factors" are number of item factors.
    This function assumes "user_ratings_train" and
    "user_ratings_test" are still in process memory. Drop off
    users who have too few movie ratings from both the training
    set and the testing set. Drop off movies that have too
    few users from the training set.
    ::
        no_test_set - all of the data is being used for training. The
        output does not have a test set.
    """
    print(_current_time(), "Shrinking training data to satisfy ALS requirements.")

    # ALS model coverage information
    num_users = [] # number of users for each factor
    num_movies = [] # number of movies for each factor

    # distribute movie median information to all processors
    _proc.send_same_data({"movie_medians": movie_medians_train})

    for factor in factors_list:
        has_changed = True
        while has_changed:
            # Each user must have at least (factor+1) movie reviews
            # Drop users that have less than (factor+1) movie reviews
            _proc.run_function("_drop_users", { "min_ratings": factor + 1 })
            has_changed = _proc.or_var_into_boolean("has_changed")

            # Each movie must have at least (factor) number of ratings
            _proc.run_function("_count_movies", {})
            movie_counts = _proc.add_merge_var_into_dict("movie_counts")

            # Go through "movie_counts" to see which one has fewer than
            # (factor) number of ratings
            uncommon_movies = set()
            for movie_id in movie_counts:
                if movie_counts[movie_id] < factor:
                    uncommon_movies.add(movie_id)

            # Drop movies that have fewer than (factor) reviews.
            if len(uncommon_movies) > 0:
                has_changed = True
                _proc.run_function("_drop_movies",
                                   { "movies_to_drop": uncommon_movies })

        # build lookup table for zero based user and movie ids
        _proc.run_function("_collect_ids", {})
        movie_ids = _proc.update_var_into_set("movie_ids")
        user_ids = _proc.update_var_into_set("user_ids")

        # movie_ids and user_ids are sets of ids
        # go through these sets to build a list of zero based ids

        als_movie_ids = {} # {standard movie id : zero based movie id}

        i = 0
        for movie_id in movie_ids:
            als_movie_ids[movie_id] = i
            i += 1

        als_user_ids = {}  # {standard user id : zero based user id}
        i = 0
        for user_id in user_ids:
            als_user_ids[user_id] = i
            i += 1

        # save id tables to disk
        print(_current_time(), 'Saving "als_user_ids" and "als_movie_ids"'
              + " for ALS factor " + str(factor))

        file_name = als_dir + "als" + str(factor) + "_user_ids.bin"
        with open(file_name, mode="wb") as file:
            pickle.dump(als_user_ids, file)

        file_name = als_dir + "als" + str(factor) + "_movie_ids.bin"
        with open(file_name, mode="wb") as file:
            pickle.dump(als_movie_ids, file)

        # update ALS coverage statistics
        num_users.append(len(als_user_ids))
        num_movies.append(len(als_movie_ids))

        # send these ID tables to each process. Build up numpy arrays
        # representing the training data. The median is subtracted off
        # "ratings_train_numpy" inside "_convert_training_data_to_numpy(...)"
        gc.collect()
        _proc.send_same_data({
            "als_movie_ids": als_movie_ids,
            "als_user_ids": als_user_ids
        })

        _proc.run_function("_convert_training_data_to_numpy", {})

        # merge numpy arrays and save to disk
        user_ids_train_numpy = _proc.concat_var_into_numpy_array("user_ids_train_numpy")
        movie_ids_train_numpy = _proc.concat_var_into_numpy_array("movie_ids_train_numpy")
        ratings_train_numpy = _proc.concat_var_into_numpy_array("ratings_train_numpy")

        print(_current_time(), 'Saving "user_ratings_train" and "user_ratings_test"'
              + " for ALS factor " + str(factor))

        file_name = als_dir + "als" + str(factor) + "_user_ratings_train.bin"

        with open(file_name, mode="wb") as file:
            pickle.dump([user_ids_train_numpy, movie_ids_train_numpy,
                         ratings_train_numpy], file)

        if no_test_set:
            # remove test set information if they are found on disk
            file_name = als_dir + "als" + str(factor) + "_user_ratings_test.bin"
            if os.path.exists(file_name): os.remove(file_name)

            file_name = als_dir + "als" + str(factor) + "_user_ratings_test_length.bin"
            if os.path.exists(file_name): os.remove(file_name)

        else:
            # get test data and save to disk
            user_ratings_test = _proc.concat_var_into_list("user_ratings_test")

            file_name = als_dir + "als" + str(factor) + "_user_ratings_test.bin"
            with open(file_name, mode="wb") as file:
                pickle.dump(user_ratings_test, file)

            # save length of test data to disk
            file_name = als_dir + "als" + str(factor) + "_user_ratings_test_length.bin"
            with open(file_name, mode="wb") as file:
                pickle.dump(len(user_ratings_test), file)

    # print ALS model coverage information
    print("".ljust(10), "Users".center(15), "Movies".center(15))
    for i in range(0, len(factors_list)):
        model_name = "ALS " + str(factors_list[i])
        print(model_name.ljust(10), str(num_users[i]).center(15), str(num_movies[i]).center(15))



def als_train(factors_list, thread_count = None, algorithm=1):
    """Trains ALS models for each factor. Each factor is a model
    with its own "user_factors" and "item_factors". """
    if thread_count is not None:
        cpp_ls.set_thread_count(thread_count)

    for factor in factors_list:
        # load the necessary data from disk, then call C++ als(...)
        num_items = len(get_als_obj("als" + str(factor) + "_movie_ids"))
        num_users = len(get_als_obj("als" + str(factor) + "_user_ids"))
        user_ids_train, item_ids_train, ratings_train = get_als_obj(
            "als" + str(factor) + "_user_ratings_train")

        print(_current_time(), "Building ALS factor", factor, "model")

        user_factors, item_factors, iterations = cpp_ls.als(
            user_ids_train, item_ids_train, ratings_train, factor,
            num_users, num_items, algorithm=algorithm)

        print(_current_time(), "ALS took", iterations, "iterations.",
              'Saving "user_factors" and "item_factors" to disk')

        # save user_factors and item_factors to disk
        file_name = als_dir + "als" + str(factor) + "_user_factors.bin"
        with open(file_name, mode="wb") as file:
            pickle.dump(user_factors, file)

        file_name = als_dir + "als" + str(factor) + "_item_factors.bin"
        with open(file_name, mode="wb") as file:
            pickle.dump(item_factors, file)




if __name__ == "__main__":
    pass
else:
    # this file is being loaded via "import"
    _create_dirs()

