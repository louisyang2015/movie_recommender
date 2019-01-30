"""
Multi processor support for "movie_lens_data.py".
"""

import datetime, math, multiprocessing, numpy, random, time
import build_similar_movies_db, my_util

_cpu_count = multiprocessing.cpu_count()

_processes = [] # child processes
_pipes = [] # pipes to child processes

_process_data = {} # data to survive in between function calls
_lock = None # a multiprocessing.Lock shared by all processes


def _message_loop(conn, lock):
    """
    :param conn: a multiprocessing.connection.PipeConnection
    :param lock: a multiprocessing.Lock shared by all processes
    """
    global _lock
    _lock = lock

    while True:
        request = conn.recv()
        op = request["op"]

        if op == "shutdown":
            return

        elif op == "set_data":
            for key in request:
                if key != "op":
                    _process_data[key] = request[key]

        elif op == "get_var":
            var_name = request["var_name"]
            conn.send(_process_data[var_name])

        elif op == "del_var":
            var_name = request["var_name"]
            del _process_data[var_name]

        elif op == "clear_all_data":
            _process_data.clear()

        elif op == "run_function":
            op_name = request["op_name"]

            if op_name in globals():
                globals()[op_name](request)
            else:
                print("Unhandled operation:", op)
                print("op_name:", op_name)
                # print("Request =", request) - request might be too large
        else:
            print("Unhandled operation:", op)
            # print("Request =", request) - request might be too large


def start_processes(cpu_count = None):
    """Spawns (cpu_count - 1) processes and pipes."""
    global _lock
    _lock = multiprocessing.Lock()

    if cpu_count is not None:
        global _cpu_count
        _cpu_count = cpu_count

    for i in range(0, _cpu_count - 1):
        pipe1, pipe2 = multiprocessing.Pipe()
        process = multiprocessing.Process(target = _message_loop,
                                          args=(pipe2, _lock))
        process.start()
        _processes.append(process)
        _pipes.append(pipe1)


def end_processes():
    """Send "shutdown" message to all processes and wait for
    them to terminate."""
    for pipe in _pipes:
        pipe.send({"op": "shutdown"})

    for process in _processes:
        process.join()


def send_same_data(data_dict):
    """Send same data to all processes. The data_dict
    is the request object itself, so the "op" key
    cannot be used."""

    # for the local process:
    for key in data_dict:
        _process_data[key] = data_dict[key]

    data_dict["op"] = "set_data"

    for pipe in _pipes:
        pipe.send(data_dict)


def split_list_and_send(data_list, var_name : str):
    """Split "data_list" and send the pieces to each process. The
    current process gets the final split."""
    # compute split
    indices_and_lengths = my_util.split(0, len(data_list), _cpu_count)

    # send to other processes
    for i in range(0, len(_pipes)):
        index = indices_and_lengths[i][0]
        length = indices_and_lengths[i][1]

        if length > 0:
            _pipes[i].send({
                "op": "set_data",
                var_name: data_list[index : index + length]
            })

    # the current process gets the final split
    index = indices_and_lengths[-1][0]
    _process_data[var_name] = data_list[index:]


def split_range_and_send(start_index : int, length : int, range_name : str):
    """For example, given start_index = 0, length = 10, and _cpu_count = 4,
    the four processes will get [(0, 2), (2, 3), (5, 2), (7, 3)].
    The starting index, such as [0, 2, 5, 7], will show up as
    "range_name_start". The length value, such as [2, 3, 2, 3], will show
    up as "range_name_length". The current process gets the final split.
    """
    # compute the split
    indices_and_lengths = my_util.split(start_index, length, _cpu_count)

    # send to other processes
    for i in range(0, len(_pipes)):
        start = indices_and_lengths[i][0]
        length = indices_and_lengths[i][1]

        _pipes[i].send({
            "op": "set_data",
            range_name + "_start": start,
            range_name + "_length": length
        })

    # the current process gets the final split
    _process_data[range_name + "_start"] = indices_and_lengths[-1][0]
    _process_data[range_name + "_length"] = indices_and_lengths[-1][1]


def delete_variable(var_name):
    del _process_data[var_name]
    for pipe in _pipes:
        pipe.send({
            "op": "del_var",
            "var_name": var_name
        })


def clear_all_data():
    for pipe in _pipes:
        pipe.send({"op": "clear_all_data"})

    _process_data.clear()


def run_function(function_name : str, arg_dict = None):
    """Run "function_name", with arg_dict being the
    argument dictionary."""
    if arg_dict is None:
        arg_dict = {}

    # The keys "op" and "op_name" will be over written - so these
    # should not be present in "arg_dict".
    if ("op" in arg_dict) or ("op_name" in arg_dict):
        raise Exception("run_function() cannot accept arg_dict with "
                        + 'keys "op" or "op_name".')

    # arg_dict is used as the request
    arg_dict["op"] = "run_function"
    arg_dict["op_name"] = function_name

    for pipe in _pipes:
        pipe.send(arg_dict)

    # for local process
    globals()[function_name](arg_dict)


def concat_var_into_list(var_name : str):
    """Retrieve "var_name" from all processes and
    use list concatenate to combine them into
    a single list."""
    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    var_list = []
    for pipe in _pipes:
        var_list += pipe.recv()

    var_list += _process_data[var_name]
    return var_list


def append_var_into_list(var_name : str):
    """Retrieve "var_name" from all processes and
    use list.append() to combine them into a
    single list."""
    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    var_list = []
    for pipe in _pipes:
        var_list.append(pipe.recv())

    var_list.append(_process_data[var_name])
    return var_list


def concat_var_into_numpy_array(var_name : str):
    """Retrieve "var_name" from all processes and
    use numpy.concatenate(...) to combine them into
    a single numpy array."""
    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    var_list = []
    for pipe in _pipes:
        var_list.append(pipe.recv())

    var_list.append(_process_data[var_name])
    return numpy.concatenate(var_list)


def update_var_into_set(var_name : str):
    """Retrieve "var_name" sets from
    all processes and use set.update()
    to combine them into a single set."""
    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    var_set = _process_data[var_name].copy()

    for pipe in _pipes:
        var_set.update(pipe.recv())

    return var_set


def update_var_into_dict(var_name : str):
    """Retrieve "var_name" dictionaries from
    all processes and use dictionary.update()
    to combine them into a single dictionary."""
    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    var_dict = {}
    for pipe in _pipes:
        var_dict.update(pipe.recv())

    var_dict.update(_process_data[var_name])
    return var_dict


def add_merge_var_into_dict(var_name : str):
    """Retrieve "var_name" dictionaries from
    all processes and merge these dictionaries
    together by adding common keys."""
    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    var_dict = _process_data[var_name].copy()

    for pipe in _pipes:
        new_dict = pipe.recv()

        for key in new_dict:
            if key in var_dict:
                var_dict[key] += new_dict[key]
            else:
                var_dict[key] = new_dict[key]

    return var_dict


def or_var_into_boolean(var_name : str):
    """Retrieve "var_name" boolean from all
    processes and "or" them into a single boolean."""
    var_flag = _process_data[var_name]

    for pipe in _pipes:
        pipe.send({
            "op": "get_var",
            "var_name": var_name
        })

    for pipe in _pipes:
        # The following does not work.
        #   var_flag = var_flag or pipe.recv()
        # When "var_flag" is already true, the above expression
        # short circuits, and "pipe.recv()" does not get executed.
        # A future call to "pipe.recv()" will then get a boolean
        # that's supposed to be read right now.
        flag = pipe.recv()
        var_flag = var_flag or flag

    return var_flag


# End of framework
##########################################################



def _compute_training_set(request):
    """ Split "user_ratings" into a training set and a testing set.
    ::
        Input:
            _process_data["user_ratings"] = [ (user id, [(movie id, rating)] ) ]
            request keys: {"training_set_ratio"}

        Result:
            _process_data["user_ratings_train"]
            _process_data["user_ratings_test"]
    """
    user_ratings = _process_data["user_ratings"]
    training_set_ratio = request["training_set_ratio"]

    if training_set_ratio == 1:
        _process_data["user_ratings_train"] = user_ratings
        _process_data["user_ratings_test"] = None
        return

    user_ratings_train = [] # [ (user_id, [(movie_id, rating)] ) ]
    user_ratings_test = []

    # the test set has to have at least two ratings
    min_length = math.ceil(2 / (1 - training_set_ratio))

    for user_entry in user_ratings:
        user_id = user_entry[0]
        movie_ratings = user_entry[1]

        if len(movie_ratings) > min_length:
            split_index = int(len(movie_ratings) * training_set_ratio)

            # attempt to create a training set for at most five times
            for i in range(0, 5):
                _shuffle(movie_ratings, split_index)

                # the test set must have at least two different ratings
                if my_util.has_different_ratings(movie_ratings, split_index):
                    user_ratings_train.append((user_id, movie_ratings[:split_index]))
                    user_ratings_test.append((user_id, movie_ratings[split_index:]))
                    break

    _process_data["user_ratings_train"] = user_ratings_train
    _process_data["user_ratings_test"] = user_ratings_test


def _shuffle(val_list, start : int):
    """Shuffles the content of "val_list", starting at
    index "start"."""
    length = len(val_list)

    for i in range(start, length):
        swap_index = random.randint(0, length - 1)

        if i != swap_index:
            val_list[i], val_list[swap_index] = val_list[swap_index], val_list[i]


def _extract_movie_ratings(request):
    """Extract movie ratings from the training set.
    ::
        Input:
            _process_data["user_ratings_train"] = [ (user id, [(movie id, rating)] ) ]
        Result:
            _process_data["movie_ratings"] = [(movie_id, [ratings list])]
    """
    user_ratings_train = _process_data["user_ratings_train"]

    movie_ratings = [] # [(movie_id, [ratings list])]
    movie_id_to_index = {} # {movie_id : list index}

    for user_entry in user_ratings_train:
        for movie_id, rating in user_entry[1]:

            if movie_id in movie_id_to_index:
                # append "rating" to existing entry
                index = movie_id_to_index[movie_id]
                movie_ratings[index][1].append(rating)

            else:
                # new (movie_id, [ratings list]) entry
                index = len(movie_ratings)
                movie_id_to_index[movie_id] = index
                movie_ratings.append((movie_id, [rating]))


    movie_ratings.sort() # so to merge with other movie_ratings

    _process_data["movie_ratings"] = movie_ratings


# def _group_list_by_col(tuple_list, col = 0):
#     """Given a list of tuples, return a new list that
#     is grouped by column number "col".
#     ::
#         Example: input = [(0, 100, "a"), (0, 101, "b"), (1, 200, "c")]
#         group by column 0 means the function will return
#         [(0, [ (100, "a"), (101, "b") ] ),
#          (1, [ (200, "c") ] )]
#     """
#     tuple_list.sort(key = lambda e: e[col])
#
#     results = []
#     current_val = None
#
#     for tuple_val in tuple_list:
#         key_val = tuple_val[col]
#         tuple_val_dropped = tuple_val[:col] + tuple_val[col + 1:]
#
#         if key_val != current_val:
#             # new entry in results
#             current_val = key_val
#             results.append((current_val, [tuple_val_dropped]))
#         else:
#             # append current tuple_val_dropped to last value in results
#             results[-1][1].append(tuple_val_dropped)
#
#     return results


def _compute_medians(request):
    """Go through "movie_ratings" and return a dictionary of movie
    medians.
    ::
        Input:
            _process_data["movie_ratings"] = [(movie id, [movie ratings list])]
        Result:
            _process_data["movie_medians"] = {movie id : movie median rating}
    """
    movie_ratings = _process_data["movie_ratings"]
    movie_medians = {}

    for movie_id, ratings in movie_ratings:
        median = numpy.median(ratings)
        movie_medians[movie_id] = median

    _process_data["movie_medians"] = movie_medians


def _compute_medians2(request):
    """Go through "movie_ratings" and return a dictionary of movie
    medians. The format of "movie_ratings" is different from
    "_compute_medians(request)".
    ::
        Input:
            _process_data["movie_ratings"] = [ (movie_id, {user_id: rating}) ]
        Result:
            _process_data["movie_medians"] = {movie id : movie median rating}
    """
    movie_ratings = _process_data["movie_ratings"]
    movie_medians = {}

    for movie_id, ratings in movie_ratings:
        median = numpy.median(list(ratings.values()))
        movie_medians[movie_id] = median

    _process_data["movie_medians"] = movie_medians


def _drop_users(request):
    """Drop users who have less than "min_ratings" in the training set,
    from both the training set and testing set. The testing set can
    be None.
    ::
        Input:
            _process_data["user_ratings_train"] = [ (user id, [(movie id, rating)] ) ]
            _process_data["user_ratings_test"] = [ (user id, [(movie id, rating)] ) ]
            request keys = { "min_ratings" }
        Result:
            _process_data["user_ratings_train"] possibly changed
            _process_data["user_ratings_test"] possibly changed
            _process_data["has_changed"] = True or False
    """
    min_ratings = request["min_ratings"]
    has_changed = False

    # check the data first
    user_ratings_train = _process_data["user_ratings_train"]
    user_ratings_test = _process_data["user_ratings_test"]
    for user_id, ratings in user_ratings_train:
        if len(ratings) < min_ratings:
            has_changed = True
            break

    _process_data["has_changed"] = has_changed

    if has_changed:
        # build new "_train" and "_test" data structures
        train2 = []
        test2 = []
        for i in range(0, len(user_ratings_train)):
            if len(user_ratings_train[i][1]) >= min_ratings:
                train2.append(user_ratings_train[i])

                if user_ratings_test is not None:
                    test2.append(user_ratings_test[i])

        _process_data["user_ratings_train"] = train2

        if user_ratings_test is not None:
            _process_data["user_ratings_test"] = test2


def _count_movies(request):
    """Count how many ratings each movie has.
        Input:
            _process_data["user_ratings_train"] = [ (user id, [(movie id, rating)] ) ]
        Result:
            _process_data["movie_counts"] = {movie id: count}
    """
    user_ratings_train = _process_data["user_ratings_train"]
    movie_counts = {} # {movie id: count}

    # go through "user_ratings_train" to build up "movie_counts"
    for user_id, movie_ratings in user_ratings_train:
        for movie_id, rating in movie_ratings:
            if movie_id not in movie_counts:
                movie_counts[movie_id] = 0

            movie_counts[movie_id] += 1

    _process_data["movie_counts"] = movie_counts


def _drop_movies(request):
    """Drop movies from "user_ratings_train".
    ::
        Input:
            _process_data["user_ratings_train"] = [ (user id, [(movie id, rating)] ) ]
            request = { "movies_to_drop": set of movie ids }
        Result:
            _process_data["user_ratings_train"] will not contain movie ids
                mentioned in "movies_to_drop".
    """
    user_ratings_train = _process_data["user_ratings_train"]
    movies_to_drop = request["movies_to_drop"]

    for user_id, movie_ratings in user_ratings_train:

        # first see if there are movies that need to be dropped
        need_mod = False
        for movie_id, rating in movie_ratings:
            if movie_id in movies_to_drop:
                need_mod = True
                break

        if need_mod:
            old_list = movie_ratings.copy()
            movie_ratings.clear()
            for movie_id, rating in old_list:
                if movie_id not in movies_to_drop:
                    movie_ratings.append((movie_id, rating))


def _collect_ids(request):
    """Collect movie and user ids from "user_ratings_train" into sets.
    ::
        Input:
            _process_data["user_ratings_train"] = [ (user id, [(movie id, rating)] ) ]
        Result:
            _process_data["movie_ids"] = set of movie ids
            _process_data["user_ids"] = set of user ids
    """
    user_ratings_train = _process_data["user_ratings_train"]
    movie_ids = set()
    user_ids = set()

    for user_id, movie_ratings in user_ratings_train:
        for movie_id, rating in movie_ratings:
            movie_ids.add(movie_id)
            user_ids.add(user_id)

    _process_data["movie_ids"] = movie_ids
    _process_data["user_ids"] = user_ids


def _convert_training_data_to_numpy(request):
    """Convert from "user_ratings_train" into numpy arrays. The median is
    also subtracted off "ratings_train_numpy" in this function.
    ::
        Input:
            _process_data["user_ratings_train"] = [ (user id, [(movie id, rating)] ) ]
            _process_data["als_movie_ids"] = {standard movie id : zero based movie id}
            _process_data["als_user_ids"] = {standard user id : zero based user id}
            _process_data["movie_medians"] = {movie id : movie median rating}
        Result:
            _process_data["user_ids_train_numpy"] = numpy array of type int32
            _process_data["movie_ids_train_numpy"] = numpy array of type int32
            _process_data["ratings_train_numpy"] = numpy array of type double
    """
    user_ratings_train = _process_data["user_ratings_train"]
    als_movie_ids = _process_data["als_movie_ids"]
    als_user_ids = _process_data["als_user_ids"]
    movie_medians = _process_data["movie_medians"]

    # find out how many entries are there in "user_ratings_train"
    length = 0
    for _, movie_ratings in user_ratings_train:
        length += len(movie_ratings)

    user_ids_train_numpy = numpy.zeros(length, dtype=numpy.int32)
    movie_ids_train_numpy = numpy.zeros(length, dtype=numpy.int32)
    ratings_train_numpy = numpy.zeros(length, dtype=numpy.double)

    index = 0

    for user_id, movie_ratings in user_ratings_train:
        for movie_id, rating in movie_ratings:
            # copy (user_id, movie_id, rating) into numpy arrays
            user_ids_train_numpy[index] = als_user_ids[user_id]
            movie_ids_train_numpy[index] = als_movie_ids[movie_id]

            # the rating for the ALS procedure needs to have "median" subtracted off
            ratings_train_numpy[index] = rating - movie_medians[movie_id]

            index += 1

    _process_data["user_ids_train_numpy"] = user_ids_train_numpy
    _process_data["movie_ids_train_numpy"] = movie_ids_train_numpy
    _process_data["ratings_train_numpy"] = ratings_train_numpy


def _find_similar_movies(request):
    """Find similar movies for a range of entries in "movie_ratings".
    ::
        Input:
            _process_data["movie_ratings"] = [(movie_id, {user_id: rating})]
            _process_data["movie_genres"] = {movie id: set of genre ids}
            _process_data["buff_point"] = buff_point parameter for SimilarMovieFinder
            _process_data["buff_limit"] = buff_limit parameter for SimilarMovieFinder
            _process_data["movie_ratings_start"] = starting index of "movie_ratings" to work on
            _process_data["movie_ratings_length"] = length of "movie_ratings" to work on
        Result:
            _process_data["similar_movies"] = {movie_id: [similar_movie_ids]}
    """
    movie_ratings = _process_data["movie_ratings"]
    movie_genres = _process_data["movie_genres"]
    buff_point = _process_data["buff_point"]
    buff_limit = _process_data["buff_limit"]

    movies_finder = build_similar_movies_db.SimilarMovieFinder(
        movie_genres, movie_ratings,buff_limit, buff_point)

    similar_movies = {}  # {movie_id: [similar_movie_ids]}

    start = _process_data["movie_ratings_start"]
    length = _process_data["movie_ratings_length"]
    start_time = time.time()

    for i in range(start, start + length):
        similar_movie_ids, _ = movies_finder.find_similar_movie(i)

        if len(similar_movie_ids) > 0:
            movie_id = movies_finder.movie_ratings[i][0]
            similar_movies[movie_id] = similar_movie_ids

        # progress estimation
        if i == start + 200:
            t_so_far = time.time() - start_time
            seconds_left = t_so_far * (length - 200) / 200
            finish_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds_left)
            _lock.acquire()
            print("Process estimated completion time", finish_time)
            _lock.release()

    _process_data["similar_movies"] = similar_movies

