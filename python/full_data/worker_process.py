import multiprocessing
import my_util
import als_predictor, tag_count_predictor, tag_ls_predictor
import build_similar_movies_db


_cpu_count = multiprocessing.cpu_count()

_processes = [] # child processes
_pipes = [] # pipes to child processes

_process_data = {} # data passed in via a "set_data" operation



def _message_loop(conn):
    """
    :param conn: a multiprocessing.connection.PipeConnection
    """
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
    if cpu_count is not None:
        global _cpu_count
        _cpu_count = cpu_count

    for i in range(0, _cpu_count - 1):
        pipe1, pipe2 = multiprocessing.Pipe()
        process = multiprocessing.Process(target = _message_loop,
                                          args=(pipe2,))
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


def clear_all_data():
    for pipe in _pipes:
        pipe.send({"op": "clear_all_data"})

    _process_data.clear()


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


def split_list_and_send(data_list, start : int, length : int, var_name : str):
    """Split "data_list" and send it to each process. The
    current process gets the final split."""
    # compute split
    indices_and_lengths = _split(start, length, _cpu_count)

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
    length = indices_and_lengths[-1][1]
    _process_data[var_name] = data_list[index : index + length]


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


def _split(start: int, length: int, num_splits: int):
    """Given a "start" and a "length", generate
    a list of (index, length) pairs. For example,
    (start=10, length=8, num_splits=4) generates
    [(10, 2), (12, 2), (14, 2), (16, 2)]."""

    if length >= num_splits:
        # standard case
        # compute the indices
        indices = []
        for i in range(0, num_splits):
            indices.append(start + int(length * i / num_splits))

        result = []
        # most of the lengths are (next index - current index)
        for i in range(0, len(indices) - 1):
            result.append((indices[i], indices[i+1] - indices[i]))

        # the length for the final index:
        final_length = start + length - indices[-1]
        result.append((indices[-1], final_length))

        return result

    else:
        # special case
        result = []
        index = start
        for i in range(0, num_splits):
            if index < start + length:
                result.append((index, 1))
                index += 1
            else:
                result.append((index, 0))

        return result


def concat_var_into_list(var_name : str):
    """Merge "var_name" in all processes and
    concatenate them into a single list."""
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


# End of framework
#####################################################################


def _test_model(model : my_util.Model, movie_ratings):
    """Test "model" using "movie_ratings".

    :param model: a derivative of my_util.Model
    :param movie_ratings: [(movie_id, rating)]
    :return: prediction rank agreement
    """
    predicted_movie_ratings = [] # format: [(movie_id, rating)]
    movie_ratings2 = [] # format: [(movie_id, rating)]

    # "movie_ratings2" has the same meaning as the "movie_ratings" argument,
    # but some of the test data might be dropped if the model is
    # unable to make a prediction

    for movie_id, actual_rating in movie_ratings:
        predicted_rating = model.predict(movie_id)

        if predicted_rating is not None:
            predicted_movie_ratings.append((movie_id, predicted_rating))
            movie_ratings2.append((movie_id, actual_rating))

    agreement = None

    if len(predicted_movie_ratings) > 1:
        agreement = my_util.compute_ranking_agreement(
            movie_ratings2, predicted_movie_ratings)

    return agreement



def _als_eval(request):
    """
    Input:
        * _process_data["user_ratings_test"]
        * _process_data["movie_medians_train"]
        * _process_data["als_user_factors"]
        * _process_data["als_user_ids"]
        * _process_data["als_movie_factors"]
        * _process_data["als_movie_ids"]
    Output:
        * _process_data["user_agreements"] - [(user id, agreement)]
    """
    # extract the necessary data structures
    user_ratings_test = _process_data["user_ratings_test"]

    movie_medians_train = _process_data["movie_medians_train"]
    als_user_factors = _process_data["als_user_factors"]
    als_user_ids = _process_data["als_user_ids"]
    als_movie_factors = _process_data["als_movie_factors"]
    als_movie_ids = _process_data["als_movie_ids"]

    user_agreements = [] # each element is (user id, agreement)

    # user_ratings_test is [ (user_id, [(movie_id, rating)] ) ]
    for i in range(0, len(user_ratings_test)):
        # extract user_factors and initialize ALS model
        user_id = user_ratings_test[i][0]
        als_user_id = als_user_ids[user_id]

        num_user_factors = _process_data["num_item_factors"] + 1
        user_factors = als_user_factors[num_user_factors * als_user_id :
                                        num_user_factors * (als_user_id + 1)]

        user_als_model = als_predictor.ALS_Model(
            user_factors, movie_medians_train, als_movie_factors, als_movie_ids)

        # test "user_als_model" using "user_ratings_test"
        movie_ratings_test = user_ratings_test[i][1]

        agreement = _test_model(user_als_model, movie_ratings_test)

        if agreement is not None:
            user_agreements.append((user_id, agreement))

    _process_data["user_agreements"] = user_agreements  # [(user id, agreement)]


def _build_similar_movies_db(request):
    """
    Input:
        * _process_data["movie_ratings_start"]
        * _process_data["movie_ratings_length"]
        * _process_data["buff_limit"]
        * _process_data["buff_point"]
        * _process_data["movie_genres"]
        * _process_data["movie_ratings"]
    Output:
        * _process_data["similar_movies"] - [(movie_id, [similar_movie_ids])]
    """
    # extract the necessary data structures
    start_index = _process_data["movie_ratings_start"]
    length = _process_data["movie_ratings_length"]

    buff_limit = _process_data["buff_limit"]
    buff_point = _process_data["buff_point"]
    movie_genres = _process_data["movie_genres"]
    movie_ratings = _process_data["movie_ratings"]

    movie_finder = build_similar_movies_db.SimilarMovieFinder(
        movie_genres, movie_ratings, buff_limit, buff_point)

    similar_movies = [] # [(movie_id, [similar_movie_ids])]

    # build up "similar_movies"
    for i in range(start_index, start_index + length):
        similar_movie_ids, _ = movie_finder.find_similar_movie(i)
        if len(similar_movie_ids) > 0:
            similar_movies.append((movie_ratings[i][0], similar_movie_ids))

    _process_data["similar_movies"] = similar_movies # [(movie_id, [similar_movie_ids])]


def _median_eval(request):
    """
    Input:
        * _process_data["user_ratings_test"]
        * _process_data["movie_medians_train"]
    Output:
        * _process_data["user_agreements"] - [(user id, agreement)]
    """
    user_ratings_test = _process_data["user_ratings_test"]
    movie_medians_train = _process_data["movie_medians_train"]

    user_agreements = [] # each element is (user id, agreement)

    # user_ratings_test is [ (user_id, [(movie_id, rating)] ) ]
    for user_id, actual_ratings in user_ratings_test:
        predicted_ratings = []
        actual_ratings2 = [] # not all of test data will be accepted

        for movie_id, rating in actual_ratings:

            if movie_id in movie_medians_train:
                predicted_ratings.append((movie_id, movie_medians_train[movie_id]))
                actual_ratings2.append((movie_id, rating))

        agreement = my_util.compute_ranking_agreement(actual_ratings2, predicted_ratings)
        if agreement is not None:
            user_agreements.append((user_id, agreement))

    _process_data["user_agreements"] = user_agreements # [(user id, agreement)]



def _tag_count_eval(request):
    """
    Input:
        * _process_data["user_ratings_train"]
        * _process_data["user_ratings_test"]
        * _process_data["movie_genres"]
        * _process_data["movie_tags"]
        * _process_data["tag_counts"]
        * _process_data["genre_counts"]
        * _process_data["movie_medians_train"]
    Output:
        * _process_data["user_agreements"] - [(user id, agreement)]
    """
    # extract the necessary data structures
    user_ratings_train = _process_data["user_ratings_train"]
    user_ratings_test = _process_data["user_ratings_test"]

    movie_genres = _process_data["movie_genres"]
    movie_tags = _process_data["movie_tags"]
    tag_counts = _process_data["tag_counts"]
    genre_counts = _process_data["genre_counts"]
    movie_medians_train = _process_data["movie_medians_train"]

    user_agreements = [] # each element is (user id, agreement)

    # user_ratings_test is [ (user_id, [(movie_id, rating)] ) ]
    for i in range(0, len(user_ratings_test)):
        # train model using "user_ratings_train"
        movie_ratings_train = user_ratings_train[i][1]
        user_profile = tag_count_predictor.UserProfile(
            movie_genres, movie_ratings_train, movie_tags,
            tag_counts, genre_counts, movie_medians_train)

        # test using "user_ratings_test"
        user_id = user_ratings_test[i][0]
        movie_ratings_test = user_ratings_test[i][1]

        agreement = _test_model(user_profile, movie_ratings_test)

        if agreement is not None:
            user_agreements.append((user_id, agreement))

    _process_data["user_agreements"] = user_agreements  # [(user id, agreement)]


def _tag_ls_eval(request):
    """
    Input:
        * _process_data["user_ratings_train"]
        * _process_data["user_ratings_test"]
        * _process_data["movie_genres"]
        * _process_data["movie_tags"]
        * _process_data["tag_counts"]
        * _process_data["movie_medians_train"]
    Output:
        * _process_data["user_agreements"] - [(user id, agreement)]
    """
    # extract the necessary data structures
    user_ratings_train = _process_data["user_ratings_train"]
    user_ratings_test = _process_data["user_ratings_test"]

    movie_genres = _process_data["movie_genres"]
    movie_tags = _process_data["movie_tags"]
    tag_counts = _process_data["tag_counts"]
    movie_medians_train = _process_data["movie_medians_train"]

    user_agreements = [] # each element is (user id, agreement)

    # user_ratings_test is [ (user_id, [(movie_id, rating)] ) ]
    for i in range(0, len(user_ratings_test)):
        # train model using "user_ratings_train"
        movie_ratings_train = user_ratings_train[i][1]
        user_profile = tag_ls_predictor.UserProfile(
            movie_genres, movie_ratings_train, movie_tags,
            tag_counts, movie_medians_train)

        # test using "user_ratings_test"
        user_id = user_ratings_test[i][0]
        movie_ratings_test = user_ratings_test[i][1]

        agreement = _test_model(user_profile, movie_ratings_test)

        if agreement is not None:
            user_agreements.append((user_id, agreement))

    _process_data["user_agreements"] = user_agreements  # [(user id, agreement)]



