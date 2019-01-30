"""
Command line flags:
    overwrite - sets the overwrite flag, overwriting existing .bin files
        default is False

    training_set_ratio=0.8 - sets percentage of data to use for training
        default is 0.8

    cpu_count=4 - sets number of processes to use for data processing
        default is None

    als_thread_count=4 - sets the number of threads to use for the ALS
        default is None

    algorithm=1 - default algorithm uses transpose multiply; algorithm 2
        will compute transpose explicitly

One time creations:
    user_ratings.bin - [ (user_id, [(movie_id, rating)] ) ]

Refresh as needed:
    user_ratings_train.bin - [ (user id, [(movie id, rating)] ) ]
    user_ratings_test.bin - [ (user id, [(movie id, rating)] ) ]
    user_ratings_test_length.bin - length of "user_ratings_test"
    movie_medians_train.bin - {movie id : movie median rating}

    movie_medians_full.bin - {movie id : movie median rating} - this
        happens only if training_set_ratio=1

    ALS files - one set for each factor:

    als3_movie_ids.bin - {standard movie id : zero based movie id}
    als3_user_ids.bin - {standard user id : zero based user id}
    als3_user_ratings_test.bin - [ (user id, [(movie id, rating)] ) ]
    als3_user_ratings_test_length.bin - length of "als3_user_ratings_test"

    als3_user_ratings_train.bin - [user_ids_train_numpy,
        movie_ids_train_numpy, ratings_train_numpy]

    als3_item_factors.bin - numpy array of item factors
    als3_user_factors.bin - numpy array of user factors
"""
import datetime, gc, os, pickle, sys, time
import movie_lens_data
import movie_lens_data_proc as _proc


def main():
    # process command line arguments
    overwrite = False
    training_set_ratio = 0.8
    cpu_count = None
    als_thread_count = None
    algorithm = 1

    for arg in sys.argv:
        if arg == "overwrite":
            overwrite = True
        elif arg.startswith("training_set_ratio="):
            training_set_ratio = float(arg.split(sep='=')[1])
        elif arg.startswith("cpu_count="):
            cpu_count = int(arg.split(sep='=')[1])
        elif arg.startswith("als_thread_count="):
            als_thread_count = int(arg.split(sep='=')[1])
        elif arg.startswith("algorithm="):
            algorithm = int(arg.split(sep='=')[1])

    # Start extra processes and shrink the data set to meet ALS factor
    # requirements.
    _proc.start_processes(cpu_count)

    als_factors_list = [3, 5, 7, 9, 11]

    user_ratings = movie_lens_data.create_user_ratings(overwrite)

    if training_set_ratio >= 1:
        # Use the whole data set as training set.
        # "movie_lens_data.als_data_set_shrink_mp(...)" will assume
        # "user_ratings_train" and "user_ratings_test" to be in process memory.
        _proc.split_list_and_send(user_ratings, "user_ratings_train")
        _proc.send_same_data({"user_ratings_test": None})

        # compute movie medians, save to disk as "movie_medians_full.bin"
        median_file_name = movie_lens_data.in_dir + os.sep + "movie_medians_full.bin"

        if os.path.exists(median_file_name) and overwrite == False:
            movie_medians = movie_lens_data.get_input_obj("movie_medians_full")

        else:
            movie_ratings = movie_lens_data.create_movie_ratings(overwrite)

            print("        Computing movie medians for the full data set")
            _proc.split_list_and_send(movie_ratings, "movie_ratings")
            _proc.run_function("_compute_medians2", {})
            movie_medians = _proc.update_var_into_dict("movie_medians")

            print("        Saving", median_file_name)
            with open(median_file_name, mode="bw") as file:
                pickle.dump(movie_medians, file)

        movie_lens_data.als_data_set_shrink_mp(movie_medians, als_factors_list,
                                               no_test_set=True)

    else:
        # training_set_ratio < 1
        movie_medians_train = movie_lens_data.refresh_training_sets_mp(
            user_ratings, training_set_ratio)

        movie_lens_data.als_data_set_shrink_mp(movie_medians_train, als_factors_list)

    _proc.end_processes()
    gc.collect()

    movie_lens_data.als_train(als_factors_list, als_thread_count, algorithm)

    # print run time
    run_time = int(time.time() - movie_lens_data.start_time)
    print("Total run time", datetime.timedelta(seconds=run_time))
    print('\a')



if __name__ == "__main__":
    main()
