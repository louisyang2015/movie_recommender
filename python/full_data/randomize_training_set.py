"""
Command line flags:
    overwrite - sets the overwrite flag, overwriting existing .bin files
        default is False

    training_set_ratio=0.8 - sets percentage of data to use for training
        default is 0.8

    cpu_count=4 - sets number of processes to use
        default is None

One time creations:
    genre_counts.bin - {genre id : genre count}
    genre_ids.bin - {genre string : genre id}
    movie_genres.bin - {movie id : set of genre ids}

    user_ratings.bin - [ (user_id, [(movie_id, rating)] ) ]

    movie_tags.bin - {movie_id: {tag_id: ln(count) + 1} }
    tag_counts.bin - {tag_id: count}
    tag_names.bin - {tag id: tag string}
    tag_ids.bin - {tag string : tag id}

Refresh as needed:
    user_ratings_train.bin - [ (user id, [(movie id, rating)] ) ]
    user_ratings_test.bin - [ (user id, [(movie id, rating)] ) ]
    user_ratings_test_length.bin - length of "user_ratings_test"
    movie_medians_train.bin - {movie id : movie median rating}
"""
import datetime, sys, time
import movie_lens_data
import movie_lens_data_proc as _proc



def main():
    # process command line arguments
    overwrite = False
    training_set_ratio = 0.8
    cpu_count = None

    for arg in sys.argv:
        if arg == "overwrite":
            overwrite = True
        elif arg.startswith("training_set_ratio="):
            training_set_ratio = float(arg.split(sep='=')[1])
        elif arg.startswith("cpu_count="):
            cpu_count = int(arg.split(sep='=')[1])

    # start processes and refresh training set
    _proc.start_processes(cpu_count)

    movie_lens_data.start_time = time.time()
    movie_lens_data.read_movies_csv(overwrite)
    movie_lens_data.read_tags_csv(6, overwrite)

    user_ratings = movie_lens_data.create_user_ratings(overwrite)

    if training_set_ratio < 1:
        movie_lens_data.refresh_training_sets_mp(user_ratings, training_set_ratio)

    _proc.end_processes()

    # print run time
    run_time = int(time.time() - movie_lens_data.start_time)
    print("Total run time", datetime.timedelta(seconds=run_time))
    print('\a')



if __name__ == "__main__":
    main()
