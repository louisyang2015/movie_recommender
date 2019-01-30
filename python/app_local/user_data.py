"""Access dynamo DB tables:
    ratings
"""

import datetime, pickle, os


user_data_file_name = "user_data.bin"

# load "user_data" from disk if possible
user_data = {}
if os.path.exists(user_data_file_name):
    with open(user_data_file_name, mode="rb") as file:
        user_data = pickle.load(file)


# recommendation rotation size = 4
# only 1/4 of the full set of recommendation is returned per API request
rotation_size = 4



def _db_write(user_id : int, attrib : str, obj):
    """Stores "obj" using pickle."""
    if user_id not in user_data: user_data[user_id] = {}
    if attrib not in user_data[user_id]:
        user_data[user_id][attrib] = {}

    user_data[user_id][attrib] = obj

    with open(user_data_file_name, mode="wb") as file:
        pickle.dump(user_data, file)


def db_write_native(user_id : int, attrib : str, obj):
    """Stores "obj" directly, without using pickle."""
    # for local disk file, this is the same as _db_write
    _db_write(user_id, attrib, obj)


def _db_get(user_id : int, attrib: str):
    if user_id not in user_data: return {}
    if attrib not in user_data[user_id]: return {}

    return user_data[user_id][attrib]


def db_get_native(user_id : int, attrib):
    """Gets the object directly, without using pickle. Returns None
    if object not found."""
    # This is similar to "_db_get", but return a "None" instead
    # of an empty dictionary
    obj = _db_get(user_id, attrib)
    if len(obj) == 0: return None

    return obj


def get_ratings(user_id : int):
    """Retrieve a {movie_id: rating} ratings dictionary
    from the database."""
    return _db_get(user_id, "ratings")


def store_ratings(user_id : int, ratings : dict):
    """Store "ratings" dictionary into the database."""
    _db_write(user_id, "ratings", ratings)

    # update status timestamp
    status = _db_get(user_id, "status")
    status["ratings_mod_time"] = datetime.datetime.utcnow().timestamp()
    _db_write(user_id, "status", status)


def is_new_recommendation_needed(user_id : int, algorithm : str):
    """Checks the timestamps to see if a new recommendation is needed.
    :param user_id: user ID
    :param algorithm: Name of the algorithm, such as "als3".
    """
    status = _db_get(user_id, "status")
    needed = False # function return value

    if "ratings_mod_time" not in status:
        status["ratings_mod_time"] = datetime.datetime.utcnow().timestamp()
        _db_write(user_id, "status", status)
        needed = True

    if algorithm + "_mod_time" not in status:
        needed = True
    else:
        algorithm_mod_time = status[algorithm + "_mod_time"]
        if algorithm_mod_time <= status["ratings_mod_time"]:
            needed = True

    return needed


def store_recommendation(user_id : int, movie_ids : list, algorithm : str):
    """Store "movie_ids" as recommendation for "user_id". Also updates
    timestamps and rotation value."""
    _db_write(user_id, algorithm + "_rec", movie_ids)

    # update status
    status = _db_get(user_id, "status")
    status[algorithm + "_mod_time"] = datetime.datetime.utcnow().timestamp()
    status[algorithm + "_rotation"] = 1 # the caller of this function will return rotation 0
    _db_write(user_id, "status", status)


def get_recommendation(user_id : int, algorithm : str):
    """Get recommendation from the database. This
    returns a list of movie ids."""
    recommendation = _db_get(user_id, algorithm + "_rec")

    # if the "recommendation" doesn't exist, it defaults to a dictionary
    if len(recommendation) == 0: return []

    # extract subset of "recommendation" based on current "rotation"
    status = _db_get(user_id, "status")
    rotation = status[algorithm + "_rotation"]
    recommendation = recommendation[rotation::rotation_size]

    # update "rotation" value
    new_rotation = rotation + 1
    if new_rotation == rotation_size: new_rotation = 0
    status[algorithm + "_rotation"] = new_rotation

    _db_write(user_id, "status", status)

    return recommendation



