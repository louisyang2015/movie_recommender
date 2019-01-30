"""Access dynamo DB tables:
    ratings
"""

import datetime, pickle
import boto3


dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
users_table = dynamodb.Table('users')

# recommendation rotation size = 4
# only 1/4 of the full set of recommendation is returned per API request
rotation_size = 4



def _db_write(user_id : int, attrib : str, obj):
    """Stores "obj" using pickle."""
    users_table.put_item(Item={"user_id": user_id,
                                 "attrib": attrib,
                                 "obj": pickle.dumps(obj)})


def db_write_native(user_id : int, attrib : str, obj):
    """Stores "obj" directly, without using pickle."""
    users_table.put_item(Item={"user_id": user_id,
                                 "attrib": attrib,
                                 "obj": obj})


def _db_get(user_id : int, attrib: str):
    r = users_table.get_item(Key={"user_id": user_id, "attrib": attrib})

    if "Item" in r:
        return pickle.loads(r['Item']['obj'].value)
    else:
        return {}


def db_get_native(user_id : int, attrib):
    """Gets the object directly, without using pickle. Returns None
    if object not found."""
    r = users_table.get_item(Key={"user_id": user_id, "attrib": attrib})

    if "Item" in r:
        return r['Item']['obj']
    else:
        return None


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



