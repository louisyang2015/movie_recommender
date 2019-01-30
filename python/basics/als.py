import numpy, math, random
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


def generate_data(num_users, k=6):
    """
    :param num_users: number of users
    :param k: higher k means more movies, and more equations (ratings) generated
    :return: users, movies, user_ids, movie_ids, ratings
    """


    # Given the number of users, the number of movies is then computed
    # The equation is assuming rank 3 modeling (3 factors per user and per movie).
    num_movies = math.ceil(4 * k * num_users / (num_users - 12))

    # the users vector get an extra bias parameter
    users = numpy.random.uniform(-1, 1, (num_users * 4, 1))
    movies = numpy.random.uniform(-1, 1, (num_movies * 3, 1))

    # for simplicity, every user has rated every movie
    ratings = numpy.zeros((num_users * num_movies, 1), dtype=numpy.double)
    user_ids = numpy.zeros((num_users * num_movies, 1), dtype=numpy.int)
    movie_ids = numpy.zeros((num_users * num_movies, 1), dtype=numpy.int)

    index = 0

    for user_id in range(0, num_users):
        for movie_id in range(0, num_movies):
            u = users[user_id*4 : user_id*4 + 3]
            u_bias = users[user_id*4 + 3]
            m = movies[movie_id*3 : movie_id*3 + 3]
        
            user_ids[index] = user_id
            movie_ids[index] = movie_id
            ratings[index] = u.T.dot(m) + u_bias

            index += 1

    return users, movies, user_ids, movie_ids, ratings


def shuffle_vectors(user_ids, movie_ids, ratings):
    """Shuffle the elements in "user_ids, movie_ids, ratings". It's
    assumed that the three vectors are the same length."""

    length = len(user_ids)
    for i in range(0, length):
        swap_index =random.randint(0, length - 1)

        # swap items at "i" with "swap_index"
        if i != swap_index:
            user_ids[i][0], user_ids[swap_index][0] = user_ids[swap_index][0], user_ids[i][0]
            movie_ids[i][0], movie_ids[swap_index][0] = movie_ids[swap_index][0], movie_ids[i][0]
            ratings[i][0], ratings[swap_index][0] = ratings[swap_index][0], ratings[i][0]


def solve_for_users(movies, user_ids, movie_ids, ratings, num_users, factors):
    """given "movies", solve for "users"
    return users """

    # create the "A" matrix in "Ax=b"
    # It's a sparse matrix in (data, (row, col)) format
    data = numpy.zeros(ratings.shape[0] * (factors + 1), 
                       dtype=numpy.double)
    row = numpy.zeros(ratings.shape[0] * (factors + 1), 
                       dtype=numpy.int)
    col = numpy.zeros(ratings.shape[0] * (factors + 1), 
                       dtype=numpy.int)

    for i in range(0, ratings.shape[0]):
        movie_id = movie_ids[i][0]
        user_id = user_ids[i][0]

        for j in range(0, factors):
            # row is always the same
            row[i * (factors + 1) + j] = i

            # column[...] is user_id
            col[i * (factors + 1) + j] = user_id * (factors + 1) + j

            # data[...] = movie factors
            data[i * (factors + 1) + j] = movies[movie_id * factors + j][0]

        # bias term = 1
        row[i * (factors + 1) + factors] = i
        col[i * (factors + 1) + factors] = user_id * (factors + 1) + factors
        data[i * (factors + 1) + factors] = 1
    

    m = sparse.coo_matrix((data, (row, col)), 
                          shape=(ratings.shape[0], num_users * (factors + 1)), 
                          dtype=numpy.double)
    A = m.tocsr()

    result = linalg.lsqr(A, ratings, iter_lim=1000)
    users = numpy.array([result[0]]).T

    # print("Size of A =", A.shape)
    # print("Number of iterations =", result[2])

    return users


def solve_for_movies(users, user_ids, movie_ids, ratings, num_movies, factors):
    """given "users", solve for "movies"
    return movies """

    # create the "A" matrix in "Ax=b"
    # It's a sparse matrix in (data, (row, col)) format
    data = numpy.zeros(ratings.shape[0] * factors, dtype=numpy.double)
    row = numpy.zeros(ratings.shape[0] * factors, dtype=numpy.int)
    col = numpy.zeros(ratings.shape[0] * factors, dtype=numpy.int)

    for i in range(0, ratings.shape[0]):
        movie_id = movie_ids[i][0]
        user_id = user_ids[i][0]

        for j in range(0, factors):
            # row is always the same
            row[i * factors + j] = i

            # column[...] is movie_id
            col[i * factors + j] = movie_id * factors + j

            # data[...] = user factors
            data[i * factors + j] = users[user_id * (factors + 1) + j][0]                

    m = sparse.coo_matrix((data, (row, col)), 
                          shape=(ratings.shape[0], num_movies * factors), 
                          dtype=numpy.double)
    A = m.tocsr()

    # the user bias needs to be subtracted from "ratings"
    bias = users[3::4]
    ratings_minus_bias = numpy.array(ratings)

    for i in range(0, ratings_minus_bias.shape[0]):
        user_id = user_ids[i][0]
        ratings_minus_bias[i] -= bias[user_id]

    result = linalg.lsqr(A, ratings_minus_bias, iter_lim=1000)
    movies = numpy.array([result[0]]).T

    # print("Size of A =", A.shape)
    # print("Number of iterations =", result[2])

    return movies


def predict(user_id, movie_id, users, movies, factors):
    """Predicts how user_id will rate movie_id."""
    user_vec = users[user_id * (factors + 1) : user_id * (factors + 1) + factors]
    user_bias = users[user_id * (factors + 1) + factors][0]
    movie_vec = movies[movie_id * factors : movie_id * factors + factors]
    prediction = user_vec.T.dot(movie_vec)[0][0] + user_bias

    return prediction


def compute_loss(users, movies, user_ids, movie_ids, ratings, factors):
    """ compute the average loss of the current "users" and "movies" estimates 
    return loss"""
    loss = 0.0

    for i in range(0, ratings.shape[0]):
        user_id = user_ids[i][0]
        movie_id = movie_ids[i][0]

        prediction = predict(user_id, movie_id, users, movies, factors)
        loss += math.fabs(prediction - ratings[i][0])

    return loss / ratings.shape[0]


def run_test(num_users, k=6):
    # generate data
    factors = 3
    (users_real, movies_real, user_ids, movie_ids, ratings) = generate_data(num_users, k)
    num_movies = int(movies_real.shape[0] / factors)

    # split data into training and test data
    shuffle_vectors(user_ids, movie_ids, ratings)

    split_index = int(len(user_ids) * 0.8)
    user_ids, user_ids_test = user_ids[:split_index], user_ids[split_index:]
    movie_ids, movie_ids_test = movie_ids[:split_index], movie_ids[split_index:]
    ratings, ratings_test = ratings[:split_index], ratings[split_index:]

    # solve using training data, print error along the way
    movies = numpy.random.uniform(-1, 1, size=(num_movies * factors, 1))

    print("iteration".center(15), "training loss".center(25))

    for i in range(0, 7):
        users = solve_for_users(movies, user_ids, movie_ids, ratings, num_users, factors)
        movies = solve_for_movies(users, user_ids, movie_ids, ratings, num_movies, factors)
        loss = compute_loss(users, movies, user_ids, movie_ids, ratings, factors)
        print(str(i).center(15), str(loss).center(25))

    # apply model to test data
    # using "users_real" and "movies_real" should produce zero loss
    loss = compute_loss(users, movies, user_ids_test, movie_ids_test, ratings_test, factors)
    print("Loss using test data:", loss)

    # compare "users" to "users_real"
    print('Average difference between "users" and "users_real"',
          numpy.average(numpy.abs(users - users_real)))

    # compare "movies" to "movies_real"
    print('Average difference between "movies" and "movies_real"',
          numpy.average(numpy.abs(users - users_real)))


run_test(30)
run_test(30, k=10)
run_test(300)
run_test(3000)


