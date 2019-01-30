import math, numpy, random
import cpp_ls


def test_cg_least_squares():
    """Test the library's ability to solve Ax=b in the least
    squares sense."""

    A_size = (200, 50)  # (row, column) size of A
    # To actually see CPU usage go to 100 at the very end of this test, use:
    # A_size = (16000, 4000) # (row, column) size of A

    print("       starting test_cg_least_squares()")

    # generate random A, x_real, b
    A = numpy.random.uniform(-1, 1, A_size)
    x_real = numpy.random.uniform(-1, 1, (A_size[1], 1))
    noise = numpy.random.normal(0, 0.1, (A_size[0], 1))
    b = A.dot(x_real) + noise

    # compute x
    print("       converting test matrix A to sparse form")
    A_row_indices, A_col_indices, A_values = convert_dense_matrix_to_sparse_format(A)

    print("       calling cpp_ls.cg_least_squares()")
    x, iterations, final_rr = cpp_ls.cg_least_squares(
        A_row_indices, A_col_indices, A_values, A_size[1], b)

    # compare x with x_real
    average_error = numpy.sum(numpy.abs(x_real - x)) / len(x_real)

    if average_error < 0.1:
        print("pass - cpp_ls.cg_least_squares() passed testing, coefficient average_error =",
              average_error)
        print("       iterations =", iterations, " final_rr =", final_rr)
    else:
        print("FAIL - cpp_ls.cg_least_squares() failed testing, coefficient average_error =",
              average_error)
        print("       iterations =", iterations, " final_rr =", final_rr)



def convert_dense_matrix_to_sparse_format(A : numpy.ndarray):
    """Converts a dense numpy 2D array (A) to three 1D numpy arrays, which
    is a sparse matrix format used by the cpp_ls library.

    :param A: numpy 2D array
    :return: row_indices, col_indices, values - "row_indices" and "col_indices"
        are numpy arrays of the type int, and "values" is a numpy array of
        the type double
    """
    row_indices = []
    col_indices = []
    values = []

    for i in range(0, A.shape[0]):
        row_indices.append(len(col_indices))

        for j in range(0, A.shape[1]):
            if A[i][j] != 0:
                # add the value A[i][j] to "col_indices" and "values"
                col_indices.append(j)
                values.append(A[i][j])

    row_indices.append(len(col_indices))

    return numpy.array(row_indices, dtype=numpy.int32), \
           numpy.array(col_indices, dtype=numpy.int32), \
           numpy.array(values, dtype=numpy.double)



def test_als():
    """Test the library's ability to build an ALS model."""
    # test parameters:
    num_item_factors = 5
    training_set_ratio = 0.8 # use 80% of data for training
    k = 6.0 # should be > 1, higher k = more data

    # to see CPU go to 100%
    # num_item_factors = 10
    # k = 100.0

    # derived test parameters
    num_user_factors = num_item_factors + 1
    num_items = math.ceil(num_user_factors * k / training_set_ratio)
    num_users = math.ceil(num_item_factors * k / training_set_ratio)

    # generate user_factors_real and item_factors_real
    print("       starting test_als(), generating test data")
    user_factors_real = numpy.random.uniform(-1, 1, num_users * num_user_factors)
    item_factors_real = numpy.random.uniform(-1, 1, num_items * num_item_factors)

    # generate user_ids, item_ids, and ratings. All users review all items
    user_ids = numpy.zeros(num_users * num_items, dtype=numpy.int32)
    item_ids = numpy.zeros(num_users * num_items, dtype=numpy.int32)
    ratings = numpy.zeros(num_users * num_items, dtype=numpy.double)

    index = 0
    for user_id in range(0, num_users):
        for item_id in range(0, num_items):
            user_ids[index] = user_id
            item_ids[index] = item_id
            ratings[index] = als_predict(user_id, item_id, user_factors_real,
                                         item_factors_real, num_item_factors)  \
                             + numpy.random.normal(0, 0.1) # ratings has noise added
            index += 1

    # shuffle data, then split into training and test sets
    user_ids_length = len(user_ids)

    for i in range(0, user_ids_length):
        swap_index = random.randint(0, user_ids_length - 1)
        if swap_index != i:
            user_ids[i], user_ids[swap_index] = user_ids[swap_index], user_ids[i]
            item_ids[i], item_ids[swap_index] = item_ids[swap_index], item_ids[i]
            ratings[i], ratings[swap_index] = ratings[swap_index], ratings[i]

    training_size = math.ceil(user_ids_length * training_set_ratio)
    user_ids_train, user_ids_test = user_ids[:training_size], user_ids[training_size:]
    item_ids_train, item_ids_test = item_ids[:training_size], item_ids[training_size:]
    ratings_train, ratings_test = ratings[:training_size], ratings[training_size:]

    # train als model using training data
    print("       calling cpp_ls.als() on training set")

    user_factors, item_factors, iterations = cpp_ls.als(
        user_ids_train, item_ids_train, ratings_train,
        num_item_factors, num_users, num_items)

    print("       cpp_ls.als() completed in", iterations, "iterations")

    # make predictions using the test data
    ratings_predictions = numpy.zeros(len(ratings_test), dtype=numpy.double)
    for i in range(0, len(ratings_predictions)):
        ratings_predictions[i] = als_predict(user_ids_test[i], item_ids_test[i],
                                             user_factors, item_factors, num_item_factors)

    # calculate the average prediction error
    average_error = numpy.sum(numpy.abs(ratings_predictions - ratings_test)) / len(ratings_test)

    # if average_error < 0.10: ?? --- Similar to the C++ testing, an average
    # error of under 0.10 is ideal, but does not always happen.
    if average_error < 0.15:
        print("pass - cpp_ls.als() passed testing, prediction average_error =", average_error)
    else:
        print("FAIL - cpp_ls.als() failed testing, prediction average_error =", average_error)



def als_predict(user_id, item_id, user_factors, item_factors, num_item_factors):
    """Make a prediction using the given ALS model (user_factors, item_factors)
    for the given (user_id, item_id)."""
    num_user_factors = num_item_factors + 1
    sum = 0.0

    for i in range(0, num_item_factors):
        sum += user_factors[user_id * num_user_factors + i] * item_factors[item_id * num_item_factors + i]

    # add in the final user factor, the user bias
    sum += user_factors[user_id * num_user_factors + num_user_factors - 1]

    return sum




def main():
    if cpp_ls.has_dll_loaded():
        print("pass - has_dll_loaded() is True")
    else:
        print("FAIL - has_dll_loaded() is False")

    test_cg_least_squares()
    test_als()


if __name__ == "__main__":
    main()
