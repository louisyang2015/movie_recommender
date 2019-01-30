import ctypes, multiprocessing, numpy, os, random

_dll = None

def _load_dll():
    global _dll

    if os.name == "nt":
        _dll = ctypes.windll.cpp_ls
    else:
        dll_file_name = os.getcwd() + os.sep + "cpp_ls_lib.so"
        _dll = ctypes.cdll.LoadLibrary(dll_file_name)

    _dll.set_thread_count(multiprocessing.cpu_count())


_load_dll()





def has_dll_loaded():
    "Returns true if the C++ dll has been successfully loaded."
    # Test the dll by calling "set_thread_count()" on a random number
    old_thread_count = _dll.get_thread_count()

    random_number = random.randint(1, 100000)
    _dll.set_thread_count(random_number)

    success = True
    if _dll.get_thread_count() != random_number: success = False

    _dll.set_thread_count(old_thread_count)

    return success


def set_thread_count(thread_count):
    _dll.set_thread_count(thread_count)



def cg_least_squares(A_row_indices : numpy.ndarray,
                     A_col_indices : numpy.ndarray,
                     A_values : numpy.ndarray, A_num_columns : int,
                     b : numpy.ndarray, min_r_decrease = 0.01,
                     max_iterations = 200, algorithm = 1):
    """Calls the library's cg_least_squares_from_python(...).
    Solves Ax = b in the least squares sense.

    :param A_row_indices: describes the starting index of each
        row, numpy array of type int32, length is 1 more than the number of rows

    :param A_col_indices: describes the column index of each value,
        numpy array of type int32

    :param A_values: content of A, numpy array of type double

    :param A_num_columns: number of columns in A

    :param b: the "b" in Ax=b, numpy array type double

    :param min_r_decrease: the minimum percentage decrease in rr, used
        as a termination condition
    :param max_iterations: the maximum number of iterations to run the
        algorithm, used as a termination condition

    :param algorithm: 1 (default) leads to cg_least_squares(...), which
        does not compute the transpose of A. Other values lead to
        cg_least_squares2(...), which does compute A transpose and so
        uses more memory, but might be faster for very large problems
        on machines with many cores.

    :return: x, iterations, final_rr - the solution vector "x" is a
        numpy column vector of type double, the number of "iterations"
        the algorithm took, and what is the final "rr" value, representing
        the amount of error
    """
    A_rows = len(A_row_indices) - 1
    A_row_indices_ptr = A_row_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    A_col_indices_ptr = A_col_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    A_values_ptr = A_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    b_length = len(b)
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # generate solution vector x
    x = numpy.random.uniform(-1, 1, (A_num_columns, 1))
    x_length = A_num_columns
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    final_rr = ctypes.c_double(0)
    iterations = 0

    if algorithm == 1:
        iterations = _dll.cg_least_squares_from_python(
            A_rows, A_num_columns, A_row_indices_ptr, A_col_indices_ptr,
            A_values_ptr, b_length, b_ptr, x_length, x_ptr,
            ctypes.c_double(min_r_decrease), max_iterations, ctypes.byref(final_rr))

    else:
        iterations = _dll.cg_least_squares_from_python2(
            A_rows, A_num_columns, A_row_indices_ptr, A_col_indices_ptr,
            A_values_ptr, b_length, b_ptr, x_length, x_ptr,
            ctypes.c_double(min_r_decrease), max_iterations, ctypes.byref(final_rr))

    return x, iterations, final_rr.value


def als(user_ids : numpy.ndarray, item_ids : numpy.ndarray,
        ratings : numpy.ndarray, num_item_factors : int,
        num_users: int, num_items : int, min_r_decrease=0.01,
        max_iterations=200, algorithm=1):
    """Calls the library's als_from_python(...)

    :param user_ids: numpy array of type int32, the ids need to be zero
        based and continuous
    :param item_ids: numpy array of type int32, the ids need to be zero
        based and continuous

    :param ratings: numpy array of type double

    :param num_item_factors: number of factors per item. The number of user
        factors will be one more than this.

    :param num_items: the number of items
    :param num_users: the number of users

    :param min_r_decrease: the minimum percentage decrease in rr, used
        as a termination condition
    :param max_iterations: the maximum number of iterations to run the
        algorithm, used as a termination condition

    :param algorithm: 1 (default) leads to cg_least_squares(...), which
        does not compute the transpose of A. Other values lead to
        cg_least_squares2(...), which does compute A transpose and so
        uses more memory, but might be faster for very large problems
        on machines with many cores.

    :return: user_factors, item_factors, iterations - "user_factors"
        and "item_factors" are numpy arrays of type double, and the
        number of "iterations" the algorithm took
    """
    # allocate "user_factors" and "item_factors"
    num_user_factors = num_item_factors + 1
    user_factors = numpy.random.uniform(-1, 1, num_users * num_user_factors)
    item_factors = numpy.random.uniform(-1, 1, num_items * num_item_factors)

    # argument construction
    user_ids_ptr = user_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    item_ids_ptr = item_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ratings_length = len(ratings)
    ratings_ptr = ratings.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    user_factors_length = len(user_factors)
    user_factors_ptr = user_factors.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    item_factors_length = len(item_factors)
    item_factors_ptr = item_factors.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    iterations = _dll.als_from_python(
        user_ids_ptr, item_ids_ptr, ratings_length, ratings_ptr,
        num_item_factors, user_factors_length, user_factors_ptr,
        item_factors_length, item_factors_ptr, ctypes.c_double(min_r_decrease),
        max_iterations, algorithm)

    return user_factors, item_factors, iterations









