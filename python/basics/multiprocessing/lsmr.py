import numpy, time
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

import lsmr_proc as _proc


def generate_data(data_low, data_high, x_low, x_high):
    """Returns A_data, b_data, x. "data_low" and "data_high"
    refers to A_data. "x_low" and "x_high" refers to x. The
    "b_data" is calculated. """

    non_zeros_per_row = 10 # factor 9 model
    zeros_per_row = int(280e3) * non_zeros_per_row # 280,000 users

    num_columns = non_zeros_per_row + zeros_per_row
    num_rows = int(27e6) # 27 million ratings

    # allocate space for column, row, and data
    length = num_rows * non_zeros_per_row
    block_length = int(num_columns / non_zeros_per_row)

    _proc.define_list(num_rows)
    _proc.send_same_data({"block_length": block_length,
                          "non_zeros_per_row": non_zeros_per_row})
    _proc.run_function("_create_data", {})

    row = _proc.concat_var_into_numpy_array("row")
    col = _proc.concat_var_into_numpy_array("col")

    data = numpy.random.uniform(data_low, data_high, length)

    m = sparse.coo_matrix((data, (row, col)), shape=(num_rows, num_columns),
                          dtype=numpy.double)

    A_data = m.tocsr()
    x = numpy.random.uniform(x_low, x_high, (num_columns, 1))
    b_data = A_data.dot(x) + numpy.random.normal(0, 0.1, (num_rows, 1))

    return A_data, b_data, x


def run_test(data_low, data_high, x_low, x_high):
    """Generates a test problem and solves it using Scipy's LSQR"""
    _proc.start_processes()

    start_time = time.time()
    A, b, x_real = generate_data(data_low, data_high, x_low, x_high)
    print("generate_data(...) run time:", time.time() - start_time, "seconds")

    _proc.end_processes()

    print("Size of A =", A.shape)
    print()

    start_time = time.time()
    lsmr_result = linalg.lsmr(A, b, maxiter=1000)

    print("linalg.lsmr(A, b) run time:", time.time() - start_time, "seconds")
    print("Number of iterations =", lsmr_result[2])

    x_lsmr = numpy.array([lsmr_result[0]]).T
    print("sum(|x_real - x_lsmr|) =", numpy.sum(numpy.abs(x_real - x_lsmr)))
    print()

    start_time = time.time()
    lsqr_result = linalg.lsqr(A, b, iter_lim=1000)

    print("linalg.lsqr(A, b) run time:", time.time() - start_time, "seconds")
    print("Number of iterations =", lsqr_result[2])

    x_lsqr = numpy.array([lsqr_result[0]]).T
    print("sum(|x_real - x_lsqr|) =", numpy.sum(numpy.abs(x_real - x_lsqr)))





def main():
    run_test(data_low=-100, data_high=100, x_low=-10, x_high=10)




if __name__ == "__main__":
    main()


