import numpy, time
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


def generate_data(data_low, data_high, x_low, x_high):
    """Returns A_data, b_data, x"""
            
    non_zeros_per_row = 10
    zeros_per_row = 1000
    # "zeros per row" should be a multiple of "non zeros per row"  

    num_columns = non_zeros_per_row + zeros_per_row

    # Low number of rows:
    num_rows = int(num_columns * (num_columns / non_zeros_per_row) * 4)

    # Medium number of rows:
    # num_rows = int(num_columns * (num_columns / non_zeros_per_row) * 40)

    # High number of rows to simulate movie lens large data set,
    # but this takes a long time to run.
    # num_rows = int(num_columns * (num_columns / non_zeros_per_row) * 200)

    # allocate space for column, row, and data
    length = num_rows * non_zeros_per_row
    block_length = int(num_columns / non_zeros_per_row)

    row = numpy.zeros(length)

    # the "col" gets a random index between 0 and "block_length"
    # an offset will be applied later
    col = numpy.random.random_integers(0, block_length - 1, length)

    data = numpy.random.uniform(data_low, data_high, length)

    index = 0

    # the following nested for loop takes a long time
    for i in range(0, num_rows):        
        for j in range(0, non_zeros_per_row):
            row[index] = i

            # apply offset to the random number already populated in "col"
            col[index] += j * block_length

            index += 1
            
    m = sparse.coo_matrix((data, (row, col)), shape=(num_rows, num_columns), 
                          dtype=numpy.double)

    A_data = m.tocsr()
    x = numpy.random.uniform(x_low, x_high, (num_columns, 1))
    b_data = A_data.dot(x) + numpy.random.normal(0, 0.1, (num_rows, 1))

    return A_data, b_data, x


def run_test(data_low, data_high, x_low, x_high):
    """Generates a test problem and solves it using Scipy's LSQR"""

    start_time = time.time()
    A, b, x_real = generate_data(data_low, data_high, x_low, x_high)
    print("generate_data(...) run time:", time.time() - start_time, "seconds")

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




run_test(data_low=0, data_high=100, x_low=0, x_high=20)


# most of the time is actually spent inside "generate_data(...)"
# A, b, x_real = generate_data(data_low=0, data_high=100, x_low=0, x_high=20)




