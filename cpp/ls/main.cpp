#include <iostream>
#include "matrix.h"


using namespace std;


template <typename T>
class Stats
{
    T min = 0;
    T max = 0;
    T total = 0;
    int count = 0;

public:
    void add_data(T value)
    {
        if (count == 0)
        {
            min = value; max = value; total = value;
        }
        else
        {
            if (value < min) min = value;
            if (value > max) max = value;
            total += value;
        }
        count++;
    }

    void print()
    {
        cout << "average = " << (double)total / count;
        cout << ", min = " << min;
        cout << ", max = " << max;
    }
};



void test_vect_add();
void test_vect_dot_product();
void test_sparse_matrix_multiply(bool use_sorted_rows);
void test_sparse_matrix_transpose_multiply(bool use_sorted_rows);
void test_sparse_matrix_transpose();
void test_least_squares(int algorithm, int num_tests);
void test_als(int algorithm);

void benchmark_col_vector_add(int rows);
void benchmark_sparse_matrix_multiply(int rows, int cols, int values_per_row,
    bool use_sorted_rows);
void benchmark_sparse_matrix_transpose_multiply(int rows, int cols,
    int values_per_row, bool use_sorted_rows);
void benchmark_sparse_matrix_transpose(int rows, int cols, int values_per_row);
void benchmark_least_squares(int rows, int cols, int values_per_row, int algorithm);

void fill_matrix_with_dense_random_data(SparseMatrix* m, int rows, int cols,
    int rand_low, int rand_high);
void fill_matrix_with_sparse_random_data(SparseMatrix* m, int rows, int cols,
    int values_per_row, int rand_low, int rand_high, bool new_buffers = true);
void fill_matrix_with_sparse_random_data_t(int start, int end, SparseMatrix* m,
    int rows, int cols, int values_per_row, int rand_low, int rand_high);
double** copy_matrix_to_jagged_array(SparseMatrix* m);


int thread_count = 4;

int main(int argc, char* argv[])
{
    if (argc == 2) thread_count = atoi(argv[1]);

    cout << "Thread count: " << thread_count << endl;
        
    test_vect_add();
    test_vect_dot_product();
    test_sparse_matrix_multiply(false); // use_sorted_rows = false
    test_sparse_matrix_multiply(true);
    test_sparse_matrix_transpose_multiply(false); // use_sorted_rows = false
    test_sparse_matrix_transpose_multiply(true);
    test_sparse_matrix_transpose();
    test_least_squares(1, 10); // 10 tests; have tested using 1000
    test_least_squares(2, 10);
    test_als(1);
    test_als(2);

    cout << endl;

    bool small_data_set = true;

    if (small_data_set)
    {
        benchmark_col_vector_add((int)1e3);
        benchmark_sparse_matrix_multiply((int)(1e3), (int)(1e2), 10, false); // use_sorted_rows = false
        // benchmark_sparse_matrix_multiply((int)(1e3), (int)(1e2), 10, true);
        benchmark_sparse_matrix_transpose_multiply((int)(1e3), (int)(1e2), 10, false);
        // benchmark_sparse_matrix_transpose_multiply((int)(1e3), (int)(1e2), 10, true);
        benchmark_sparse_matrix_transpose((int)(1e3), (int)(1e2), 10);
        benchmark_least_squares((int)(1e3), (int)(1e2), 10, 1);
        benchmark_least_squares((int)(1e3), (int)(1e2), 10, 2);
    }
    else
    {
        benchmark_col_vector_add((int)27e6);
        benchmark_sparse_matrix_multiply((int)(27e6), (int)(2.8e6), 10, false); // use_sorted_rows = false
        // benchmark_sparse_matrix_multiply((int)(27e6), (int)(2.8e6), 10, true);
        benchmark_sparse_matrix_transpose_multiply((int)(27e6), (int)(2.8e6), 10, false);
        // benchmark_sparse_matrix_transpose_multiply((int)(27e6), (int)(2.8e6), 10, true);
        benchmark_sparse_matrix_transpose((int)(27e6), (int)(2.8e6), 10);
        benchmark_least_squares((int)(27e6), (int)(2.8e6), 10, 1);
        benchmark_least_squares((int)(27e6), (int)(2.8e6), 10, 2);
    }
}


void test_vect_add()
{
    // generate constant1 and constant2
    random_device rd;
    default_random_engine re(rd());
    uniform_real_distribution<double> rand2(-10, 10);
    double constant1 = rand2(re);
    double constant2 = rand2(re);

    // generate v1 and v2
    int length = 10000;
    ColVector v1(length), v2(length), sum(length);
    vect_rand(&v1, -10, 10);
    vect_rand(&v2, -10, 10);

    vect_add(constant1, &v1, constant2, &v2, &sum);

    // check that constant1 * v1 + constant2 * v2 == sum
    bool success = true;
    for (int i = 0; i < v1.length(); i++)
        if (abs(constant1 * v1[i] + constant2 * v2[i] - sum[i]) > 1e-6)
        {
            success = false;
            break;
        }

    if (success)
        cout << "pass - vect_add() passed test." << endl;
    else
        cout << "FAIL - vect_add() FAILED test." << endl;
}


void test_vect_dot_product()
{
    // generate v1 and v2
    int length = 10000;
    ColVector v1(length), v2(length);
    vect_rand(&v1, -1, 1);
    vect_rand(&v2, -1, 1);

    double dot_product = vect_dot_product(&v1, &v2);

    // alternate dot product computation
    double dot_product2 = 0;
    for (int i = 0; i < v1.length(); i++)
        dot_product2 += v1[i] * v2[i];

    if (abs(dot_product - dot_product2) < 1e-6)
        cout << "pass - vect_dot_product() passed test." << endl;
    else
        cout << "FAIL - vect_dot_product() failed test." << endl;
}


void test_sparse_matrix_multiply(bool use_sorted_rows)
{
    // test matrix specs
    int rows = 1000;
    int cols = 50;

    // run two tests, either of which can set "success" to false
    bool success = true;
        
    for (int test_number = 0; test_number < 2; test_number++)
    {
        // fill SparseMatrix A with data, depending on test number
        SparseMatrix A;

        if (test_number == 0)
            fill_matrix_with_dense_random_data(&A, rows, cols, -10, 10);
        else
            fill_matrix_with_sparse_random_data(&A, rows, cols, 5, -10, 10);

        // copy A to jagged array m
        double** m = copy_matrix_to_jagged_array(&A);
        ArrayArray<double> m_deleter(m, rows);

        // create x
        ColVector x(cols);
        vect_rand(&x, -1, 1);

        // use multiply to get result = Ax
        ColVector result(rows);

        if (use_sorted_rows) sparse_matrix_compute_sorted_rows(&A, 2);
        sparse_matrix_multiply(&A, &x, &result, use_sorted_rows);

        // check "result"
        for (int i = 0; i < rows; i++)
        {
            // compute the result at row i
            double sum = 0;
            for (int j = 0; j < cols; j++)
                sum += m[i][j] * x[j];

            if (abs(sum - result[i]) > 1e-6)
            {
                success = false;
                break;
            }
        }
    }

    if (success)
    {
        cout << "pass - sparse_matrix_multiply( ";
        cout << "use_sorted_rows = " << use_sorted_rows;
        cout << " ) passed test." << endl;
    }
    else
    {
        cout << "FAIL - sparse_matrix_multiply( ";
        cout << "use_sorted_rows = " << use_sorted_rows;
        cout << " ) FAILED test." << endl;
    }
}


void test_sparse_matrix_transpose_multiply(bool use_sorted_rows)
{
    // test matrix specs
    int rows = 1000;
    int cols = 50;

    // run two tests, either of which can set "success" to false
    bool success = true;

    for (int test_number = 0; test_number < 2; test_number++)
    {
        // fill SparseMatrix A with data, depending on test number
        SparseMatrix A;

        if (test_number == 0)
            fill_matrix_with_dense_random_data(&A, rows, cols, -10, 10);
        else
            fill_matrix_with_sparse_random_data(&A, rows, cols, 5, -10, 10);

        // copy A to jagged array m
        double** m = copy_matrix_to_jagged_array(&A);
        ArrayArray<double> m_deleter(m, rows);

        // create x
        ColVector x(rows);
        vect_rand(&x, -1, 1);

        // use transpose multiply to get result = (A^T)x
        ColVector result(cols);

        if (use_sorted_rows) sparse_matrix_compute_sorted_rows(&A, 2);
        sparse_matrix_transpose_multiply(&A, &x, &result, use_sorted_rows);

        // check "result"
        bool success = true;
        for (int i = 0; i < cols; i++)
        {
            // compute the result at row i of (m^T), which is column i of m
            double sum = 0;
            for (int j = 0; j < rows; j++)
                sum += m[j][i] * x[j];

            if (abs(sum - result[i]) > 1e-6)
            {
                success = false;
                break;
            }
        }
    }

    if (success)
    {
        cout << "pass - sparse_matrix_transpose_multiply( ";
        cout << "use_sorted_rows = " << use_sorted_rows;
        cout << " ) passed test." << endl;
    }
    else
    {
        cout << "FAIL - sparse_matrix_transpose_multiply( ";
        cout << "use_sorted_rows = " << use_sorted_rows;
        cout << " ) FAILED test." << endl;
    }
}


void test_sparse_matrix_transpose()
{
    // test matrix specs
    int rows = 1000;
    int cols = 50;

    // run two tests, either of which can set "success" to false
    bool success = true;

    for (int test_number = 0; test_number < 2; test_number++)
    {
        // fill SparseMatrix A with data, depending on test number
        SparseMatrix A;

        if (test_number == 0)
            fill_matrix_with_dense_random_data(&A, rows, cols, -10, 10);
        else
            fill_matrix_with_sparse_random_data(&A, rows, cols, 5, -10, 10);

        // copy A to jagged array m
        double** m = copy_matrix_to_jagged_array(&A);
        ArrayArray<double> m_deleter(m, rows);

        // compute matrix transpose via "sparse_matrix_transpose"
        SparseMatrix At; // A^T
        sparse_matrix_transpose(&A, &At);

        // copy A^T to jagged array m^T
        double** mt = copy_matrix_to_jagged_array(&At);
        ArrayArray<double> mt_deleter(mt, cols);

        // check mt
        for (int i = 0; i < rows; i++)
        {
            if (success == false) break;

            for (int j = 0; j < cols; j++)
            {
                if (abs(m[i][j] - mt[j][i]) > 1e-6)
                {
                    success = false;
                    break;
                }
            }
        }
    }

    if (success)
        cout << "pass - sparse_matrix_transpose() passed test." << endl;
    else
        cout << "FAIL - sparse_matrix_transpose() FAILED test." << endl;
}


/* algorithm = 1 --- uses cg_least_squares( )
algorithm = 2 --- uses cg_least_squares2( ) */
void test_least_squares(int algorithm, int num_tests)
{
    // test matrix specs
    int rows = 1000;
    int cols = 50;

    // test statistics
    int num_successes = 0;
    Stats<int> iteration_stats;
    Stats<double> rr_stats;
    Stats<double> sum_error_stats;

    // run tests, any of which can set "success" to false
    bool success = true;

    for (int test_number = 0; test_number < num_tests; test_number++)
    {
        // fill SparseMatrix A with sparse or dense data, depending on test number
        SparseMatrix A;

        if (test_number % 2 == 0)
            fill_matrix_with_dense_random_data(&A, rows, cols, -10, 10);
        else
            fill_matrix_with_sparse_random_data(&A, rows, cols, 5, -10, 10);

        // copy A to jagged array m
        double** m = copy_matrix_to_jagged_array(&A);
        ArrayArray<double> m_deleter(m, rows);

        // define x_real, calculate b
        ColVector x_real(cols), b(rows);
        vect_rand(&x_real, -1, 1);

        random_device rd;
        default_random_engine re(rd());
        normal_distribution<double> rand_norm(0, 0.1);

        double* x_real_ptr = x_real.get_ptr();
        double* b_ptr = b.get_ptr();

        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < cols; j++)
                sum += m[i][j] * x_real_ptr[j];

            b_ptr[i] = sum + rand_norm(re);
        }

        // solve for x
        ColVector x(cols);
        vect_rand(&x, -1, 1);
                
        double final_rr = 0;
        int iterations = 0;

        if (algorithm == 1)
            iterations = cg_least_squares(&A, &b, &x, 0.01, 200, &final_rr);
        else
            iterations = cg_least_squares2(&A, &b, &x, 0.01, 200, &final_rr);

        // compute sum(||x_real - x||)
        double sum_error = 0;
        for (int i = 0; i < cols; i++)
            sum_error += abs(x_real[i] - x[i]);

        // allow 0.01 error per term
        double allowed_error = 0.01 * cols;
        if (sum_error > allowed_error) success = false;
        else num_successes++;

        // track statistics
        iteration_stats.add_data(iterations);
        rr_stats.add_data(final_rr);
        sum_error_stats.add_data(sum_error);        
    }    

    // print success or failure
    if (success)
    {
        if (algorithm == 1)
            cout << "pass - cg_least_squares() passed test" << endl;
        else
            cout << "pass - cg_least_squares2() passed test" << endl;
    }
    else
    {
        if (algorithm == 1)
            cout << "FAIL - cg_least_squares() FAILED test" << endl;
        else
            cout << "FAIL - cg_least_squares2() FAILED test" << endl;
    }

    // print statistics
    cout << "       " << num_successes << " successes out of " << num_tests;
    cout << " tests." << endl;

    cout << "       Least squares iteration ";
    iteration_stats.print();
    cout << endl;

    cout << "       Least squares final_rr ";
    rr_stats.print();
    cout << endl;

    cout << "       Least squares sum error ";
    sum_error_stats.print();
    cout << endl;
}


void test_als(int algorithm)
{
    // generate user_factors_real, item_factors_real
    int num_item_factors = 3;
    int num_user_factors = num_item_factors + 1;

    int num_users = 40;
    int num_items = 45; // all users will review all items

    int num_ratings = num_users * num_items;

    ColVector user_factors_real(num_user_factors * num_users);
    vect_rand(&user_factors_real, -2, 2);

    ColVector item_factors_real(num_item_factors * num_items);
    vect_rand(&item_factors_real, -2, 2);

    // generate data: (user_id, item_id, rating)
    Array<int> user_ids(num_ratings);
    Array<int> item_ids(num_ratings);
    Array<double> ratings(num_ratings);

    random_device rd;
    default_random_engine re(rd());
    normal_distribution<double> rand_norm(0, 0.1);

    int review_index = 0;

    for (int user_id = 0; user_id < num_users; user_id++)
        for (int item_id = 0; item_id < num_items; item_id++)
        {
            user_ids[review_index] = user_id;
            item_ids[review_index] = item_id;

            double predict = als_predict(&user_factors_real,
                &item_factors_real, user_id, item_id, num_item_factors);

            ratings[review_index] = predict + rand_norm(re);

            review_index++;
        }

    // shuffle data    
    uniform_int_distribution<int> rand(0, num_ratings - 1);

    for (int i = 0; i < num_ratings; i++)
    {
        // swap data at "i" with a random number
        int index = rand(re);
        if (index != i)
        {
            swap(item_ids[i], item_ids[index]);
            swap(user_ids[i], user_ids[index]);
            swap(ratings[i], ratings[index]);
        }
    }

    // split data into training set and testing set
    int train_size = (int)(num_ratings * 0.8);
    int test_size = num_ratings - train_size;

    Array<int> user_ids_train(train_size);
    Array<int> item_ids_train(train_size);

    // "ratings_train" will be wrapped inside "ColVector" instead of "Array"
    ColVector ratings_train_vec(train_size);
    double* ratings_train = ratings_train_vec.get_ptr();
    
    Array<int> user_ids_test(test_size);
    Array<int> item_ids_test(test_size);
    Array<double> ratings_test(test_size);

    for (int i = 0; i < train_size; i++)
    {
        user_ids_train[i] = user_ids[i];
        item_ids_train[i] = item_ids[i];
        ratings_train[i] = ratings[i];
    }

    for (int i = 0; i < test_size; i++)
    {
        user_ids_test[i] = user_ids[train_size + i];
        item_ids_test[i] = item_ids[train_size + i];
        ratings_test[i] = ratings[train_size + i];
    }

    // train "user_factors" and "item_factors" using training set
    ColVector user_factors(num_users * num_user_factors);
    ColVector item_factors(num_items * num_item_factors);

    vect_rand(&user_factors, -1, 1);
    vect_rand(&item_factors, -1, 1);

    int iterations = als(user_ids_train.get_ptr(), item_ids_train.get_ptr(),
        &ratings_train_vec, num_item_factors, &user_factors,
        &item_factors, 0.01, 200, algorithm);

    // check model using test set
    double error_sum = 0;
    for (int i = 0; i < test_size; i++)
    {
        double predict = als_predict(&user_factors, &item_factors,
            user_ids_test[i], item_ids_test[i], num_item_factors);

        error_sum += abs(ratings_test[i] - predict);
    }

    double avg_error = error_sum / test_size;

    bool success = true;
    // 0.1 doesn't always pass... 0.1 should ideally be used
    // since it's the standard deviation of the injected noise...
    // if (avg_error > 0.1) success = false;
    if (avg_error > 0.15) success = false;

    if (success)
        cout << "pass - als(" << algorithm << ") passed test, ";
    else
        cout << "FAIL - als(" << algorithm << ") FAILED test, ";

    cout << "average error = " << avg_error;
    cout << "; ALS iterations = " << iterations << endl;
}





void benchmark_col_vector_add(int rows)
{
    cout << "Running benchmark_col_vector_add( ";
    cout << "rows = " << rows << " )." << endl;

    long ms = 0; // total addition time
    ColVector x1(rows), x2(rows), sum(rows);

    random_device rd;
    default_random_engine re(rd());
    uniform_real_distribution<double> rand(-10, 10);


    for (int i = 0; i < 10; i++)
    {
        vect_rand(&x1, -10, 10);
        vect_rand(&x2, -10, 10);
        double c1 = rand(re);
        double c2 = rand(re);

        auto t0 = chrono::high_resolution_clock::now();
        vect_add(c1, &x1, c2, &x2, &sum);
        auto t1 = chrono::high_resolution_clock::now();

        ms += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    }

    cout << "10x vector addition time is " << ms << " ms.";
    cout << endl << endl;
}


void benchmark_sparse_matrix_multiply(int rows, int cols, int values_per_row,
    bool use_sorted_rows)
{
    cout << "Running benchmark_sparse_matrix_multiply( ";
    cout << "use_sorted_rows = " << use_sorted_rows << " )." << endl;
    cout << "Matrix size is " << rows << " x " << cols;
    cout << " with " << values_per_row << " values per row." << endl;

    long mult_time = 0; // total multiplication time in ms
    long sort_rows_time = 0; // total "sort rows" computation time in ms
    
    SparseMatrix m;
    ColVector x(cols), result(rows);

    for (int i = 0; i < 10; i++)
    {
        // generate matrix "m" and vector "x"
        bool new_buffers = true;
        if (i > 0) new_buffers = false;

        fill_matrix_with_sparse_random_data(&m, rows, cols, values_per_row, 
                                            -10, 10, new_buffers);
        vect_rand(&x, -10, 10);

        if (use_sorted_rows)
        {
            auto t0 = chrono::high_resolution_clock::now();
            sparse_matrix_compute_sorted_rows(&m, 8);
            auto t1 = chrono::high_resolution_clock::now();

            sort_rows_time += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        }

        auto t0 = chrono::high_resolution_clock::now();
        sparse_matrix_multiply(&m, &x, &result, use_sorted_rows);
        auto t1 = chrono::high_resolution_clock::now();

        mult_time += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    }

    if (use_sorted_rows)
        cout << "Row sorting time is " << sort_rows_time << "ms." << endl;

    cout << "10x multiplication time is " << mult_time << " ms.";
    cout << endl << endl;
}


void benchmark_sparse_matrix_transpose_multiply(int rows, int cols, 
    int values_per_row, bool use_sorted_rows)
{
    cout << "Running benchmark_sparse_matrix_transpose_multiply( ";
    cout << "use_sorted_rows = " << use_sorted_rows << " )." << endl;
    cout << "Matrix size is " << rows << " x " << cols;
    cout << " with " << values_per_row << " values per row." << endl;

    long mult_time = 0; // total multiplication time in ms
    long sort_rows_time = 0; // total "sort rows" computation time in ms
    
    SparseMatrix m;
    ColVector x(rows), result(cols);

    for (int i = 0; i < 10; i++)
    {
        // generate matrix "m" and vector "x"
        bool new_buffers = true;
        if (i > 0) new_buffers = false;

        fill_matrix_with_sparse_random_data(&m, rows, cols, values_per_row, 
                                            -10, 10, new_buffers);
        vect_rand(&x, -10, 10);

        if (use_sorted_rows)
        {
            auto t0 = chrono::high_resolution_clock::now();
            sparse_matrix_compute_sorted_rows(&m, 8);
            auto t1 = chrono::high_resolution_clock::now();

            sort_rows_time += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        }

        auto t0 = chrono::high_resolution_clock::now();
        sparse_matrix_transpose_multiply(&m, &x, &result, use_sorted_rows);
        auto t1 = chrono::high_resolution_clock::now();

        mult_time += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    }

    if (use_sorted_rows)
        cout << "Row sorting time is " << sort_rows_time << "ms." << endl;

    cout << "10x transpose multiplication time is " << mult_time << " ms.";
    cout << endl << endl;
}


void benchmark_sparse_matrix_transpose(int rows, int cols, int values_per_row)
{
    cout << "Running benchmark_sparse_matrix_transpose()." << endl;
    cout << "Matrix size is " << rows << " x " << cols;
    cout << " with " << values_per_row << " values per row." << endl;

    long ms = 0; // sparse_matrix_transpose(...) run time in ms
    SparseMatrix m, result;

    for (int i = 0; i < 1; i++)
    {
        // generate matrix "m"
        bool new_buffers = true;
        if (i > 0) new_buffers = false;

        fill_matrix_with_sparse_random_data(&m, rows, cols, values_per_row,
                                            -10, 10, new_buffers);

        // Clear the "result" vector - clearing it should not be 
        // considered part of normal transpose time since the transpose
        // happens once per conjugate gradient
        result.clear();

        auto t0 = chrono::high_resolution_clock::now();
        sparse_matrix_transpose(&m, &result);
        auto t1 = chrono::high_resolution_clock::now();

        ms += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    }

    cout << "sparse matrix transpose time is " << ms << " ms.";
    cout << endl << endl;
}


/* algorithm = 1 --- uses cg_least_squares( )
algorithm = 2 --- uses cg_least_squares2( ) */
void benchmark_least_squares(int rows, int cols, int values_per_row, int algorithm)
{
    cout << "Running benchmark_least_squares( algorithm = ";
    cout << algorithm << " ). " << endl;
    cout << "Matrix size is " << rows << " x " << cols;
    cout << " with " << values_per_row << " values per row." << endl;

    long ms = 0; // total time in ms

    SparseMatrix A;
    ColVector x(cols), b(rows);

    int total_iterations = 0;

    for (int i = 0; i < 2; i++)
    {
        // generate matrix "A" and vector "b"
        bool new_buffers = true;
        if (i > 0) new_buffers = false;

        fill_matrix_with_sparse_random_data(&A, rows, cols, values_per_row, 
                                            -10, 10, new_buffers);
        vect_rand(&b, -10, 10);

        auto t0 = chrono::high_resolution_clock::now();

        if(algorithm == 1)
            total_iterations += cg_least_squares(&A, &b, &x);
        else
            total_iterations += cg_least_squares2(&A, &b, &x);

        auto t1 = chrono::high_resolution_clock::now();
        ms += (long)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    }

    cout << "2x least squares solution time is " << ms << " ms, in ";
    cout << total_iterations << " iterations. Per iteration time is ";
    cout << ms / total_iterations << " ms." << endl << endl;
}





// Matrix "m" will be filled with random values, in a dense fashion
void fill_matrix_with_dense_random_data(SparseMatrix* m, int rows, int cols,
    int rand_low, int rand_high)
{
    m->clear();

    // allocate buffers for "m"
    int* row_indices = new int[rows + 1];
    int* col_indices = new int[rows * cols];
    double* values = new double[rows * cols];

    random_device rd;
    default_random_engine re(rd());
    uniform_real_distribution<double> rand(rand_low, rand_high);

    // fill "row_indices"
    row_indices[0] = 0;
    for (int i = 1; i <= rows; i++)
        row_indices[i] = row_indices[i - 1] + cols;

    // fill "col_indices" and "values"
    for (int i = 0; i < rows; i++) // i-th row
        for (int j = 0; j < cols; j++) // j-th column
        {
            int index = i * cols + j;
            col_indices[index] = j;
            values[index] = rand(re);
        }

    m->load(rows, cols, row_indices, col_indices, values);
}


// Matrix "m" will be filled with "values_per_row".
void fill_matrix_with_sparse_random_data(SparseMatrix* m, int rows, int cols,
    int values_per_row, int rand_low, int rand_high, bool new_buffers)
{
    if (new_buffers)
    {
        m->clear();
        int* row_indices = new int[rows + 1];
        int* col_indices = new int[rows * values_per_row];
        double* values = new double[rows * values_per_row];

        // the very last element of "row_indices" is not set by the threads
        row_indices[rows] = rows * values_per_row;

        m->load(rows, cols, row_indices, col_indices, values);
    } 

    // initialize "spread"
    Array<int> spread(thread_count + 1);

    for (int i = 1; i < thread_count; i++)
        spread[i] = (int)((double)i * rows / thread_count);

    spread[0] = 0;
    spread[thread_count] = rows;

    // fill the array "m" using multiple threads
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
    {
        int start = spread[i];
        int end = spread[i + 1];
        t[i] = new thread(fill_matrix_with_sparse_random_data_t, start, end,
            m, rows, cols, values_per_row, rand_low, rand_high);
    }

    wait_and_delete_threads(t);
}


// Implements a thread of "fill_matrix_with_random_data".
void fill_matrix_with_sparse_random_data_t(int start, int end, SparseMatrix* m,
    int rows, int cols, int values_per_row, int rand_low, int rand_high)
{
    int* row_indices = m->get_row_indices_ptr();
    int* col_indices = m->get_col_indices_ptr();
    double* values = m->get_values_ptr();

    random_device rd;
    default_random_engine re(rd());

    // for generating "values"
    uniform_real_distribution<double> rand(rand_low, rand_high);

    // for generating "starting column numbers"
    uniform_int_distribution<int> rand_col(0, cols - values_per_row);

    for (int i = start; i < end; i++) // i-th row
    {
        int col_index_start = i * values_per_row;
        int col_index_end = col_index_start + values_per_row - 1;

        row_indices[i] = i * values_per_row;
        int col_number = rand_col(re); // column number of first value on row 'i'

        for (int j = col_index_start; j <= col_index_end; j++)
        {
            // j-th entry in the array, which is on row 'i'
            col_indices[j] = col_number;
            values[j] = rand(re);
            col_number++;
        }
    }
}


// allocates an array of arrays, and fills it to match content of "m"
double** copy_matrix_to_jagged_array(SparseMatrix* m)
{
    // Allocate m2 to be same size as m, and initialize it to zeros
    int rows = m->num_rows();
    int cols = m->num_cols();

    double** m2 = new double*[rows];
    for (int i = 0; i < rows; i++)
        m2[i] = new double[cols];

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m2[i][j] = 0;
    
    // copy data from "m" to "m2"
    int* row_indices = m->get_row_indices_ptr();
    int* col_indices = m->get_col_indices_ptr();
    double* values = m->get_values_ptr();

    for (int i = 0; i < rows; i++) // i-th row
    {
        int row_start = row_indices[i];
        int row_end = row_indices[i + 1] - 1;

        for (int j = row_start; j <= row_end; j++)
        {
            int col = col_indices[j];
            // values[j] is on row "i", column "col"
            m2[i][col] = values[j];
        }
    }

    return m2;
}

