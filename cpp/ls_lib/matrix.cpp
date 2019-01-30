#include "matrix.h"

extern int thread_count;


// initialize spread
void ColVector::init_spread()
{    
    spread = new int[thread_count + 1];

    for (int i = 1; i < thread_count; i++)
        spread[i] = (int)(((float)i) / (thread_count)* _length);

    spread[0] = 0;
    spread[thread_count] = _length;
}


ColVector::ColVector(int length)
{
    _length = length;
    values = new double[length];

    init_spread();
}


ColVector::ColVector(double* values, int length, bool delete_values)
{
    _length = length;
    this->values = values;
    this->delete_values = delete_values;

    init_spread();
}


// randomize a subset of the vector with the uniform distribution
void ColVector::rand(int thread_id, double low, double high)
{
    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    random_device rd;
    default_random_engine re(rd());
    uniform_real_distribution<double> rand2(low, high);

    for (int i = start; i < end; i++)
        values[i] = rand2(re);
}


// randomize a subset of the vector with the normal distribution
void ColVector::rand_norm(int thread_id, double mean, double stddev)
{
    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    random_device rd;
    default_random_engine re(rd());
    normal_distribution<double> rand_norm(mean, stddev);

    for (int i = start; i < end; i++)
        values[i] = rand_norm(re);
}


// constant1 * (this) + constant2 * v2 => sum
void ColVector::add(int thread_id, double constant1, double constant2, ColVector* v2,
    ColVector* sum)
{
    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    for (int i = start; i < end; i++)
        sum->values[i] = constant1 * values[i] + constant2 * v2->values[i];
}


// (this) * constant => result
void ColVector::times_constant(int thread_id, double constant, ColVector* result)
{
    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    for (int i = start; i < end; i++)
        result->values[i] = values[i] * constant;
}


// (this) dot product v2 ==> dot_product
void ColVector::dot_product(int thread_id, ColVector* v2, double* dot_product)
{
    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    double sum = 0;
    for (int i = start; i < end; i++)
        sum += values[i] * v2->values[i];

    *dot_product = sum;
}


// (this) = vectors[0] + vectors[1] + ...
void ColVector::add_merge(int thread_id, ColVector** vectors, int num_vectors)
{
    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    // copy vectors[0] to (this)
    double* vector = vectors[0]->get_ptr();

    for (int i = start; i < end; i++) // i-th row
        values[i] = vector[i];
    
    // add other vectors
    for (int j = 1; j < num_vectors; j++) // j-th vector
    {
        vector = vectors[j]->get_ptr();

        for (int i = start; i < end; i++) // i-th row
            values[i] += vector[i];
    }
}




void SparseMatrix::clear()
{
    if (rows == 0) return;
    rows = 0;
    cols = 0;
    
    if (delete_values)
    {
        if (row_indices != nullptr)
            delete[] row_indices;
         
        if (col_indices != nullptr)
            delete[] col_indices;
            
        if (values != nullptr)
            delete[] values;
    }

    // regardless of the "delete_values" flag, clear always reset the pointers
    row_indices = nullptr;
    col_indices = nullptr;
    values = nullptr;

    delete_values = true;
    
    if (spread != nullptr)
    {
        delete[] spread;
        spread = nullptr;
    }
}


void SparseMatrix::load(int rows, int cols, int* row_indices, 
    int* col_indices, double* values, bool delete_values)
{
    clear();

    this->rows = rows;
    this->cols = cols;
    this->delete_values = delete_values;

    this->row_indices = row_indices;
    this->col_indices = col_indices;
    this->values = values;

    // initialize spread
    spread = new int[thread_count + 1];

    for (int i = 1; i < thread_count; i++)
        spread[i] = (int)(((float)i) / (thread_count)* rows);

    spread[0] = 0;
    spread[thread_count] = rows;
}


// (this) * x => result
void SparseMatrix::multiply(int thread_id, ColVector* x, ColVector* result)
{
    double* x_vec = x->get_ptr();
    double* result_vec = result->get_ptr();

    int start = spread[thread_id];
    int end = spread[thread_id + 1];
        
    for (int i = start; i < end; i++) // i-th row
    {
        int row_number = i;

        int row_start = row_indices[row_number];
        int row_end = row_indices[row_number + 1] - 1;

        double sum = 0;
        for (int j = row_start; j <= row_end; j++) 
        {
            // values[j] is on row 'i', at location col_indices[j]
            int col_number = col_indices[j];
            sum += values[j] * x_vec[col_number];
        }

        result_vec[row_number] = sum;
    }
}


// (this^T) * x => result
void SparseMatrix::transpose_multiply(int thread_id, ColVector* x, 
    ColVector* result)
{
    double* x_vec = x->get_ptr();
    double* result_vec = result->get_ptr();

    // initialize "result_vec" to zero
    for (int i = 0; i < cols; i++)
        result_vec[i] = 0;

    int start = spread[thread_id];
    int end = spread[thread_id + 1];

    for (int i = start; i < end; i++) // i-th row
    {
        int row_number = i;

        int row_start = row_indices[row_number];
        int row_end = row_indices[row_number + 1] - 1;

        // c_i = the constant that is paired with i-th row
        double c_i = x_vec[row_number];

        for (int j = row_start; j <= row_end; j++)
        {
            // values[j] is on row 'i', at location col_indices[j]
            int col_number = col_indices[j];
            result_vec[col_number] += c_i * values[j];
        }
    }
}


// count the number of elements in each column
void SparseMatrix::count_columns(int thread_id, int* column_counts)
{
    // initialize counts to zero
    for (int i = 0; i < cols; i++)
        column_counts[i] = 0;

    int start_row = spread[thread_id];
    int start = row_indices[start_row]; // the start index

    int end_row = spread[thread_id + 1]; // this row is not covered by this thread
    int end = row_indices[end_row] - 1; // the final index to look at

    for (int i = start; i <= end; i++) // i-th element in "values"
    {
        int col = col_indices[i];
        column_counts[col]++;
    }
}


// Copy data to new "values" and "indices", using a column major format.
// This is the final step in the transpose process.
void SparseMatrix::copy_to_column_major_form(int thread_id, int* new_col_indices,
    int* new_row_indices, double* new_values)
{
    int first_row = spread[thread_id];
    int last_row = spread[thread_id + 1] - 1; // inclusive end

    for (int row = first_row; row <= last_row; row++)
    {
        int row_start = row_indices[row];
        int row_end = row_indices[row + 1] - 1;

        for (int j = row_start; j <= row_end; j++)
        {
            int col = col_indices[j];
            // values[j] is at row "row", column "col"

            int new_index = new_col_indices[col];
            new_values[new_index] = values[j];
            new_row_indices[new_index] = row;

            // for future elements in this column:
            new_col_indices[col]++;
        }
    }
}






// call join on all threads and delete threads when finished
void wait_and_delete_threads(thread** threads)
{
    for (int i = 0; i < thread_count; i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    delete[] threads;
}


// randomize the content of "v" using uniform distribution
void vect_rand(ColVector* v, double low, double high)
{
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([v, i, low, high]() {v->rand(i, low, high); });

    wait_and_delete_threads(t);
}


// randomize the content of "v" using normal distribution
void vect_rand_norm(ColVector* v, double mean, double stddev)
{
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([v, i, mean, stddev]()
                          {v->rand_norm(i, mean, stddev); });

    wait_and_delete_threads(t);
}


// constant1 * v1 + constant2 * v2 => sum
void vect_add(double constant1, ColVector* v1, double constant2,
    ColVector* v2, ColVector* sum)
{
    if (thread_count <= 1)
        v1->add(0, constant1, constant2, v2, sum);

    else
    {
        thread** t = new thread*[thread_count];

        for (int i = 0; i < thread_count; i++)
            t[i] = new thread([v1, i, constant2, constant1, v2, sum]()
        {v1->add(i, constant1, constant2, v2, sum); });

        wait_and_delete_threads(t);
    }
}


// v * constant => result
void vect_times_constant(ColVector* v, double constant, ColVector* result)
{
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([v, i, constant, result]()
                          {v->times_constant(i, constant, result); });

    wait_and_delete_threads(t);
}

// v1 dot product v2
double vect_dot_product(ColVector* v1, ColVector* v2)
{
    thread** t = new thread*[thread_count];
    double* sums = new double[thread_count];

    for (int i = 0; i < thread_count; i++)
    {
        double* sum = &sums[i];
        t[i] = new thread([v1, i, v2, sum]() {v1->dot_product(i, v2, sum); });
    }

    wait_and_delete_threads(t);

    double dot_product = 0;

    for (int i = 0; i < thread_count; i++)
        dot_product += sums[i];

    delete[] sums;

    return dot_product;
}


// m * x => result
void sparse_matrix_multiply(SparseMatrix* m, ColVector* x, ColVector* result)
{
    // check for dimensional mismatch
    if ((m->num_cols() != x->length())
        || (m->num_rows() != result->length()))
        throw "sparse_matrix_multiply() dimensional mismatch error";

    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([m, i, x, result]()
                          {m->multiply(i, x, result); });

    wait_and_delete_threads(t);
}


// m^T * x => result
void sparse_matrix_transpose_multiply(SparseMatrix* m, ColVector* x, 
    ColVector* result)
{
    // check for dimensional mismatch
    if ((m->num_rows() != x->length())
        || (m->num_cols() != result->length()))
        throw "sparse_matrix_transpose_multiply() dimensional mismatch error";

    // Create one column vector per thread id
    ColVector** col_vectors = new ColVector*[thread_count];
    for (int i = 0; i < thread_count; i++)
        col_vectors[i] = new ColVector(result->length());

    ArrayPtr<ColVector> array_ptr_wrapper(col_vectors, thread_count);
            
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([m, i, x, col_vectors]()
            {m->transpose_multiply(i, x, col_vectors[i]); });

    wait_and_delete_threads(t);

    // Merge column vectors together
    t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([result, i, col_vectors]()
                          {result->add_merge(i, col_vectors, thread_count); });

    wait_and_delete_threads(t);
}


/* Solve Ax = b in the least squares sense, using conjugate gradient.
Caller needs to initialize "x". This is the low memory version that uses
transpose multiplication to evaluate A^T * vector. Returns number of
iterations used. */
int cg_least_squares(SparseMatrix* A, ColVector* b, ColVector* x,
    double min_r_decrease, int max_iteration, double* final_rr)
{
    // dimensions
    int r = A->num_rows();
    int c = A->num_cols();

    // b2 = At b
    ColVector b2(c);
    sparse_matrix_transpose_multiply(A, b, &b2);

    // r0 = (A^T A)x - b2; here Ap = (A^T A)x, later it will be (A^T A)p
    ColVector r_vec(c), temp(r), Ap(c);
    sparse_matrix_multiply(A, x, &temp); // temp = A x
    sparse_matrix_transpose_multiply(A, &temp, &Ap); // Ap = A^T temp = A^T A x
        
    vect_add(1, &Ap, -1, &b2, &r_vec);

    // p0 = -r0
    ColVector p(c);
    vect_times_constant(&r_vec, -1, &p);

    int iteration = 0;

    // The algorithm does not quit immediately for "min_r_decrease" failures
    // Record number of failures.
    int failed_min_r_decrease_count = 0;

    // rr = rk dot rk
    double rr = vect_dot_product(&r_vec, &r_vec);
    if (final_rr != nullptr) *final_rr = rr;

    while (iteration < max_iteration)
    {
        if (rr < 1e-6) return iteration;

        // Ap = (At A pk)
        sparse_matrix_multiply(A, &p, &temp); // temp = A p
        sparse_matrix_transpose_multiply(A, &temp, &Ap); // Ap = A^T temp = A^T A p

        // alpha_k = rr / (p dot Ap)
        double p_dot_Ap = vect_dot_product(&p, &Ap);
        double alpha = rr / p_dot_Ap;

        // x_k+1 = x_k + alpha_k * p_k
        vect_add(1, x, alpha, &p, x);

        // r_k+1 = r_k + alpha A p_k
        vect_add(1, &r_vec, alpha, &Ap, &r_vec);

        // beta = (r_k+1 dot r_k+1) / (r_k dot r_k)
        double rr2 = vect_dot_product(&r_vec, &r_vec);
        if (final_rr != nullptr) *final_rr = rr2;

        double beta = rr2 / rr;

        // termination check - exit if rr2 is decreasing too slowly
        if (beta > 1 - min_r_decrease)
            failed_min_r_decrease_count++;
        else
            failed_min_r_decrease_count = 0;

        if (failed_min_r_decrease_count >= 2) return iteration;

        // p_k+1 = -r_k+1 + Beta p_k
        vect_add(-1, &r_vec, beta, &p, &p);

        // for the next iteration:
        rr = rr2;
        iteration++;
    }

    return iteration;
}


/* Solve Ax = b in the least squares sense, using conjugate gradient.
Caller needs to initialize "x". This version computes the matrix A^T,
so it uses more memory, but can be faster if there are sufficient number
of cores. Returns number of iterations used. */
int cg_least_squares2(SparseMatrix* A, ColVector* b, ColVector* x,
    double min_r_decrease, int max_iteration, double* final_rr)
{
    // dimensions
    int r = A->num_rows();
    int c = A->num_cols();

    // compute At = A^T
    SparseMatrix At;
    sparse_matrix_transpose(A, &At);

    // b2 = At b
    ColVector b2(c);
    sparse_matrix_multiply(&At, b, &b2);

    // r0 = (A^T A)x - b2; here Ap = (A^T A)x, later it will be (A^T A)p
    ColVector r_vec(c), temp(r), Ap(c);
    sparse_matrix_multiply(A, x, &temp); // temp = A x
    sparse_matrix_multiply(&At, &temp, &Ap); // Ap = A^T temp = A^T A x

    vect_add(1, &Ap, -1, &b2, &r_vec);

    // p0 = -r0
    ColVector p(c);
    vect_times_constant(&r_vec, -1, &p);

    int iteration = 0;

    // The algorithm does not quit immediately for "min_r_decrease" failures
    // Record number of failures.
    int failed_min_r_decrease_count = 0;

    // rr = rk dot rk
    double rr = vect_dot_product(&r_vec, &r_vec);
    if (final_rr != nullptr) *final_rr = rr;

    while (iteration < max_iteration)
    {
        if (rr < 1e-6) return iteration;

        // Ap = (At A pk)
        sparse_matrix_multiply(A, &p, &temp); // temp = A p
        sparse_matrix_multiply(&At, &temp, &Ap); // Ap = A^T temp = A^T A p

        // alpha_k = rr / (p dot Ap)
        double p_dot_Ap = vect_dot_product(&p, &Ap);
        double alpha = rr / p_dot_Ap;

        // x_k+1 = x_k + alpha_k * p_k
        vect_add(1, x, alpha, &p, x);

        // r_k+1 = r_k + alpha A p_k
        vect_add(1, &r_vec, alpha, &Ap, &r_vec);

        // beta = (r_k+1 dot r_k+1) / (r_k dot r_k)
        double rr2 = vect_dot_product(&r_vec, &r_vec);
        if (final_rr != nullptr) *final_rr = rr2;

        double beta = rr2 / rr;

        // termination check - exit if rr2 is decreasing too slowly
        if (beta > 1 - min_r_decrease)
            failed_min_r_decrease_count++;
        else
            failed_min_r_decrease_count = 0;

        if (failed_min_r_decrease_count >= 2) return iteration;

        // p_k+1 = -r_k+1 + Beta p_k
        vect_add(-1, &r_vec, beta, &p, &p);

        // for the next iteration:
        rr = rr2;
        iteration++;
    }

    return iteration;
}


// m^T => mt
void sparse_matrix_transpose(SparseMatrix* m, SparseMatrix* mt)
{
    // Allocate "column_counts"
    int** column_counts = new int*[thread_count];
    for (int i = 0; i < thread_count; i++)
        column_counts[i] = new int[m->num_cols()];
    
    ArrayArray<int> column_counts_deleter(column_counts, thread_count);

    // Compute column_counts
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread([m, i, column_counts]()
                          {m->count_columns(i, column_counts[i]); });

    wait_and_delete_threads(t);

    // Create column oriented "spread" boundaries
    Array<int> spread(thread_count + 1);

    for (int i = 1; i < thread_count; i++)
        spread[i] = (int)(((float)i) / (thread_count)* m->num_cols());

    spread[0] = 0;
    spread[thread_count] = m->num_cols();

    // Allocate "column_count_total". This will eventually be loaded into "mt".
    int* column_count_total = new int[m->num_cols() + 1];

    t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread(add_arrays, spread[i], spread[i + 1] - 1,
            column_counts, column_count_total, thread_count, 1);

    wait_and_delete_threads(t);

    // the zero-th item is not set by the threads (due to col_offset = +1)
    column_count_total[0] = 0;
    
    // accumulate the counts in column_count_total (in place operation)
    for (int col = 1; col < m->num_cols() + 1; col++)
        column_count_total[col] += column_count_total[col - 1];
        
    // col_indices0 + accumulate column_counts --> "col_indices[]"
    int** col_indices = column_counts; // in place operation
        
    t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread(compute_col_indices, spread[i], spread[i + 1] - 1,
            column_count_total, col_indices, thread_count);

    wait_and_delete_threads(t);

    // Allocate "row_indices" and "values". These will eventually be 
    // loaded into "mt".
    int size = column_count_total[m->num_cols()];
    int* row_indices = new int[size];
    double* values = new double[size];

    // Copy data to "row_indices" and "values"
    t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
    {
        t[i] = new thread([m, i, col_indices, row_indices, values]()
                          {m->copy_to_column_major_form(i, col_indices[i],
                            row_indices, values); });
    }

    wait_and_delete_threads(t);

    mt->load(m->num_cols(), m->num_rows(), column_count_total, row_indices, values);
}


/* sum the arrays in arr and store the result in sum, applying a
column offset. The "start_col" and "end_col" are inclusive.*/
void add_arrays(int start_col, int end_col, int** arr, int* sum, int num_rows,
    int col_offset)
{
    for (int col = start_col; col <= end_col; col++)
    {
        int _sum = 0;
        for (int row = 0; row < num_rows; row++)
            _sum += arr[row][col];

        sum[col + col_offset] = _sum;
    }
}


/* compute "col_indices", part of the sparse matrix transpose algorithm.
The "start_col" and "end_col" are inclusive.*/
void compute_col_indices(int start_col, int end_col, int* col_indices_base,
    int** col_indices, int num_rows)
{
    // At the start of this function, the "col_indices" contains 
    // the column counts, aka "column_counts". This is an in place
    // operation to turn "column_counts" into "col_indices".

    // Working down each column, the column count needs to be saved.
    int column_count = 0;

    for (int col = start_col; col <= end_col; col++)
    {
        // the first row is copied off "col_indices_base"
        column_count = col_indices[0][col]; // save this column count
        col_indices[0][col] = col_indices_base[col];
                
        // each row is the previous index + column count
        for (int row = 1; row < num_rows; row++)
        {
            int index = col_indices[row - 1][col] + column_count;
            column_count = col_indices[row][col]; // for next iteration

            col_indices[row][col] = index;
        }
    }
}


/*Derive "user_factors" and "item_factors" from a set of
(user_ids, item_ids, ratings) using the ALS procedure.
Returns number of iterations used.*/
int als(int* user_ids, int* item_ids, ColVector* ratings,
    int num_item_factors, ColVector* user_factors, ColVector* item_factors,
    double min_r_decrease, int max_iteration, int algorithm)
{
    // algorithm = 1 --- use cg_least_squares(...)
    // algorithm = 2 --- use cg_least_squares2(...)

    int num_user_factors = num_item_factors + 1;
    int num_ratings = ratings->length();

    // construct users' A, to be used as: (user_A)(user_factors) = (ratings)
    SparseMatrix user_A;

    int* user_A_row_indices = new int[num_ratings + 1];
    int* user_A_col_indices = new int[num_ratings * num_user_factors];
    double* user_A_values = new double[num_ratings * num_user_factors];

    user_A.load(num_ratings, user_factors->length(), user_A_row_indices,
        user_A_col_indices, user_A_values);

    // initialize spread - representing the set of rows
    Array<int> spread(thread_count + 1);

    for (int i = 1; i < thread_count; i++)
        spread[i] = (int)(((float)i) / (thread_count)* num_ratings);

    spread[0] = 0;
    spread[thread_count] = num_ratings;

    // fill user_A
    thread** t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread(fill_user_A, spread[i], spread[i + 1] - 1,
                          &user_A, user_ids, item_ids, item_factors, 
                          num_item_factors, true); // first fill = true

    wait_and_delete_threads(t);

    // the final element of user_A.row_indices[] is not filled by the threads
    user_A_row_indices[num_ratings] = num_ratings * num_user_factors;

    // construct items' A, to be used as: (item_A)(item_factors) = (ratings)
    SparseMatrix item_A;

    int* item_A_row_indices = new int[num_ratings + 1];
    int* item_A_col_indices = new int[num_ratings * num_item_factors];
    double* item_A_values = new double[num_ratings * num_item_factors];

    item_A.load(num_ratings, item_factors->length(), item_A_row_indices,
        item_A_col_indices, item_A_values);

    // fill item_A
    t = new thread*[thread_count];

    for (int i = 0; i < thread_count; i++)
        t[i] = new thread(fill_item_A, spread[i], spread[i + 1] - 1,
                          &item_A, user_ids, item_ids, user_factors, 
                          num_item_factors, true); // first fill = true

    wait_and_delete_threads(t);

    // the final element of item_A.row_indices[] is not filled by the threads
    item_A_row_indices[num_ratings] = num_ratings * num_item_factors;

    // ALS settings
    int iteration = 0;
    double old_rr = 0;
    ColVector ratings_minus_bias(num_ratings);

    while (iteration < max_iteration)
    {
        // solve for user_factors
        if(algorithm == 1)
            cg_least_squares(&user_A, ratings, user_factors);
        else
        {
            // Computing the transpose is only effective if the least square
            // has many iterations. For now just limit algorithm 2
            // to the very beginning.
            if (iteration < 1)
                cg_least_squares2(&user_A, ratings, user_factors);
            else
                cg_least_squares(&user_A, ratings, user_factors);
        }

        // update items' A
        t = new thread*[thread_count];

        for (int i = 0; i < thread_count; i++)
            t[i] = new thread(fill_item_A, spread[i], spread[i + 1] - 1,
                              &item_A, user_ids, item_ids, user_factors,
                              num_item_factors, false); // first fill = false

        wait_and_delete_threads(t);
        
        // update ratings_minus_bias
        t = new thread*[thread_count];

        for (int i = 0; i < thread_count; i++)
            t[i] = new thread(fill_ratings_minus_bias, spread[i], 
                              spread[i + 1] - 1, &ratings_minus_bias, user_ids,
                              ratings, user_factors, num_item_factors);

        wait_and_delete_threads(t);

        // solve for item_factors
        double rr = 0;
        
        if (algorithm == 1)
            cg_least_squares(&item_A, &ratings_minus_bias, item_factors,
                             0.01, 200, &rr);
        else
        {
            // Computing the transpose is only effective if the least square
            // has many iterations. For now just limit algorithm 2
            // to the very beginning.
            if (iteration < 1)
                cg_least_squares2(&item_A, &ratings_minus_bias, item_factors,
                                  0.01, 200, &rr);
            else
                cg_least_squares(&item_A, &ratings_minus_bias, item_factors,
                                 0.01, 200, &rr);
        }

        // check rr progress
        // if (iteration >= 1) // sometimes exit after 2 iterations
        if (iteration >= 3) // hard code minimum 3 iterations
        {
            double decrease = (old_rr - rr) / old_rr;
            if (decrease < min_r_decrease) return iteration;
        }

        // update users' A
        t = new thread*[thread_count];

        for (int i = 0; i < thread_count; i++)
            t[i] = new thread(fill_user_A, spread[i], spread[i + 1] - 1,
                              &user_A, user_ids, item_ids, item_factors,
                              num_item_factors, false); // first fill = false

        wait_and_delete_threads(t);

        // for the next iteration:
        old_rr = rr;
        iteration++;
    }

    return iteration;
}


/*Fills the "user_A" matrix, which is used as: (user_A)(user_factors) = (ratings).
The "start" and "end" are inclusive. */
void fill_user_A(int row_start, int row_end, SparseMatrix* user_A,
    int* user_ids, int* item_ids, ColVector* item_factors, int num_item_factors,
    bool first_fill)
{
    int num_user_factors = num_item_factors + 1;
    
    if (first_fill)
    {
        // If filling out user_A for the first time, need to fill all three:
        // "row_indices", "col_indices", and "values".
        int* row_indices = user_A->get_row_indices_ptr();
        int* col_indices = user_A->get_col_indices_ptr();
        double* values = user_A->get_values_ptr();
        double* item_factors_ptr = item_factors->get_ptr();

        // "row_indices"        
        for (int row = row_start; row <= row_end; row++)
            row_indices[row] = row * num_user_factors;

        for (int row = row_start; row <= row_end; row++)
        {
            int row_start_index = row * num_user_factors;
            int user_id = user_ids[row];
            int item_id = item_ids[row];

            for (int j = 0; j < num_user_factors; j++) // j-th item of the row
            {                
                // col_indices
                col_indices[row_start_index + j] = user_id * num_user_factors + j;

                // values - it's either from item_factors, 
                // or it's "1" the bias term
                if (j == num_user_factors - 1)
                    values[row_start_index + j] = 1;
                else
                    values[row_start_index + j] = item_factors_ptr[item_id * num_item_factors + j];
            }
        }
    }
    else
    {
        // refresh "values" with latest "item_factors"
        double* values = user_A->get_values_ptr();
        double* item_factors_ptr = item_factors->get_ptr();

        for (int row = row_start; row <= row_end; row++)
        {
            int row_start_index = row * num_user_factors;
            int item_id = item_ids[row];

            for (int j = 0; j < num_item_factors; j++) // j-th item of the row
                values[row_start_index + j] = item_factors_ptr[item_id * num_item_factors + j];
        }
    }
}


/*Fills the "item_A" matrix, which is used as: (item_A)(item_factors) = (ratings_minus_bias).
The "start" and "end" are inclusive.*/
void fill_item_A(int row_start, int row_end, SparseMatrix* item_A,
    int* user_ids, int* item_ids, ColVector* user_factors, int num_item_factors,
    bool first_fill)
{
    int num_user_factors = num_item_factors + 1;

    if (first_fill)
    {
        // If filling out item_A for the first time, need to fill all three:
        // "row_indices", "col_indices", and "values".
        int* row_indices = item_A->get_row_indices_ptr();
        int* col_indices = item_A->get_col_indices_ptr();
        double* values = item_A->get_values_ptr();
        double* user_factors_ptr = user_factors->get_ptr();

        // "row_indices"
        for (int row = row_start; row <= row_end; row++)
            row_indices[row] = row * num_item_factors;

        for (int row = row_start; row <= row_end; row++)
        {
            int row_start_index = row * num_item_factors;
            int user_id = user_ids[row];
            int item_id = item_ids[row];

            for (int j = 0; j < num_item_factors; j++) // j-th item of the row
            {
                // col_indices
                col_indices[row_start_index + j] = item_id * num_item_factors + j;

                // values
                values[row_start_index + j] = user_factors_ptr[user_id * num_user_factors + j];
            }
        }
    }
    else
    {
        // refresh "values" with latest "user_factors"
        double* values = item_A->get_values_ptr();
        double* user_factors_ptr = user_factors->get_ptr();

        for (int row = row_start; row <= row_end; row++)
        {
            int row_start_index = row * num_item_factors;
            int user_id = user_ids[row];

            for (int j = 0; j < num_item_factors; j++) // j-th item of the row
                values[row_start_index + j] = user_factors_ptr[user_id * num_user_factors + j];
        }
    }
}


/*Fills the "ratings_minus_bias" vector, which is the (ratings) vector
minus the (user bias) vector. The "start" and "end" are inclusive.*/
void fill_ratings_minus_bias(int row_start, int row_end,
    ColVector* ratings_minus_bias, int* user_ids, ColVector* ratings,
    ColVector* user_factors, int num_item_factors)
{
    int num_user_factors = num_item_factors + 1;
    double* user_factors_ptr = user_factors->get_ptr();
    double* ratings_minus_bias_ptr = ratings_minus_bias->get_ptr();
    double* ratings_ptr = ratings->get_ptr();

    for (int row = row_start; row <= row_end; row++)
    {
        // get user_bias
        int user_id = user_ids[row];
        int user_bias_index = (user_id + 1) * num_user_factors - 1;
        double user_bias = user_factors_ptr[user_bias_index];

        // ratings_minus_bias = ratings - bias
        ratings_minus_bias_ptr[row] = ratings_ptr[row] - user_bias;
    }
}


// Predict rating using the ALS model provided.
double als_predict(ColVector* user_factors, ColVector* item_factors,
    int user_id, int item_id, int num_item_factors)
{
    int num_user_factors = num_item_factors + 1;
    int user_index = user_id * num_user_factors;
    int item_index = item_id * num_item_factors;

    double* user_factors_ptr = user_factors->get_ptr();
    double* item_factors_ptr = item_factors->get_ptr();

    double sum = 0;

    for (int i = 0; i < num_item_factors; i++)
        sum += user_factors_ptr[user_index + i] * item_factors_ptr[item_index + i];

    sum += user_factors_ptr[user_index + num_user_factors - 1];

    return sum;
}



