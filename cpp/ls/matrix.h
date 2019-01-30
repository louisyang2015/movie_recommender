#ifndef MATRIX_H
#define MATRIX_H

#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace std;


// Container for an array of objects
template <typename T>
class Array
{
    T* arr = nullptr;
    int _length = 0;

public:
    ~Array() { delete[] arr; }

    Array(int length)
    {
        _length = length;
        arr = new T[length];
    }

    int length() { return _length; }
    T& operator[](int i) { return arr[i]; }

    T* get_ptr() { return arr; }
};

// Container for array of pointers to handle delete
template <typename T>
class ArrayPtr
{
    T** arr = nullptr;
    int length = 0;

public:
    ArrayPtr(T** arr, int length)
    {
        this->arr = arr;
        this->length = length;
    }

    ~ArrayPtr()
    {
        for (int i = 0; i < length; i++)
            delete arr[i];

        delete[] arr;
    }
};

// Container for array of array to handle delete
template <typename T>
class ArrayArray
{
    T** arr = nullptr;
    int length = 0;

public:
    ArrayArray(T** arr, int length)
    {
        this->arr = arr;
        this->length = length;
    }

    ~ArrayArray()
    {
        for (int i = 0; i < length; i++)
            delete[] arr[i];

        delete[] arr;
    }
};


class ColVector
{
    int _length = 0;
    double* values = nullptr;
    int* spread = nullptr;

public:
    ~ColVector()
    {
        delete[] values;
        delete[] spread;
    }

    ColVector(int length);

    double* get_ptr() { return values; }
    double& operator[](int i) { return values[i]; }

    int length() { return _length; }
    

    // randomize a subset of the vector with the uniform distribution
    void rand(int thread_id, double low, double high);

    // randomize a subset of the vector with the normal distribution
    void rand_norm(int thread_id, double mean, double stddev);
        
    // constant1 * (this) + constant2 * v2 => sum
    void add(int thread_id, double constant1, double constant2, ColVector* v2,
        ColVector* sum);

    // (this) * constant => result
    void times_constant(int thread_id, double constant, ColVector* result);

    // (this) dot product v2 ==> dot_product
    void dot_product(int thread_id, ColVector* v2, double* dot_product);

    // (this) = vectors[0] + vectors[1] + ...
    void add_merge(int thread_id, ColVector** vectors, int num_vectors);
};



class SparseMatrix
{
    int rows = 0;
    int cols = 0;
    int bin_bit_shift = 0;

    int* row_indices = nullptr; // row_indices[5] = 100 means first value on row #5 is at values[100]
    int* col_indices = nullptr; // col_indices[5] = 90 means values[5] is on column #90
    double* values = nullptr;

    int* spread = nullptr;
    int* sorted_rows = nullptr;

public:
    ~SparseMatrix()
    {
        if (row_indices != nullptr) delete[] row_indices;
        if (col_indices != nullptr) delete[] col_indices;
        if (values != nullptr) delete[] values;
        if (spread != nullptr) delete[] spread;
        if (sorted_rows != nullptr) delete[] sorted_rows;
    }

    void clear();
    void load(int rows, int cols, int* row_indices, int* col_indices, double* values);

    int num_rows() { return rows; }
    int num_cols() { return cols; }

    int* get_row_indices_ptr() { return row_indices; }
    int* get_col_indices_ptr() { return col_indices; }
    double* get_values_ptr() { return values; }

    // (this) * x => result
    void multiply(int thread_id, ColVector* x, ColVector* result,
        bool use_sorted_rows);

    // (this^T) * x => result
    void transpose_multiply(int thread_id, ColVector* x, ColVector* result,
        bool use_sorted_rows);

    // group rows into bins and report the count
    void count_rows(int thread_id, int* bin_counts, int bin_bit_shift);
 
    // initialize "sorted_rows" based on "bin_counts"
    void compute_sorted_rows(int* bin_counts, int bin_bit_shift);

    // count the number of elements in each column
    void count_columns(int thread_id, int* column_counts);

    // Copy data to new "values" and "indices", using a column major format.
    // This is the final step in the transpose process.
    void copy_to_column_major_form(int thread_id, int* new_col_indices,
        int* new_row_indices, double* new_values);
};





// call join on all threads and delete threads when finished
void wait_and_delete_threads(thread** threads);

// randomize the content of "v" using uniform distribution
void vect_rand(ColVector* v, double low, double high);

// randomize the content of "v" using normal distribution
void vect_rand_norm(ColVector* v, double mean, double stddev);

// constant1 * v1 + constant2 * v2 => sum
void vect_add(double constant1, ColVector* v1, double constant2, ColVector* v2,
    ColVector* sum);

// v * constant => result
void vect_times_constant(ColVector* v, double constant, ColVector* result);

// v1 dot product v2
double vect_dot_product(ColVector* v1, ColVector* v2);

// m * x => result
void sparse_matrix_multiply(SparseMatrix* m, ColVector* x, ColVector* result, 
    bool use_sorted_rows = false);

// m^T * x => result
void sparse_matrix_transpose_multiply(SparseMatrix* m, ColVector* x, ColVector* result,
    bool use_sorted_rows = false);

// initialize m->sorted_rows
void sparse_matrix_compute_sorted_rows(SparseMatrix* m, int bin_bit_shift);

/* Solve Ax = b in the least squares sense, using conjugate gradient.
Caller needs to initialize "x". This is the low memory version that uses
transpose multiplication to evaluate A^T * vector. Returns number of 
iterations used. */
int cg_least_squares(SparseMatrix* A, ColVector* b, ColVector* x,
    double min_r_decrease = 0.01, int max_iteration = 200,
    double* final_rr = nullptr);

/* Solve Ax = b in the least squares sense, using conjugate gradient.
Caller needs to initialize "x". This version computes the matrix A^T,
so it uses more memory, but can be faster if there are sufficient number
of cores. Returns number of iterations used. */
int cg_least_squares2(SparseMatrix* A, ColVector* b, ColVector* x,
    double min_r_decrease = 0.01, int max_iteration = 200,
    double* final_rr = nullptr);

// m^T => mt
void sparse_matrix_transpose(SparseMatrix* m, SparseMatrix* mt);

/* sum the arrays in arr and store the result in sum, applying a 
column offset. The "start_col" and "end_col" are inclusive.*/
void add_arrays(int start_col, int end_col, int** arr, int* sum, int num_rows, 
    int col_offset);

/* compute "col_indices", part of the sparse matrix transpose algorithm.
The "start_col" and "end_col" are inclusive.*/
void compute_col_indices(int start_col, int end_col, int* col_indices_base, 
    int** col_indices, int num_rows);

/*Derive "user_factors" and "item_factors" from a set of
(user_ids, item_ids, ratings) using the ALS procedure.
Returns number of iterations used.*/
int als(int* user_ids, int* item_ids, ColVector* ratings,
    int num_item_factors, ColVector* user_factors, ColVector* item_factors,
    double min_r_decrease = 0.01, int max_iteration = 200, int algorithm = 1);

/*Fills the "user_A" matrix, which is used as: (user_A)(user_factors) = (ratings). 
The "start" and "end" are inclusive. */
void fill_user_A(int row_start, int row_end, SparseMatrix* user_A, 
    int* user_ids, int* item_ids, ColVector* item_factors, int num_item_factors, 
    bool first_fill = false);

/*Fills the "item_A" matrix, which is used as: (item_A)(item_factors) = (ratings_minus_bias).
The "start" and "end" are inclusive.*/
void fill_item_A(int row_start, int row_end, SparseMatrix* item_A, 
    int* user_ids, int* item_ids, ColVector* user_factors, int num_item_factors,
    bool first_fill = false);

/*Fills the "ratings_minus_bias" vector, which is the (ratings) vector
minus the (user bias) vector. The "start" and "end" are inclusive.*/
void fill_ratings_minus_bias(int row_start, int row_end, 
    ColVector* ratings_minus_bias, int* user_ids, ColVector* ratings, 
    ColVector* user_factors, int num_item_factors);

// Predict rating using the ALS model provided.
double als_predict(ColVector* user_factors, ColVector* item_factors,
    int user_id, int item_id, int num_item_factors);

#endif

