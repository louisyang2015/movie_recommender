// ls_windows_dll.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

// #include <iostream>
#include "matrix.h"

using namespace std;

int thread_count = 4;

extern "C" __declspec(dllexport) void set_thread_count(int thread_count)
{
    ::thread_count = thread_count;
}

extern "C" __declspec(dllexport) int get_thread_count()
{
    return thread_count;
}

/* extern "C" __declspec(dllexport) void print_double_array(double* array, int length)
{
    for (int i = 0; i < length; i++)
        cout << array[i] << " ";

    cout << endl;
    return;
} */

// Wrapper around "cg_least_squares(...)"
extern "C" __declspec(dllexport) int cg_least_squares_from_python(
    // instead of SparseMatrix* A
    int A_rows, int A_cols, int* A_row_indices, int* A_col_indices, 
    double* A_values,

    // instead of ColVector* b
    int b_length, double* b_values,

    // instead of ColVector* x
    int x_length, double* x_values,
        
    double min_r_decrease, int max_iteration, double* final_rr)
{
    // Create A, b, x - with "delete_values" set to "false"
    SparseMatrix A;
    A.load(A_rows, A_cols, A_row_indices, A_col_indices, A_values, false);

    ColVector b(b_values, b_length, false);
    ColVector x(x_values, x_length, false);

    return cg_least_squares(&A, &b, &x, min_r_decrease, max_iteration, 
        final_rr);
}


// Wrapper around "cg_least_squares2(...)"
extern "C" __declspec(dllexport) int cg_least_squares2_from_python(
    // instead of SparseMatrix* A
    int A_rows, int A_cols, int* A_row_indices, int* A_col_indices,
    double* A_values,

    // instead of ColVector* b
    int b_length, double* b_values,

    // instead of ColVector* x
    int x_length, double* x_values,

    double min_r_decrease, int max_iteration,
    double* final_rr)
{
    // Create A, b, x - with "delete_values" set to "false"
    SparseMatrix A;
    A.load(A_rows, A_cols, A_row_indices, A_col_indices, A_values, false);

    ColVector b(b_values, b_length, false);
    ColVector x(x_values, x_length, false);

    return cg_least_squares2(&A, &b, &x, min_r_decrease, max_iteration, 
        final_rr);
}


// Wrapper around "als(...)"
extern "C" __declspec(dllexport) int als_from_python(int* user_ids, int* item_ids,
    // instead of ColVector* ratings,
    int ratings_length, double* ratings_values,

    int num_item_factors, 

    // instead of ColVector* user_factors, 
    int user_factors_length, double* user_factors_values,

    // instead of ColVector* item_factors,
    int item_factors_length, double* item_factors_values,

    double min_r_decrease, int max_iteration, int algorithm)
{
    // Create ratings, user_factors, item_factors - with "delete_values"
    // set to "false"
    ColVector ratings(ratings_values, ratings_length, false);
    ColVector user_factors(user_factors_values, user_factors_length, false);
    ColVector item_factors(item_factors_values, item_factors_length, false);

    return als(user_ids, item_ids, &ratings, num_item_factors, &user_factors, 
        &item_factors, min_r_decrease, max_iteration, algorithm);
}