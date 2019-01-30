from math import fabs
from random import random

import add_proc




def main():
    add_proc.start_processes()


    # create two lists of 500 numbers
    x1 = []
    x2 = []
    for i in range(0, 500):
        x1.append(random())
        x2.append(random())

    # compute: x1 * 5 + x2

    add_proc.split_list_and_send(x1, "x1")
    add_proc.split_list_and_send(x2, "x2")

    add_proc.run_function("_multiply", {
        "list_name": "x1",
        "factor" : 5
    })

    add_proc.run_function("_add", {
        "list1_name": "x1",
        "list2_name": "x2",
        "result_name": "sum_list"
    })

    result = add_proc.concat_var_into_list("sum_list")

    add_proc.clear_all_data()

    # compute: x1 * 5 + x2 manually and check
    result2 = []
    for i in range(0, len(x1)):
        x1[i] *= 5
        result2.append(x1[i] + x2[i])

    test_pass = True

    if len(result) != len(result2): test_pass = False

    for i in range(0, len(result)):
        if fabs(result[i] - result2[i]) > 1e-6:
            test_pass = False
            break

    if test_pass:
        print("add_proc computed the result correctly.")
    else:
        print("add_proc is incorrect.")

    add_proc.end_processes()


if __name__ == '__main__':
    main()
