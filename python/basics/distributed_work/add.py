import numpy, pickle
import cluster

shared_directory = cluster.shared_directory


def _setup_problem(length):
    """ Creates two arrays on disk, as "input.bin".

    :param length: length of each array
    :return: x1, x2 - the arrays created
    """
    x1 = numpy.random.uniform(-1, 1, length)
    x2 = numpy.random.uniform(-1, 1, length)

    with open(shared_directory + "input.bin", mode="wb") as file:
        pickle.dump(x1, file)
        pickle.dump(x2, file)

    return x1, x2


def _check_output_bin():
    """Check the data in "output.bin" for correctness. Return
    True for correct."""

    with open(shared_directory + "input.bin", mode="rb") as file:
        x1 = pickle.load(file)
        x2 = pickle.load(file)
        real_output = x1 + x2

    with open(shared_directory + "output.bin", mode="rb") as file:
        output = pickle.load(file)

    max_error = max(numpy.abs(real_output - output))

    if max_error < 1e-6:
        print('The content of "output.bin" is correct.')
        return True
    else:
        print('The content of "output.bin" is NOT correct.')
        return False




def main():
    print("Creating input.bin")
    x1, x2 = _setup_problem(10000)

    # Trigger an add operation by sending a distributed
    # add command to cluster server.
    if cluster.cluster_info is None:
        print("Cannot connect to cluster.")
        return

    cluster.send_command({
        "op": "distribute",
        "worker_op": "add",
        "length": len(x1)
    })

    cluster.wait_for_completion()
    cluster.print_status()
    cluster.merge_list_results("output.bin")

    _check_output_bin()



if __name__ == "__main__":
    main()

