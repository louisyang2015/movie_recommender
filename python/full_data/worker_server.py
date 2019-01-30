"""
Command line arguments:
    aws_instance - sets the aws_instance flag, causing "handle_echo" to check
        special URL, default is False

    port_number=10001 - sets port number to listen on
        default is 10001

    cpu_count=4 - sets the number of processes to spawn
        default is multiprocessing.cpu_count()
"""


import multiprocessing, pickle, requests, socket, sys
import cluster, config, movie_lens_data, my_util
import worker_process as _proc


port_number = 10001
aws_instance = False


_data = {} # data that needs to survive between op requests


class WorkerServer:

    def __init__(self, cpu_count):
        self.address = socket.gethostbyname(socket.gethostname())

        if self._register_with_cluster(cpu_count):
            self._message_loop()
        else:
            print("Failed to register with cluster.")


    def _register_with_cluster(self, cpu_count):
        """Register with cluster server. Return True for success."""
        try:
            reply = cluster.send_command({
                    "op": "register",
                    "address": self.address,
                    "port_number": port_number,
                    "cpu_count": cpu_count
                }, wait_for_reply=True)

            if reply["op"] == "reply":
                # print cluster information
                cluster_info = cluster.cluster_info
                print("Worker server at", self.address, "port number", port_number)
                print("Joined cluster at", cluster_info[0]["address"],
                      "port number", cluster_info[0]["port_number"])
                return True

            else: return False

        except Exception as ex:
            print("Error:", ex)
            return False


    def _message_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", port_number))
            sock.listen()

            while True:
                conn, _ = sock.accept()
                b_array = conn.recv(4096)
                request = pickle.loads(b_array)
                print("Request received:", request)

                op = request["op"]

                if op == "shutdown":
                    return

                elif op == "echo":
                    self._handle_echo(request, conn)

                elif op == "run_op":
                    op_name = request["op_name"]

                    if op_name.endswith("_cleanup"):
                        # same clean up handler
                        _data.clear()
                        _proc.clear_all_data()
                        self._send_done_reply()

                    elif op_name in globals():
                        globals()[op_name](request)
                        self._send_done_reply()
                    else:
                        print("Unhandled operation:", op)
                        print("op_name:", op_name)
                        # print("Request =", request)

                else:
                    print("Unhandled operation:", op)
                    # print("Request =", request)


    def _handle_echo(self, request, conn):
        value = request["value"]

        # for AWS (spot) instance, check the special url for interruption notice
        if aws_instance:
            r = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action")
            # standard response is error code 404
            if r.status_code != 404:
                # if status_code is not 404, then there is a cancellation notice
                value += 1

        response = {
            "op": "reply",
            "value": value
        }
        conn.sendall(pickle.dumps(response))


    def _send_done_reply(self):
        cluster.send_command({
            "op": "done",
            "address": self.address,
            "port_number": port_number
        })


def write_output_to_disk(output, file_name, sequence_number):
    file_name = config.temp_out_dir + file_name + "." + str(sequence_number)
    with open(file_name, mode="bw") as file:
        pickle.dump(output, file)


def main():
    # process cpu_count from command line
    cpu_count = multiprocessing.cpu_count()

    for arg in sys.argv:
        if arg.startswith("cpu_count="):
            cpu_count = int(arg.split(sep='=')[1])

    # exit if unable to find cluster server
    if cluster.cluster_info is None:
        print("No cluster found, exiting")
        return

    _proc.start_processes(cpu_count)

    WorkerServer(cpu_count)  # starts message loop automatically

    _proc.end_processes()


# End of framework
#####################################################################


def als_eval_setup(request):
    """
    Sets:
        * _data["user_ratings_test"]
        * _process_data["movie_medians_train"]
        * _process_data["num_item_factors"]
        * _process_data["als_user_factors"]
        * _process_data["als_user_ids"]
        * _process_data["als_movie_factors"]
        * _process_data["als_movie_ids"]
    """
    factor = request["setup_param"]
    als_prefix = "als" + str(factor) + "_"

    # data for worker server
    _data["user_ratings_test"] = movie_lens_data.get_als_obj(
            als_prefix + "user_ratings_test")

    # data for worker processors
    movie_medians_train = movie_lens_data.get_input_obj("movie_medians_train")
    als_user_factors = movie_lens_data.get_als_obj(als_prefix + "user_factors")
    als_user_ids = movie_lens_data.get_als_obj(als_prefix + "user_ids")
    als_movie_factors = movie_lens_data.get_als_obj(als_prefix + "item_factors")
    als_movie_ids = movie_lens_data.get_als_obj(als_prefix + "movie_ids")

    _proc.send_same_data({
        "movie_medians_train": movie_medians_train,
        "num_item_factors": factor,
        "als_user_factors": als_user_factors,
        "als_user_ids": als_user_ids,
        "als_movie_factors": als_movie_factors,
        "als_movie_ids": als_movie_ids
    })


def als_eval(request):
    """
    Input:
        * _data["user_ratings_test"]
        * request["start"]
        * request["length"]
    Output:
        * file: output.bin.start
    """
    # Split up work for the processes, combine response, write result to disk.
    start = request["start"]
    length = request["length"]

    user_ratings_test = _data["user_ratings_test"]
    _proc.split_list_and_send(user_ratings_test, start, length, "user_ratings_test")

    _proc.run_function("_als_eval")

    # combine response
    user_agreements = _proc.concat_var_into_list("user_agreements")

    # write result to disk
    write_output_to_disk(user_agreements, "output.bin", start)


def build_similar_movies_db_setup(request):
    """
    Sets:
        * _process_data["buff_point"]
        * _process_data["buff_limit"]
        * _process_data["movie_ratings"]
        * _process_data["movie_genres"]
    """
    # collect the necessary data
    buff_point = request["setup_param"]["buff_point"]
    buff_limit = request["setup_param"]["buff_limit"]

    movie_ratings = movie_lens_data.get_input_obj("movie_ratings")
    movie_genres = movie_lens_data.get_input_obj("movie_genres")

    # send data to worker processors
    _proc.send_same_data({
        "buff_point": buff_point,
        "buff_limit": buff_limit,
        "movie_ratings": movie_ratings,
        "movie_genres": movie_genres
    })


def build_similar_movies_db(request):
    """
    Input:
        * request["start"]
        * request["length"]
    Output:
        * file: output.bin.start
    """
    # Split up work for the processes
    start = request["start"]
    length = request["length"]
    _proc.split_range_and_send(start, length, "movie_ratings")

    _proc.run_function("_build_similar_movies_db")

    # merge results
    similar_movies = _proc.concat_var_into_list("similar_movies")

    # write result to disk
    write_output_to_disk(similar_movies, "output.bin", start)


def median_eval_setup(request):
    """
    Sets:
        * _data["user_ratings_test"]
        * _process_data["movie_medians_train"]
    """
    # data for worker server
    _data["user_ratings_test"] = movie_lens_data.get_input_obj("user_ratings_test")

    # data for worker processors
    movie_medians_train = movie_lens_data.get_input_obj("movie_medians_train")
    _proc.send_same_data({"movie_medians_train": movie_medians_train})


def median_eval(request):
    """
    Input:
        * _data["user_ratings_test"]
        * request["start"]
        * request["length"]
    Output:
        * file: output.bin.start
    """
    # Split up work for the processes, combine response, write result to disk.
    start = request["start"]
    length = request["length"]

    user_ratings_test = _data["user_ratings_test"]
    _proc.split_list_and_send(user_ratings_test, start, length, "user_ratings_test")
    _proc.run_function("_median_eval")

    # combine response
    user_agreements = _proc.concat_var_into_list("user_agreements")

    # write result to disk
    write_output_to_disk(user_agreements, "output.bin", start)


def tag_count_eval_setup(request):
    """
    Sets:
        * _data["user_ratings_train"]
        * _data["user_ratings_test"]
        * _process_data["movie_genres"]
        * _process_data["movie_tags"]
        * _process_data["tag_counts"]
        * _process_data["genre_counts"]
        * _process_data["movie_medians_train"]
    """
    # data for worker server
    _data["user_ratings_train"] = movie_lens_data.get_input_obj("user_ratings_train")
    _data["user_ratings_test"] = movie_lens_data.get_input_obj("user_ratings_test")

    # data for worker processors
    movie_genres = movie_lens_data.get_input_obj("movie_genres")
    movie_tags = movie_lens_data.get_input_obj("movie_tags")
    tag_counts = movie_lens_data.get_input_obj("tag_counts")
    genre_counts = movie_lens_data.get_input_obj("genre_counts")
    movie_medians_train = movie_lens_data.get_input_obj("movie_medians_train")

    _proc.send_same_data({
        "movie_genres": movie_genres,
        "movie_tags": movie_tags,
        "tag_counts": tag_counts,
        "genre_counts": genre_counts,
        "movie_medians_train": movie_medians_train
    })


def tag_count_eval(request):
    """
    Input:
        * _data["user_ratings_train"]
        * _data["user_ratings_test"]
        * request["start"]
        * request["length"]
    Output:
        * file: output.bin.start
    """
    # Split up work for the processes, combine response, write result to disk.
    start = request["start"]
    length = request["length"]

    user_ratings_test = _data["user_ratings_test"]
    user_ratings_train = _data["user_ratings_train"]
    _proc.split_list_and_send(user_ratings_train, start, length, "user_ratings_train")
    _proc.split_list_and_send(user_ratings_test, start, length, "user_ratings_test")

    _proc.run_function("_tag_count_eval")

    # combine response
    user_agreements = _proc.concat_var_into_list("user_agreements")

    # write result to disk
    write_output_to_disk(user_agreements, "output.bin", start)


def tag_ls_eval_setup(request):
    """
    Sets:
        * _data["user_ratings_train"]
        * _data["user_ratings_test"]
        * _process_data["movie_genres"]
        * _process_data["movie_tags"]
        * _process_data["tag_counts"]
        * _process_data["movie_medians_train"]
    """
    # data for worker server
    _data["user_ratings_train"] = movie_lens_data.get_input_obj("user_ratings_train")
    _data["user_ratings_test"] = movie_lens_data.get_input_obj("user_ratings_test")

    # data for worker processors
    movie_genres = movie_lens_data.get_input_obj("movie_genres")
    movie_tags = movie_lens_data.get_input_obj("movie_tags")
    tag_counts = movie_lens_data.get_input_obj("tag_counts")
    movie_medians_train = movie_lens_data.get_input_obj("movie_medians_train")

    _proc.send_same_data({
        "movie_genres": movie_genres,
        "movie_tags": movie_tags,
        "tag_counts": tag_counts,
        "movie_medians_train": movie_medians_train
    })


def tag_ls_eval(request):
    """
    Input:
        * _data["user_ratings_train"]
        * _data["user_ratings_test"]
        * request["start"]
        * request["length"]
    Output:
        * file: output.bin.start
    """
    # Split up work for the processes, combine response, write result to disk.
    start = request["start"]
    length = request["length"]

    user_ratings_test = _data["user_ratings_test"]
    user_ratings_train = _data["user_ratings_train"]
    _proc.split_list_and_send(user_ratings_train, start, length, "user_ratings_train")
    _proc.split_list_and_send(user_ratings_test, start, length, "user_ratings_test")

    _proc.run_function("_tag_ls_eval")

    # combine response
    user_agreements = _proc.concat_var_into_list("user_agreements")

    # write result to disk
    write_output_to_disk(user_agreements, "output.bin", start)




if __name__ == "__main__":
    # process global command line arguments
    for arg in sys.argv:
        if arg.startswith("port_number="):
            port_number = int(arg.split(sep='=')[1])
        elif arg == "aws_instance":
            aws_instance = True

    main()

