import multiprocessing, pickle, requests, socket, sys
import cluster
import worker_process as _proc

shared_directory = cluster.shared_directory

port_number = 10001
aws_instance = False

_data = {} # data that needs to survive between op requests


class WorkerServer:

    def __init__(self):
        self.address = socket.gethostbyname(socket.gethostname())

        if self._register_with_cluster():
            self._message_loop()
        else:
            print("Failed to register with cluster.")


    def _register_with_cluster(self):
        """Register with cluster server. Return True for success."""
        try:
            reply = cluster.send_command({
                    "op": "register",
                    "address": self.address,
                    "port_number": port_number,
                    "cpu_count": multiprocessing.cpu_count()
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
    file_name = shared_directory + file_name + "." + str(sequence_number)
    with open(file_name, mode="bw") as file:
        pickle.dump(output, file)


def main():
    # exit if unable to find cluster server
    if cluster.cluster_info is None:
        print("No cluster found, exiting")
        return

    _proc.start_processes()

    WorkerServer() # starts message loop automatically

    _proc.end_processes()


# End of framework
#####################################################################


def add_setup(request):
    """Load data from "input.bin" to x1 and x2."""
    with open(shared_directory + "input.bin", mode="rb") as file:
        _data["x1"] = pickle.load(file)
        _data["x2"] = pickle.load(file)


def add(request):
    """Split up work for the processes, merge result, then
    write output to disk."""
    start = request["start"]
    length = request["length"]

    _proc.split_list_and_send(_data["x1"], start, length, "x1")
    _proc.split_list_and_send(_data["x2"], start, length, "x2")
    _proc.run_function("add", None)

    # merge result
    sum = _proc.concat_var_into_numpy_array("sum")

    # write output to disk
    write_output_to_disk(sum, "output.bin", start)





if __name__ == "__main__":
    # process global command line arguments
    for arg in sys.argv:
        if arg.startswith("port_number="):
            port_number = int(arg.split(sep='=')[1])
        elif arg == "aws_instance":
            aws_instance = True

    main()

