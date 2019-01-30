import datetime, json, os, pickle, random, socket, time
import config

shared_directory = config.shared_directory



cluster_info = None

# load cluster_info from "cluster_info.json"
file_name = shared_directory + "cluster_info.json"
if os.path.exists(file_name):
    with open(file_name, mode="r") as file:
        cluster_info = json.load(file)
# else:
#     print('The "cluster_info.json" file cannot be found.',
#           "Cannot join cluster.")



def send_command(command: dict, wait_for_reply = False):
    """Sends a "command" to the cluster server. Returns
     the response object if "wait_for_reply" is true."""
    address = cluster_info[0]["address"]
    port_number = cluster_info[0]["port_number"]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((address, port_number))
        sock.sendall(pickle.dumps(command))

        if wait_for_reply:
            sock.settimeout(1)
            reply = pickle.loads(sock.recv(4096))
            return reply

        else:
            return None


def send_command_to_address(address, port_number, command: dict,
                            wait_for_reply = False):
    """Sends a "command" to a general address. Returns
     the response object if "wait_for_reply" is true."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((address, port_number))
        sock.sendall(pickle.dumps(command))

        if wait_for_reply:
            sock.settimeout(1)
            reply = pickle.loads(sock.recv(4096))
            return reply

        else:
            return None


def shutdown():
    """Shutdown the cluster."""
    send_command({"op": "shutdown"})


def check_for_echo(address, port_number):
    """Send an echo command to the (address, port_number)
    node. Return True if correct the echo is received."""
    number = random.randint(0, 10000)

    try:
        reply = send_command_to_address(address, port_number, {
            "op": "echo",
            "value": number
        }, wait_for_reply=True)

        if int(reply["value"]) == number:
            return True
        else:
            return False

    except:
        return False


def print_status():
    status = send_command({"op": "status"}, wait_for_reply=True)

    print("current_op =", status["current_op"])

    # worker information header
    print(" ".ljust(25),
          "state".center(15),
          "total_units / total_time".center(30))

    keys = list(status["workers"].keys())
    keys.sort()

    for key in keys:
        worker = status["workers"][key]
        total_units = worker["total_units"]
        total_time = int(worker["total_time"])
        speed = 0
        if total_time > 0:
            speed = total_units / total_time

        speed_str = str(total_units) + " / " + str(total_time)
        speed_str += " = {0:.4g}".format(speed)

        print(key.ljust(25),
              worker["state"].center(15),
              speed_str.center(30))


def print_progress():
    progress = send_command({"op": "progress"}, wait_for_reply=True)
    seconds_left = progress["seconds_left"]

    if seconds_left == 0:
        print("Operation completed.")
    else:
        length = progress["length"]
        completed = progress["completed"]
        print(completed, "/", length,
              "("+ "{0:.4g}".format(completed / length * 100) + "%)",
              "items completed.")

        seconds_left = int(seconds_left)
        print("Estimated time left:",
              str(datetime.timedelta(seconds=seconds_left)))


def get_progress():
    """Return a data structure with progress information."""
    return send_command({"op": "progress"}, wait_for_reply=True)


def wait_for_completion():
    """Print out the estimated time remaining while waiting
    for the cluster to finish the current distributed
    operation."""
    seconds_left = None

    while seconds_left != 0:
        time.sleep(0.5)
        seconds_left = get_progress()["seconds_left"]

        if seconds_left is not None:
            seconds_left = int(seconds_left)
            s = "Time left: " + str(datetime.timedelta(seconds=seconds_left))
            s = s.ljust(50)
            print("\r" + s, end=" ")

    print()

    time.sleep(1)  # delay for output file to be flushed to disk (ideally not needed)


def merge_list_results(output_file_name : str):
    """Merge all "output.bin.xxx" files in "temp_out_dir" into
    a single list called "output_file_name" at "out_dir".
    ::
        Returns the merged data as a list.
    """
    temp_out_dir = config.temp_out_dir
    out_dir = config.out_dir

    # find a list of all "output.bin.xxx" files
    file_names = os.listdir(temp_out_dir)
    output_files = [] # each element is (sequence number, file name)

    for file_name in file_names:
        if file_name.startswith("output.bin."):
            # try to extract sequence number
            try:
                index = len("output.bin.")
                sequence_str =  file_name[index:]
                sequence_num = int(sequence_str)
                output_files.append((sequence_num, file_name))
            except:
                pass

    if len(output_files) == 0:
        print("No output file found.")
        return

    output_files.sort()

    # merge output_files into a single "results" list
    results = []
    for _, file_name in output_files:
        with open(temp_out_dir + file_name, mode="rb") as file:
            result = list(pickle.load(file))
            results += result

    # save "results" to disk as "output_file_name"
    with open(out_dir + output_file_name, mode="wb") as file:
        pickle.dump(results, file)

    # remove the partial output files
    for _, file_name in output_files:
        os.remove(temp_out_dir + file_name)

    return results


def merge_list_results_into_dict(output_file_name : str):
    """Merge all "output.bin.xxx" files in "temp_out_dir" into
    a single dictionary called "output_file_name" at "out_dir".
    ::
        Returns the merged data as a dictionary.
    """
    temp_out_dir = config.temp_out_dir
    out_dir = config.out_dir

    # find a list of all "output.bin.xxx" files
    file_names = os.listdir(temp_out_dir)
    output_files = [] # each element is (sequence number, file name)

    for file_name in file_names:
        if file_name.startswith("output.bin."):
            # try to extract sequence number
            try:
                index = len("output.bin.")
                sequence_str =  file_name[index:]
                sequence_num = int(sequence_str)
                output_files.append((sequence_num, file_name))
            except:
                pass

    if len(output_files) == 0:
        print("No output file found.")
        return

    output_files.sort()

    # merge output_files into a single "results" dictionary
    results = {}
    for _, file_name in output_files:
        with open(temp_out_dir + file_name, mode="rb") as file:
            result_list = list(pickle.load(file)) # [(movie_id, [similar_movie_ids])]

            for movie_id, similar_movie_ids in result_list:
                results[movie_id] = similar_movie_ids

    # save "results" to disk as "output_file_name"
    with open(out_dir + output_file_name, mode="wb") as file:
        pickle.dump(results, file)

    # remove the partial output files
    for _, file_name in output_files:
        os.remove(temp_out_dir + file_name)

    return results


def run_echo_test(address : str, port : int, num_tests = 30000):
    """Sends echo commands to (address, port) and check their return."""
    success = 0
    failures = 0
    try:
        for i in range(0, num_tests):
            value = random.randint(0, 10000)
            reply = send_command_to_address(address, port, {
                "op": "echo",
                "value": value
            }, wait_for_reply=True)

            if reply["value"] == value:
                success += 1
            else:
                print("Echo test received the wrong echo value.")
                failures += 1

            # On Windows, a slight delay between connection is needed
            # (tested with Python 3.5.4)
            # time.sleep(0.01)
            # without this slight delay, I get:
            # [WinError 10048] Only one usage of each socket address is normally permitted

            if i % 100 == 0: print(".", end="")
            if i >= 5000 and i % 5000 == 0: print()

        print()

    except Exception as ex:
        print("Echo test exception:", ex)
        failures += 1

    print("Test result:", success, "successes and", failures, "failure.")
