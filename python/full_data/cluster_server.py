import json, os, pickle, random, socket, time
import config

port_number = 10000
batch_time = 20 # number of seconds per batch job


class ClusterServer:
    def __init__(self):
        self.shared_directory = config.shared_directory
        self.address = socket.gethostbyname(socket.gethostname())
        self.port_number = port_number
        self.current_op = DistributedOp()

        self.worker_nodes = {} # indexed by "address:port" string

        self.request_handlers = {
            "distribute": self.handle_distribute,
            "done": self.handle_done,
            "echo": self.handle_echo,
            "progress": self.handle_progress,
            "register": self.handle_register,
            "status": self.handle_status
        }

        self.register_with_disk()


    def register_with_disk(self):
        """Create the "cluster_info.json" file. Save address and
        port number of this node to the file."""
        if os.path.exists(self.shared_directory) == False:
            os.mkdir(self.shared_directory)

        file_name = self.shared_directory + "cluster_info.json"

        cluster_info = [{
            "address": self.address,
            "port_number": port_number
        }]

        print("Cluster server started at ", self.address, "port", port_number)

        with open(file_name, mode="w") as file:
            json.dump(cluster_info, file, indent=4)


    def message_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", port_number))
            sock.listen()

            while True:
                conn, addr = sock.accept()
                b_array = conn.recv(4096)
                request = pickle.loads(b_array)

                op = request["op"]
                if op == "shutdown":
                    self.handle_shutdown()
                    return

                elif op in self.request_handlers:
                    self.request_handlers[op](request, conn)

                else:
                    print("Unhandled operation:", op)
                    print("Request =", request)


    def send_command(self, worker_address_port: str, command: dict,
                     wait_for_reply=False):
        """Sends a "command" to the worker identified by its
        "address:port" string. Returns the response object
        if "wait_for_reply" is true."""
        worker = self.worker_nodes[worker_address_port]
        address = worker.address
        port_number = worker.port_number

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((address, port_number))
            sock.sendall(pickle.dumps(command))

            if wait_for_reply:
                sock.settimeout(1)
                reply = pickle.loads(sock.recv(4096))
                return reply

            else:
                return None


    def handle_shutdown(self):
        """Message handler for "op"="shutdown"."""
        # Delete "cluster_info.json" and tell all worker
        # nodes to shutdown.
        file_name = self.shared_directory + "cluster_info.json"
        if os.path.exists(file_name):
            os.remove(file_name)

        for k in self.worker_nodes.keys():
            self.send_command(k, {"op": "shutdown"})


    def handle_register(self, request, conn):
        """Message handler for "op"="register"."""
        # Handles registration of new "worker_server" joining the cluster.
        # Add new worker's information to "cluster_info" and save to disk
        address = request["address"]
        port_number = request["port_number"]
        cpu_count = request["cpu_count"]

        response = {"op": "reply"}
        conn.sendall(pickle.dumps(response))

        # key = address
        # For testing, there's multiple worker
        # nodes per computer, so using just the address for
        # key is not sufficient.

        key = address + ":" + str(port_number)
        worker_nodes = self.worker_nodes
        worker_nodes[key] = WorkerNode(address, port_number, cpu_count)

        # save information to "cluster_info.json"
        file_name = self.shared_directory + "cluster_info.json"

        cluster_info = [{
            "address": self.address,
            "port_number": self.port_number
        }]

        for k in worker_nodes.keys():
            cluster_info.append({
                "address": worker_nodes[k].address,
                "port_number": worker_nodes[k].port_number,
                "cpu_count": worker_nodes[k].cpu_count
            })

        with open(file_name, mode="w") as file:
            json.dump(cluster_info, file, indent=4)

        print("New worker node", address, "port number", port_number,
              "joined with", cpu_count, "CPUs.")

        # send work to the new node if there is an active operation
        current_op = self.current_op

        if current_op.has_more_work():
            self.send_command(key, {
                "op": "run_op",
                "op_name": current_op.op + "_setup",
                "setup_param": current_op.setup_param
            })
            self.worker_nodes[key].state = "setting up"


    def handle_distribute(self, request, conn):
        """Message handler for "op"="distribute"."""
        current_op = self.current_op

        # check that there is no distributed operation ongoing
        if current_op.has_completed():
            current_op.__init__()
            current_op.op = request["worker_op"]
            current_op.index = 0
            current_op.length = request["length"]

            # send a "_setup" request to all worker nodes
            current_op.setup_param = None  # optional argument "setup_param"
            if "setup_param" in request:
                current_op.setup_param = request["setup_param"]

            for k in self.worker_nodes:
                self.worker_nodes[k].reset()
                self.send_command(k, {
                    "op": "run_op",
                    "op_name": current_op.op + "_setup",
                    "setup_param": current_op.setup_param
                })
                self.worker_nodes[k].state = "setting up"

        else:
            print("Cluster busy. Operation", request["worker_op"], "ignored.")
            return


    def handle_done(self, request, conn):
        """Message handler for "op"="done"."""
        current_op = self.current_op
        worker_id = request["address"] + ":" + str(request["port_number"])
        worker = self.worker_nodes[worker_id]

        if worker.state == "setting up":
            if current_op.has_more_work():
                # first time working
                worker.state = "working"

                # "units" need to default to non-full load, to check that
                # the worker_xxx.py code works with non-full load
                units = worker.cpu_count - 1

                if units < 1: units = 1
                self.send_work(worker_id, units)

                # alternatively:
                # self.send_work(worker_id, worker.cpu_count)
                # The -1 makes sure not all processes are loaded in the
                # beginning, so to test that special, infrequent case.
                return
            else:
                worker.state = "cleaning up"
                self.send_command(worker_id, {
                    "op": "run_op", "op_name": current_op.op + "_cleanup"})
                return

        elif worker.state == "working":
            # track the amount of work the worker has done
            worker.archive_timing_stats(time.time())
            current_op.completed += worker.length

            if current_op.has_more_work():
                # estimate the amount of work that can be done, then send_work(...)
                if worker.total_time > 0.1:
                    # standard formula
                    units = int(worker.total_units / worker.total_time * batch_time)
                else:
                    # special formula - worker.total_time too small
                    units = worker.total_units * 4

                if units < 1: units = 1

                self.send_work(worker_id, units)
                return

            else:
                worker.state = "cleaning up"
                self.send_command(worker_id, {
                    "op": "run_op", "op_name": current_op.op + "_cleanup"})
                return

        elif worker.state == "cleaning up":
            worker.state = "idle"
            if self.workers_all_idle():
                print("The", current_op.op, "operation has been completed.")
                current_op.op = None


    def workers_all_idle(self):
        """Return True if all worker nodes are idle."""
        workers = self.worker_nodes

        for k in workers:
            if workers[k].state != "idle": return False

        return True


    def worker_health_check(self, worker_address_port):
        """Check the node at "worker_address_port" is able
        to handle more work. Return True for success."""
        # For AWS spot instances, check the URL
        # for local testing, send an echo command
        try:
            number = random.randint(0, 10000)

            reply = self.send_command(worker_address_port, {
                "op": "echo",
                "value": number
            }, wait_for_reply=True)

            if int(reply["value"]) == number: return True
            else: return False

        except:
            return False


    def send_work(self, worker_address_port, units):
        """Send "units" of work to the worker at
        "worker_address_port"."""

        healthy = self.worker_health_check(worker_address_port)
        if not healthy:
            del self.worker_nodes[worker_address_port]
            return

        current_op = self.current_op
        worker = self.worker_nodes[worker_address_port]

        worker.index = current_op.index
        units = current_op.get_work(units)
        worker.length = units
        worker.start_time = time.time()

        print("Asking", worker_address_port, "to work on index",
              worker.index, "~", worker.index + units - 1,
              "length", worker.length)

        self.send_command(worker_address_port, {
            "op": "run_op",
            "op_name": current_op.op,
            "start": worker.index,
            "length": worker.length
        })


    def handle_echo(self, request, conn):
        """Message handler for "op"="echo"."""
        value = request["value"]
        response = {
            "op": "reply",
            "value": value
        }
        conn.sendall(pickle.dumps(response))


    def handle_status(self, request, conn):
        """Message handler for "op"="status"."""
        # current op
        response = {"current_op": self.current_op.op,
                    "workers": {}}

        # worker node information
        for key in self.worker_nodes:
            worker = self.worker_nodes[key]
            response["workers"][key] = {
                "state": worker.state,
                "total_time": worker.total_time,
                "total_units": worker.total_units
            }

        conn.sendall(pickle.dumps(response))


    def handle_progress(self, request, conn):
        """Message handler for "op"="progress"."""
        response = {
            "length": self.current_op.length,
            "completed": self.current_op.completed,
            "seconds_left": self.current_op.estimate_seconds_left(self.worker_nodes)
        }

        conn.sendall(pickle.dumps(response))


class WorkerNode:

    def __init__(self, address, port_number, cpu_count):
        self.address = address
        self.port_number = port_number
        self.cpu_count = cpu_count

        self.state = "idle"

        # current responsibility indicator
        self.index = -1
        self.length = 0

        # to estimate how long it takes to process data
        self.start_time = None
        self.total_time = 0
        self.total_units = 0


    def reset(self):
        self.state = "idle"
        self.index = -1
        self.length = 0
        self.start_time = None
        self.total_time = 0
        self.total_units = 0


    def archive_timing_stats(self, stop_time):
        """Accumulate the current timing results (length,
        start_time, stop_time) as (total_time, total_units)."""
        self.total_time += stop_time - self.start_time
        self.total_units += self.length



class DistributedOp:

    def __init__(self):
        self.op = None
        self.length = 0
        self.index = 0
        self.completed = 0
        self.setup_param = None


    def get_work(self, units : int):
        """Request up to "units" amount of work. Increase
        "index" by "units" if possible. Returns the number
        of "units" allocated."""
        if self.index + units <= self.length:
            self.index += units
            return units

        else:
            units = self.length - self.index
            self.index += units
            return units


    def has_more_work(self):
        """Returns True if there is more work to be farmed out."""
        if self.op is None: return False

        if self.index < self.length: return True
        else: return False


    def has_completed(self):
        """Returns True if the operation itself has completed.
        This means all worker nodes have finished their cleanup()."""
        if self.op is None: return True
        else: return False


    def estimate_seconds_left(self, worker_nodes):
        """
        Estimate number of seconds it takes to finish
        the current distributed op.

        :param worker_nodes: dictionary of WorkerNode objects.
        :return: Number of seconds remaining. As long as the
         operation has not been completed, the number of seconds
         remaining is always least 1. Return "None" if no estimation
         is possible due to lack of data.
        """
        # special cases
        if self.op is None: return 0 # op already done
        if len(worker_nodes) == 0: return None # no worker

        if self.index == self.length:
            # the current batch is the last batch
            # compute the end_time for each worker
            end_times = []

            for k in worker_nodes:
                worker = worker_nodes[k]
                if worker.start_time is not None and worker.total_units > 0:
                    end_time = worker.start_time + worker.length * worker.total_time / worker.total_units
                    end_times.append(end_time)

            if len(end_times) == 0: return None

            # estimate using the highest end_time
            end_times.sort()
            time_left = end_times[-1] - time.time()
            if time_left < 1: time_left = 1

            return time_left

        else:
            # general case - current batch is not the last

            # collect information from each worker node and then sort by start_time
            worker_info = [] # each element is (start time, units per batch)

            for k in worker_nodes:
                worker = worker_nodes[k]

                if worker.total_units > 0 and worker.start_time is not None:
                    u = worker.total_units / worker.total_time * batch_time
                    worker_info.append((worker.start_time, u))

            if len(worker_info) == 0: return None

            worker_info.sort()

            # compute the total amount of work (from all workers) per batch
            units_per_batch = 0
            for info in worker_info:
                units_per_batch += info[1]

            # compute number of batches to go
            num_batches = int((self.length - self.index) / units_per_batch)

            # compute size of the final batch
            final_batch = (self.length - self.index) % units_per_batch

            # determine the last worker on the final batch
            index = 0
            for index in range(0, len(worker_info)):
                if final_batch > worker_info[index][1]:
                    final_batch -= worker_info[index][1]
                else:
                    break

            # compute the two possible ending times:
            # t1 = worker at "index" does "final_batch" amount of work on the last batch
            # t2 = worker at "index - 1" spends "batch_time" on the last batch
            t1 = worker_info[index][0] + (num_batches + 1) * batch_time
            t1 += final_batch / worker_info[index][1] * batch_time

            # for t2 computation, there are two cases
            if index > 0:
                t2 = worker_info[index - 1][0] + (num_batches + 2) * batch_time
            else:
                t2 = worker_info[-1][0] + (num_batches + 1) * batch_time

            # time_left = the later of the two possible ending times
            if t1 > t2:
                time_left = t1 - time.time()
            else:
                time_left = t2 - time.time()

            if time_left < 1: time_left = 1
            return time_left



def main():
    cluster_server = ClusterServer()
    cluster_server.message_loop()


if __name__ == "__main__":
    main()
else:
    print("import as module")

