import time

def read_mem_info():
    """Returns mem_total, mem_free in GB"""
    with open("/proc/meminfo", mode="r") as file:
        mem_total = int(file.readline().split()[-2]) / (1024 * 1024.0)
        mem_free = int(file.readline().split()[-2]) / (1024 * 1024.0)

    return mem_total, mem_free


def main():
    print("This script tracks peak memory usage by reading",
          'the first two lines of "/proc/meminfo".')
    max_mem = 0

    while True:
        mem_total, mem_free = read_mem_info()
        mem_used = mem_total - mem_free

        if mem_used > max_mem:
            max_mem = mem_used
            print("Peak memory use:", max_mem, "GB")

        time.sleep(1)


if __name__ == "__main__":
    main()

    