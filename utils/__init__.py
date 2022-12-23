# Several useful function
import time


class ElapsedTime():
    def __init__(self):
        self.t_start = 0.0
        self.t_end = 0.0

    def start(self):
        self.t_start = time.time()

    def end(self):
        self.t_end = time.time()

    def print_elapsed(self):
        counter = (self.t_end - self.t_start) * 1000
        print(f"Elapsed time: {counter} ms")


def get_name(path: str):
    return path.split('/')[-1].split('.')[0]
