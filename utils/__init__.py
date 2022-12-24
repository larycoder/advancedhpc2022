# Several useful function
import time
import numpy as np


class ElapsedTime():
    def __init__(self):
        self.t_start = 0.0
        self.t_end = 0.0

    def start(self):
        self.t_start = time.time()

    def end(self):
        self.t_end = time.time()

    def get_elapsed(self):
        return (self.t_end - self.t_start) * 1000

    def print_elapsed(self):
        counter = (self.t_end - self.t_start) * 1000
        print(f"Elapsed time: {counter} ms")


def get_name(path: str):
    return path.split('/')[-1].split('.')[0]


def crop_img(img: np.ndarray, w=None, h=None, base=(0, 0)):
    r_w = img.shape[1]
    r_h = img.shape[0]

    if base != (0, 0):
        print("Could not rebase of crop for now...")
        return img

    if w is None or w > r_w:
        w = r_w
    if h is None or h > r_h:
        h = r_h

    b = base  # name is too long for fitting my screen
    return img[b[0]: b[0] + int(h), b[1]: b[1] + int(w)]
