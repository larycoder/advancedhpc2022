import matplotlib.pyplot as plt
import time
import numpy as np


def debug(id, obj):
    """ Control debug flow by id """
    if id == 0:
        return

    print(f"[DEBUG] {obj}")
    pass


def cpu_rgb_2_gray(image_file):
    full_img = plt.imread(image_file)

    img = full_img[0:100]
    debug(0, f"image type: {type(img)}")
    debug(0, f"image sample value: \n{img[0][0:5]}")

    flatten = np.reshape(img, -1)
    debug(0, f"old image shape: {img.shape}")
    debug(0, f"flatten image shape: {flatten.shape}")
    debug(0, f"flatten sample value: {flatten[0:10]}")

    new_img = []
    for i in range(int(flatten.shape[0] / 3)):
        idx = i * 3
        new_img.append(
            (flatten[idx] + flatten[idx + 1] + flatten[idx + 2]) / 3)
    new_img = np.array(new_img)
    debug(0, f"gray image shape: {new_img.shape}")
    debug(0, f"gray image sample value: {new_img[1:10]}")


if __name__ == "__main__":
    image_file = "./resource/eiffel.jpg"

    print("CPU Running...")
    start = time.time()
    cpu_rgb_2_gray(image_file)
    end = time.time()
    print(f"Elapsed time: {(end - start) * 1000} ms")
