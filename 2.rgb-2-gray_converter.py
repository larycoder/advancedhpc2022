import matplotlib.pyplot as plt
from numba import cuda
import time
import numpy as np


def debug(id, obj):
    """ Control debug flow by id """
    if id == 0:
        return

    print(f"[DEBUG] {obj}")
    pass


def cpu_rgb_2_gray(img):
    debug(0, f"image type: {type(img)}")
    debug(0, f"image sample value: \n{img[0][0:5]}")

    flatten = np.reshape(img, -1)
    debug(0, f"old image shape: {img.shape}")
    debug(0, f"flatten image shape: {flatten.shape}")
    debug(0, f"flatten sample value: {flatten[0:10]}")

    new_img = []
    for i in range(int(flatten.shape[0] / 3)):
        idx = i * 3
        value = (flatten[idx] + flatten[idx + 1] + flatten[idx + 2]) / 3
        new_img.append(int(value))
    new_img = np.array(new_img)
    debug(0, f"new image shape: {new_img.shape}")
    debug(0, f"new image sample value: {new_img[1:10]}")

    gray_img = np.reshape(new_img, (img.shape[0], img.shape[1]))
    debug(0, f"gray image shape: {img.shape}")

    return gray_img


def gpu_rgb_2_gray(img):
    """
    1. Feed data to device
    2. Build kernel
    3. Execute kernel
    4. Get result
    """
    @cuda.jit
    def convert(src, dst):
        tidx = cuda.threadIdx.x
        bidx = cuda.blockIdx.x
        bdidx = cuda.blockDim.x
        #gidx = tidx + bidx * bdidx
        pass


def save_img(saved_path, img):
    saved_path = f"result/{saved_path}"
    plt.imsave(saved_path, img, cmap="gray")
    debug(1, f"saved gray image path: " + saved_path)


if __name__ == "__main__":
    image_file = "resource/eiffel.jpg"
    image = plt.imread(image_file)

    print("CPU Running...")
    start = time.time()
    img = cpu_rgb_2_gray(image[0:100])
    end = time.time()
    print(f"Elapsed time: {(end - start) * 1000} ms")

    img_name = f"{image_file.split('/')[-1].split('.')[0]}_cpu_gray.jpg"
    save_img(img_name, img)
