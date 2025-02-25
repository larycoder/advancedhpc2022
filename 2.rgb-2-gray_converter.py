import matplotlib.pyplot as plt
import threading
import numpy as np
from numba import cuda
from utils import *

debug_activate = True


def debug(id, obj):
    """ Control debug flow by id """

    if not debug_activate:
        return
    if id == 0:
        return
    if id == 2:
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

    def process(src, dest, idx, len, max):
        for i in range(idx, idx + len):
            if (i >= max):  # prevent index out of range
                break

            pos = i * 3
            value = (int(src[pos]) + int(src[pos + 1]) + int(src[pos + 2])) / 3
            dest[i] = int(value)

    length = int(flatten.shape[0] / 3)
    chunk = int(length / 6)

    new_img = [0 for i in range(length)]
    idx = 0
    thread_pool = []

    # build thread
    while (idx < length):
        arg = (flatten, new_img, idx, chunk, length,)
        p = threading.Thread(target=process, args=arg)
        debug(0, f"thread info: {idx}, {chunk}, {length}")
        thread_pool.append(p)
        idx += chunk

    # run and wait thread
    for p in thread_pool:
        p.start()
        debug(0, f"start thread: {p}")
    for p in thread_pool:
        p.join()
        debug(0, f"end thread: {p}")

    new_img = np.array(new_img)
    debug(0, f"new image shape: {new_img.shape}")
    debug(0, f"new image sample value: {new_img[1:10]}")

    gray_img = np.reshape(new_img, (img.shape[0], img.shape[1]))
    debug(0, f"gray image shape: {img.shape}")

    return gray_img


def gpu_rgb_2_gray(img, thread_per_block=64):
    """
    1. Build kernel
    2. Feed data to device
    3. Execute kernel
    4. Get result
    """
    debug(2, f"image type: {type(img)}")
    debug(2, f"image sample value: \n{img[0][0:5]}")

    flatten = np.reshape(img, -1)
    debug(2, f"old image shape: {img.shape}")
    debug(2, f"flatten image shape: {flatten.shape}")
    debug(2, f"flatten sample value: {flatten[0:10]}")

    @cuda.jit
    def kernel_convert(src, dst, size):
        tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if (tidx >= size):
            return
        sum = src[tidx * 3] + src[tidx * 3 + 1] + src[tidx * 3 + 2]
        dst[tidx] = np.uint8(sum / 3)

    dev_input = cuda.to_device(flatten)
    dev_output = cuda.device_array(
        (int(flatten.shape[0] / 3),), dtype=np.uint8)

    pixel = int(flatten.shape[0] / 3)
    block_per_grid = (pixel + thread_per_block - 1) // thread_per_block

    debug(2, "start execute thread in gpu...")
    kernel_convert[block_per_grid, thread_per_block](
        dev_input, dev_output, int(flatten.shape[0] / 3))
    debug(2, "done execute thread in gpu !!!")

    new_img = dev_output.copy_to_host()
    gray_img = np.reshape(new_img, (img.shape[0], img.shape[1]))
    debug(2, f"gray image shape: {img.shape}")

    return gray_img


def save_img(saved_path, img):
    saved_path = f"result/{saved_path}"
    plt.imsave(saved_path, img, cmap="gray")
    debug(1, f"saved gray image path: " + saved_path)


if __name__ == "__main__":
    timer = ElapsedTime()

    image_file = "resource/eiffel.jpg"
    o_image = plt.imread(image_file)
    image = crop_img(o_image, o_image.shape[1] / 2)

    # initialize GPU
    debug_activate = False
    gpu_rgb_2_gray(image)
    debug_activate = True

    gpu_thread_pool = {}
    img = None

    print("GPU Running...")
    for i in range(1, 10):
        thread_per_block = 16 * i
        timer.start()
        img = gpu_rgb_2_gray(image, thread_per_block)
        timer.end()
        gpu_thread_pool[thread_per_block] = f"{timer.get_elapsed()} ms"
    print("Elapsed Pool:")
    for i in gpu_thread_pool:
        print(f"[{i}] time: {gpu_thread_pool[i]}")
    save_img(f"{get_name(image_file)}_gpu_gray.jpg", img)

    print("\nCPU Running...")
    timer.start()
    img = cpu_rgb_2_gray(image)
    timer.end()
    timer.print_elapsed()
    save_img(f"{get_name(image_file)}_cpu_gray.jpg", img)
