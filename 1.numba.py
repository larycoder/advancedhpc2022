from numba import cuda

cc_cores_per_SM_dict = {
    (2, 0): 32,
    (2, 1): 48,
    (3, 0): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,
    (5, 2): 128,
    (6, 0): 64,
    (6, 1): 128,
    (7, 0): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 9): 128,
    (9, 0): 128
}

# device info
print("=== DEVICE DETECT ===")
print(cuda.detect())
print("=========")

print("=== DEVICE INFO ===")
for i in cuda.gpus:
    print(f"Device name: {i.name.decode('UTF-8')} ({i.id})")
    device = cuda.select_device(i.id)

    # core info: multiprocessor count, core count
    multiprocessor_count = device.MULTIPROCESSOR_COUNT
    compute_capability = device.compute_capability
    core_per_sm = cc_cores_per_SM_dict[compute_capability]
    core_count = core_per_sm * multiprocessor_count

    print("Multiprocessor count: ", multiprocessor_count)
    print("Core count: ", core_count)

    # memory info: memory size
    print(device.get_memory_info())
print("=========")
