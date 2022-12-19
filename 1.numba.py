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
print("=========\n")

print("=== DEVICE INFO ===")
for i in cuda.gpus:
    print(f"Device name: {i.name.decode('UTF-8')} ({i.id})")
    device = cuda.select_device(i.id)
    mem_info = cuda.current_context(i.id).get_memory_info()

    # core info: multiprocessor count, core count
    multiprocessor_count = device.MULTIPROCESSOR_COUNT
    compute_capability = device.compute_capability
    core_per_sm = cc_cores_per_SM_dict[compute_capability]
    core_count = core_per_sm * multiprocessor_count

    print("Multiprocessor count: ", multiprocessor_count)
    print("Core count: ", core_count)

    # memory info: memory size
    print(f"Memory total: {round(mem_info.total / (1024 * 1024 * 1024), 2)} Gb")
    print(f"Memory free: {round(mem_info.free / (1024 * 1024 * 1024), 2)} Gb")

    print()
print("=========\n")
