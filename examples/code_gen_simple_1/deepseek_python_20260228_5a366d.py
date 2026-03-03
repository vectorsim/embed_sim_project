import time
import numpy as np

# Test Python version
start = time.time()
for _ in range(1000):
    processor_py.compute(0, 0.001, [input_signal])
py_time = time.time() - start

# Test C version
start = time.time()
for _ in range(1000):
    processor.compute(0, 0.001, [input_signal])
c_time = time.time() - start

print(f"Python: {py_time*1000:.3f} ms")
print(f"C:      {c_time*1000:.3f} ms")
print(f"Speedup: {py_time/c_time:.1f}x")