import numpy as np 
import cupy as cp 
import time 
from Utilities.config import *
from Utilities.pathfile import *
import matplotlib.pyplot as plt 




mempool = cp.get_default_memory_pool()

with cp.cuda.Device(0):
    mempool.set_limit(size=40*1024**3)  # 30 GiB

numpy_time = []
cupy_time = []
test_events = [10**x for x in [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 6.9, 7, 7.5]]
byte = []

for events in test_events:
    events = int(events)

    print(" ")
    print(f"Nr events: {events}")
    try:
        
        time1 = time.time()

        arr = np.random.uniform(0, 1, size=(events, 529))

        np.save(DATA_PATH / "cpy_test/test_np.npy", arr)
        arr = np.load(DATA_PATH / "cpy_test/test_np.npy")

        time1end = time.time()
        
        print(f"size: {(arr.nbytes)/1000000000} Gbytes")
        string_time = f"numpy storing and loading large array: {(time1end-time1)/60} minutes"
        numpy_time.append((time1end-time1)/60)

        print(string_time)
    except:
        pass 
    
    
    
    byte.append((arr.nbytes)/1000000000)
    
    try:
        time2 = time.time()
        arr = cp.random.uniform(0, 1, size=(events, 529))

        cp.save(DATA_PATH / "cpy_test/test_cp.npy", arr)

        arr = cp.load(DATA_PATH / "cpy_test/test_cp.npy")

        time2end = time.time()
        
        

        cupy_time.append((time2end-time2)/60)
        string_time = f"CuPy storing and loading large array: {(time2end-time2)/60} minutes"

        print(string_time)
    except:
        pass
    

fig, ax1 = plt.subplots()
    
p1 = ax1.plot(test_events[:len(numpy_time)], numpy_time, label="NumPy")
p2 = ax1.plot(test_events[:len(cupy_time)], cupy_time, label="CuPy")


ax1.set_yscale('log')
ax1.set_xscale('log')
ax2 = ax1.twinx()
p3 = ax2.scatter(test_events, byte, label="Arr size", marker="+", color="black")
ax2.set_yscale("log")
ax2.set_ylabel("GBytes", fontsize=25)
ax1.legend(prop={"size": 15}, loc="upper left")
ax2.legend(prop={"size": 15}, loc="lower right")
ax1.set_xlabel("Nr of events", fontsize=25)
ax1.set_ylabel("Minutes", fontsize=25)
plt.title("Creation, saving and loading ", fontsize=25)
fig.tight_layout()
plt.savefig(f"/home/sgfrette/MasterThesis/Figures/hardware/cupy_testing.pdf")
plt.close()