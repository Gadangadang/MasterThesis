import numpy as np 
import cupy as cp 
import time 
from Utilities.config import *
from Utilities.pathfile import *
import matplotlib.pyplot as plt 


numpy_time = []
cupy_time = []
test_events = [10**x for x in [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]]


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


        string_time = f"numpy storing and loading large array: {(time1end-time1)/60} minutes"
        numpy_time.append((time1end-time1)/60)

        print(string_time)
    except:
        pass 
    
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
    
    
plt.plot(test_events[:len(numpy_time)], numpy_time, label="NumPy")
plt.plot(test_events[:len(cupy_time)], cupy_time, label="CuPy")
plt.yscale('log')
plt.xscale('log')
plt.legend(prop={"size": 15})
plt.xlabel("Nr of events", fontsize=25)
plt.ylabel("Minutes", fontsize=25)
plt.title("Creation, saving and loading ", fontsize=25)
plt.tight_layout()
plt.savefig(f"/home/sgfrette/MasterThesis/Figures/hardware/cupy_testing.pdf")
plt.close()