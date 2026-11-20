"""One short-lived requester queries CUDA free/total bytes in its own context."""
import argparse
import ctypes
import json
import os
import time

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",default="/usr/local/cuda-13.3/lib64/libcudart.so")
    a=p.parse_args()
    cuda=ctypes.CDLL(a.library)
    free,total=ctypes.c_size_t(),ctypes.c_size_t()
    cuda.cudaMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    status=cuda.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
    print(json.dumps({"unix_ns":time.time_ns(),"pid":os.getpid(),"library":a.library,
                      "code":status,"free_bytes":free.value if not status else None,
                      "total_bytes":total.value if not status else None}))
    raise SystemExit(bool(status))
