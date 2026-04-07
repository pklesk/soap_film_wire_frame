__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

import numpy as np
import time
from numba import cuda, jit, prange
from numba import void, int8, int32, float32, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_type
from numba.core.errors import NumbaPerformanceWarning
import math
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
import os
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"

DEFAULT_TPB = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
DEFAULT_TPB_SIDE = 16 
DEFAULT_LAZY_STOP_CHECK = 100
DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE = 64
DEFAULT_CORES = 1024
DEFAULT_MC_CUDA_TPB = 128
DEFAULT_MC_CPU_NUMPY_CHUNK_SIZE = 1000 # 1000 (250 for sample plot)
DEFAULT_MC_CUDA_RPT = 10 # repetitions (walks) per thread (owing to this, fewer random generators can be initialized and CUDA blocks scheduled)                                          

def sfwf_contraction_cpu_numpy(heights_in, eps, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CPU NUMPY... [eps: {eps}]")
    t1 = time.time()
    h_full = np.copy(heights_in) 
    h = h_full[1:-1, 1:-1]
    k = 0    
    d = np.inf         
    while True:
        h_t = h_full[0:-2, 1:-1]
        h_b = h_full[2:, 1:-1]
        h_l = h_full[1:-1, 0:-2]
        h_r = h_full[1:-1, 2:]        
        h_new = 0.25 * (h_t + h_b + h_l + h_r)
        d = np.max(np.abs(h_new - h))        
        h[:] = h_new[:] 
        k += 1
        if d <= eps:
            break            
    heights_out = np.copy(heights_in)
    heights_out[1:-1, 1:-1] = h
    t2 = time.time()
    if verbose:
        print(f"SFWF CONTRACTION CPU NUMPY DONE. [d_inf: {str(d)}, iterations: {k}, time: {t2 - t1} s]")
    return heights_out, d, k, t2 - t1

def sfwf_contraction_cpu_numba_parallel(heights_in, eps, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CPU NUMBA PARALLEL... [eps: {eps}]")
    t1 = time.time()
    k = 0    
    d = np.inf
    h_in = np.copy(heights_in)
    h_out = np.copy(heights_in)
    while True:
        sfwf_contraction_cpu_numba_parallel_job(h_in, h_out)
        d = np.max(np.abs(h_out - h_in))
        k += 1        
        if d <= eps:
            break
        h_in, h_out = h_out, h_in        
    t2 = time.time()
    if verbose:
        print(f"SFWF CONTRACTION CPU NUMBA PARALLEL. [d_inf: {str(d)}, iterations: {k}, time: {t2 - t1} s]")
    return h_out, d, k, t2 - t1

@jit(void(float32[:, :], float32[:, :]), nopython=True, parallel=True)
def sfwf_contraction_cpu_numba_parallel_job(h_in, h_out):
    for i in prange(1, h_in.shape[0] - 1):
        for j in prange(1, h_in.shape[1] - 1):
            h_out[i, j] = 0.25 * (h_in[i - 1, j] + h_in[i + 1, j] + h_in[i, j - 1] + h_in[i, j + 1])

def sfwf_contraction_cuda_small(heights_in, eps, tpb, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CUDA SMALL... [eps: {eps}, bpg: 1, tpb: {tpb}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    dev_d = cuda.device_array(1, dtype=np.float32)
    dev_k = cuda.device_array(1, dtype=np.int32)    
    sfwf_contraction_cuda_small_job[1, tpb](dev_h_in, eps, dev_h_out, dev_d, dev_k)
    heights_out = dev_h_out.copy_to_host()   
    d = dev_d.copy_to_host()[0]
    k = dev_k.copy_to_host()[0]
    cuda.synchronize()
    t2 = time.time()
    if verbose:    
        print(f"SFWF CONTRACTION CUDA SMALL DONE. [d_inf: {str(d)}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:, :], float32, float32[:, :], float32[:], int32[:]))    
def sfwf_contraction_cuda_small_job(h_in, eps, h_out, d, k):         
    shared_h = cuda.shared.array((64, 64), dtype=float32) # side corresponds to DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE 
    shared_d = cuda.shared.array((64, 64), dtype=float32) # side corresponds to DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    m, n = h_in.shape
    m_n = int32(m * n)
    it, jt = t // n, t % n
    ept = (m_n + tpb - 1) // tpb    
    k_ = int32(0)
    e = t
    for _ in range(ept):
        if e < m_n:
            i, j = e // n, e % n        
            hij = h_in[i, j]
            shared_h[i, j] = hij
        e += tpb
    cuda.syncthreads()
    while True:
        e = t
        for _ in range(ept):
            if e < m_n:
                i, j = e // n, e % n     
                shared_d[i, j] = shared_h[i, j] # using shared_d temporarily to store old contents of shared_h
            e += tpb
        cuda.syncthreads()
        e = t
        for _ in range(ept):
            if e < m_n:
                i, j = e // n, e % n
                if i > 0 and i < m - 1 and j > 0 and j < n - 1:
                    shared_h[i, j] = float32(0.25) * (shared_d[i - 1, j] + shared_d[i + 1, j] + shared_d[i, j - 1] + shared_d[i, j + 1]) # contraction
            e += tpb
        cuda.syncthreads()    
        e = t
        for _ in range(ept):
            if e < m_n:    
                i, j = e // n, e % n            
                shared_d[i, j] = math.fabs(shared_h[i, j] - shared_d[i, j]) # using shared_d to store absolute differences
            e += tpb
        cuda.syncthreads()  
        for q in range(1, ept): # gathering max point-wise differences within first tpb entries of array (preparation before reduction) 
            e = t + q * tpb
            if e < m_n:                
                ie, je = e // n, e % n                            
                shared_d[it, jt] = max(shared_d[it, jt], shared_d[ie, je])                        
        stride = tpb >> 1  
        cuda.syncthreads()
        while stride > 0: # max-reduction (over 2d shared array)
            if t < stride:
                t_s = t + stride
                if t_s < m_n:             
                    i_s, j_s = t_s // n, t_s % n                
                    shared_d[it, jt] = max(shared_d[it, jt], shared_d[i_s, j_s])
            cuda.syncthreads()
            stride >>= 1
        k_ += 1
        if shared_d[0, 0] <= eps:
            break  
    e = t
    for _ in range(ept):
        if e < m_n:
            i, j = e // n, e % n
            h_out[i, j] = shared_h[i, j]        
        e += tpb        
    if t == 0:
        d[0] = shared_d[0, 0]
        k[0] = k_

def sfwf_contraction_cuda_large_atomicmax(heights_in, eps, lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb_side=DEFAULT_TPB_SIDE, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAX... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    d = np.zeros(1, dtype=np.float32)
    dev_d = cuda.to_device(d)   
    tpb = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side      
    bpg = (bpg_j, bpg_i)
    if verbose:
        print(f"[bpg: {bpg}, tpb: {tpb}]")
    k = 0
    while True:
        sfwf_contraction_cuda_large_atomicmax_reset[1, 1](dev_d)
        sfwf_contraction_cuda_large_atomicmax_job[bpg, tpb](dev_h_in, dev_h_out, dev_d)
        k += 1
        if k % lazy_stop_check == 0:        
            dev_d.copy_to_host(ary=d)
            cuda.synchronize()
            if d[0] <= eps:
                break
        tmp = dev_h_in
        dev_h_in = dev_h_out
        dev_h_out = tmp
    heights_out = dev_h_out.copy_to_host()
    d = d[0]    
    t2 = time.time()
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAX DONE. [d_inf: {str(d)}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:]))    
def sfwf_contraction_cuda_large_atomicmax_reset(d): # called exactly for 1 thread
    d[0] = float32(0.0) 

@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))    
def sfwf_contraction_cuda_large_atomicmax_job(h_in, h_out, d):       
    shared_h = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values 
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2
    j, i = cuda.grid(2)
    tj, ti = cuda.threadIdx.x, cuda.threadIdx.y
    tip1, tjp1 = ti + 1, tj + 1
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    hij = h_in[i, j] if (i < m and j < n) else float32(0.0)
    shared_h[tip1, tjp1] = hij
    new_val = hij
    if ti == 0 and i > 0 and j < n:
        shared_h[0, tjp1] = h_in[i - 1, j]
    elif ti == cuda.blockDim.x - 1 and i < m - 1 and j < n:
        shared_h[cuda.blockDim.x + 1, tjp1] = h_in[i + 1, j]        
    if tj == 0 and j > 0 and i < m:
        shared_h[tip1, 0] = h_in[i, j - 1]
    elif tj == cuda.blockDim.y - 1 and j < n - 1 and i < m:
        shared_h[tip1, cuda.blockDim.y + 1] = h_in[i, j + 1]    
    cuda.syncthreads()
    if i > 0 and i < m - 1 and j > 0 and j < n - 1:
        new_val = float32(0.25) * (shared_h[tip1 - 1, tjp1] + shared_h[tip1 + 1, tjp1] + shared_h[tip1 , tjp1 - 1] + shared_h[tip1, tjp1 + 1]) # contraction
    if i < m and j < n:
        h_out[i, j] = new_val
    shared_d[t] = math.fabs(new_val - hij)
    tpb = cuda.blockDim.x * cuda.blockDim.y
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:                        
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:
        cuda.atomic.max(d, 0, shared_d[0])        

def sfwf_contraction_cuda_large_atomicmaxglosten(heights_in, eps, lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb_side=DEFAULT_TPB_SIDE, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAXGLOSTEN... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}]") # "glosten": global stencil (i.e. stencil on global memory)l
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    d = np.zeros(1, dtype=np.float32)
    dev_d = cuda.to_device(d)   
    tpb = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side      
    bpg = (bpg_j, bpg_i)
    if verbose:
        print(f"[bpg: {bpg}, tpb: {tpb}]")
    k = 0
    while True:
        sfwf_contraction_cuda_large_atomicmax_reset[1, 1](dev_d)
        sfwf_contraction_cuda_large_atomicmaxglosten_job[bpg, tpb](dev_h_in, dev_h_out, dev_d)
        k += 1
        if k % lazy_stop_check == 0:        
            dev_d.copy_to_host(ary=d)
            cuda.synchronize()
            if d[0] <= eps:
                break
        tmp = dev_h_in
        dev_h_in = dev_h_out
        dev_h_out = tmp        
    heights_out = dev_h_out.copy_to_host()
    d = d[0]    
    t2 = time.time()
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAXGLOSTEN DONE. [d_inf: {str(d)}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))    
def sfwf_contraction_cuda_large_atomicmaxglosten_job(h_in, h_out, d):       
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2
    j, i = cuda.grid(2)
    tj, ti = cuda.threadIdx.x, cuda.threadIdx.y    
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    d_ij = float32(0.0)
    if i < m and j < n:
        h_ij = h_in[i, j]
        if 0 < i < m - 1 and 0 < j < n - 1:
            new_val = float32(0.25) * (h_in[i - 1, j] + h_in[i + 1, j] + h_in[i, j - 1] + h_in[i, j + 1])
            h_out[i, j] = new_val
            d_ij = math.fabs(new_val - h_ij)
        else:
            h_out[i, j] = h_ij
    shared_d[t] = d_ij
    tpb = cuda.blockDim.x * cuda.blockDim.y
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:                        
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1    
    if t == 0:
        cuda.atomic.max(d, 0, shared_d[0])

def sfwf_contraction_cuda_large_hreducemax(heights_in, eps, lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb_side=DEFAULT_TPB_SIDE, tpb_reduce=DEFAULT_TPB, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE HREDUCEMAX... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}, tpb_reduce: {tpb_reduce}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    tpb_job = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side      
    bpg = (bpg_j, bpg_i)
    d = np.zeros(bpg_i * bpg_j, dtype=np.float32)
    dev_d = cuda.to_device(d)
    if verbose:       
        print(f"[job bpg: {bpg}, tpb: {tpb_job}]")
        print(f"[reduce bpg: {1}, tpb: {tpb_reduce}]")
    k = 0
    while True:
        sfwf_contraction_cuda_large_hreducemax_job[bpg, tpb_job](dev_h_in, dev_h_out, dev_d)        
        k += 1
        if k % lazy_stop_check == 0:
            sfwf_contraction_cuda_large_hreducemax_reduce[1, tpb_reduce](dev_d)
            dev_d.copy_to_host(ary=d)
            cuda.synchronize()        
            if d[0] <= eps:
                break        
        tmp = dev_h_in
        dev_h_in = dev_h_out
        dev_h_out = tmp 
    heights_out = dev_h_out.copy_to_host()
    d = d[0]    
    t2 = time.time()
    if verbose:
        print(f"SFWF ITERATE CONTRACTION CUDA LARGE HREDUCEMAX DONE. [d_inf: {str(d)} iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))    
def sfwf_contraction_cuda_large_hreducemax_job(h_in, h_out, d):         
    shared_h = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values 
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2
    j, i = cuda.grid(2)
    tj, ti = cuda.threadIdx.x, cuda.threadIdx.y
    tip1, tjp1 = ti + 1, tj + 1
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    hij = h_in[i, j] if (i < m and j < n) else float32(0.0)
    shared_h[tip1, tjp1] = hij
    new_val = hij
    if ti == 0 and i > 0 and j < n:
        shared_h[0, tjp1] = h_in[i - 1, j]
    if ti == cuda.blockDim.x - 1 and i < m - 1 and j < n:                
        shared_h[cuda.blockDim.x + 1, tjp1] = h_in[i + 1, j]
    if tj == 0 and j > 0 and i < m:
        shared_h[tip1, 0] = h_in[i, j - 1]
    if tj == cuda.blockDim.y - 1 and j < n - 1 and i < m:
        shared_h[tip1, cuda.blockDim.y + 1] = h_in[i, j + 1]    
    cuda.syncthreads()
    if i > 0 and i < m - 1 and j > 0 and j < n - 1 :
        new_val = float32(0.25) * (shared_h[tip1 - 1, tjp1] + shared_h[tip1 + 1, tjp1] + shared_h[tip1 , tjp1 - 1] + shared_h[tip1, tjp1 + 1]) # contraction
    if i < m and j < n:
        h_out[i, j] = new_val 
    shared_d[t] = math.fabs(new_val - hij)
    tpb = cuda.blockDim.x * cuda.blockDim.y
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:        
        b = cuda.blockIdx.x * cuda.gridDim.y + cuda.blockIdx.y
        d[b] = shared_d[0]

@cuda.jit(void(float32[:]))    
def sfwf_contraction_cuda_large_hreducemax_reduce(d):         
    shared_d = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    tpb = cuda.blockDim.x
    job_blocks = d.shape[0]
    ept = (job_blocks + tpb - 1) // tpb 
    t = cuda.threadIdx.x
    e = t
    shared_d[t] = float32(0.0)
    for _ in range(ept):
        if e < job_blocks:
            shared_d[t] = max(shared_d[t], d[e])
        e += tpb    
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:    
        d[0] = shared_d[0]

def sfwf_contraction_cuda_large_hreducemaxgs(heights_in, eps, lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb_side=DEFAULT_TPB_SIDE, tpb_reduce=DEFAULT_TPB, cores=DEFAULT_CORES, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE HREDUCEMAXGS... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}, tpb_reduce: {tpb_reduce}, cores: {cores}]") # "gs" suffix: grid-stride 
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    tpb_job = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side        
    bpg = (bpg_i, bpg_j)
    bpg_i_j = bpg_i * bpg_j
    tpb_gs = 128 # common choice for grid-stride loops
    bpg_gs = min(max(cores // tpb_gs, 1), tpb_reduce)   
    d = np.zeros(bpg_i_j, dtype=np.float32)
    dev_d = cuda.to_device(d)
    d_gs = np.zeros(bpg_gs, dtype=np.float32)
    dev_d_gs = cuda.to_device(d_gs)
    if verbose:       
        print(f"[job bpg: {bpg}, tpb: {tpb_job}]")
        print(f"[reduce1 (grid-stride) bpg: {bpg_gs}, tpb: {tpb_gs}]")
        print(f"[reduce2 bpg: 1, tpb: {tpb_reduce}]")
    k = 0
    while True:
        sfwf_contraction_cuda_large_hreducemax_job[bpg, tpb_job](dev_h_in, dev_h_out, dev_d) # same main kernal as in simple hreducemax
        k += 1
        if k % lazy_stop_check == 0:
            sfwf_contraction_cuda_large_hreducemaxgs_reduce1[bpg_gs, tpb_gs](dev_d, dev_d_gs)
            sfwf_contraction_cuda_large_hreducemaxgs_reduce2[1, tpb_reduce](dev_d_gs)
            dev_d_gs.copy_to_host(ary=d_gs)
            cuda.synchronize()
            if d_gs[0] <= eps:
                break        
        tmp = dev_h_in
        dev_h_in = dev_h_out
        dev_h_out = tmp        
    heights_out = dev_h_out.copy_to_host()
    d = d_gs[0]    
    t2 = time.time()
    if verbose:
        print(f"SFWF ITERATE CONTRACTION CUDA LARGE HREDUCEMAXGS DONE. [d_inf: {str(d)} iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:], float32[:]))    
def sfwf_contraction_cuda_large_hreducemaxgs_reduce1(d, d_gs):         
    shared_d = cuda.shared.array(128, dtype=float32) # corresponds to DEFAULT_TPB_GS
    tpb = cuda.blockDim.x
    bpg_i_j = d.shape[0]
    grid_stride = cuda.gridDim.x * tpb
    ept = (bpg_i_j + grid_stride - 1) // grid_stride
    t = cuda.threadIdx.x    
    e = cuda.grid(1) 
    shared_d[t] = float32(0.0)
    for _ in range(ept):
        if e < bpg_i_j:
            shared_d[t] = max(shared_d[t], d[e])
        e += grid_stride
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:    
        d_gs[cuda.blockIdx.x] = shared_d[0]

@cuda.jit(void(float32[:]))    
def sfwf_contraction_cuda_large_hreducemaxgs_reduce2(d_gs):
    shared_d = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    tpb = cuda.blockDim.x 
    t = cuda.threadIdx.x
    shared_d[t] = d_gs[t] if t < d_gs.shape[0] else float32(0.0)
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:    
        d_gs[0] = shared_d[0]

def sfwf_contraction_cuda_large_gridsync(heights_in, eps, tpb_side=DEFAULT_TPB_SIDE, max_bpg_gridsync=None, verbose=True):
    if verbose:
        print(f"SFWF CONTRACTION CUDA LARGE GRIDSYNC... [eps: {eps}, tpb_side: {tpb_side}, assumed max_bpg_gridsync: {max_bpg_gridsync}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    tpb = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side
    if verbose:
        print(f"[wanted ideal bpg: {(bpg_i, bpg_j)}]")
    if max_bpg_gridsync is None:
        t1_discover = time.time()
        compiled = sfwf_contraction_cuda_large_gridsync_job.overloads[(float32[:, :], float32, float32[:, :], float32[:], int32[:], boolean[:],)]
        max_bpg_gridsync = compiled.max_cooperative_grid_blocks(tpb)
        t2_discover = time.time()
        if verbose:
            print(f"[discovered max_bpg_gridsync: {max_bpg_gridsync}; time: {t2_discover - t1_discover} s]")
    ratio = min(np.sqrt(max_bpg_gridsync / (bpg_i * bpg_j)), 1.0)
    bpg_i = max(int(np.ceil(ratio * bpg_i)), 1)
    bpg_j = max(int(np.ceil(ratio * bpg_j)), 1)
    if bpg_i * bpg_j > max_bpg_gridsync:
        bpg_i -= 1
    if bpg_i * bpg_j > max_bpg_gridsync:
        bpg_j -= 1
    bpg = (bpg_j, bpg_i)
    tpg_i = bpg_i * tpb_side
    tpg_j = bpg_j * tpb_side
    ept_i = (heights_in.shape[0] + tpg_i - 1) // tpg_i
    ept_j = (heights_in.shape[1] + tpg_j - 1) // tpg_j
    ept = ept_i * ept_j
    if verbose:
        print(f"[bpg: {bpg}, tpb: {tpb}, ept: {ept}]")
    dev_d = cuda.to_device(np.zeros(1, dtype=np.float32))
    dev_k = cuda.to_device(np.zeros(1, dtype=np.int32))
    dev_stop_all = cuda.to_device(np.zeros(1, dtype=bool))       
    sfwf_contraction_cuda_large_gridsync_job[bpg, tpb](dev_h_in, eps, dev_h_out, dev_d, dev_k, dev_stop_all)     
    heights_out = dev_h_out.copy_to_host()
    d = dev_d.copy_to_host()[0]
    k = dev_k.copy_to_host()[0]
    cuda.synchronize()
    t2 = time.time()
    if verbose:
        print(f"SFWF ITERATE CONTRACTION CUDA LARGE GRIDSYNC DONE. [d_inf: {str(d)}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1        

@cuda.jit(void(float32[:, :], float32, float32[:, :], float32[:], int32[:], boolean[:]))    
def sfwf_contraction_cuda_large_gridsync_job(h_in, eps, h_out, d, k, stop_all):           
    shared_h = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values 
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2    
    tj, ti = cuda.threadIdx.x, cuda.threadIdx.y
    j0, i0 = cuda.grid(2)
    b = cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x # block 1d index
    g = cuda.cg.this_grid() 
    tip1, tjp1 = ti + 1, tj + 1
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    tpb = cuda.blockDim.x * cuda.blockDim.y
    tpg_j = cuda.gridDim.x * cuda.blockDim.x
    tpg_i = cuda.gridDim.y * cuda.blockDim.y
    ept_i = (m + tpg_i - 1) // tpg_i
    ept_j = (n + tpg_j - 1) // tpg_j
    ept = ept_i * ept_j
    mg_j = (n + tpg_j - 1) // tpg_j # super-grid dim j
    src = h_in
    dst = h_out
    while True: 
        for e in range(ept):
            ei, ej = e // mg_j, e % mg_j
            i = i0 + ei * tpg_i
            j = j0 + ej * tpg_j            
            hij = src[i, j] if (i < m and j < n) else float32(0.0)
            shared_h[tip1, tjp1] = hij
            if ti == 0 and i > 0 and i < m and j < n:
                shared_h[0, tjp1] = src[i - 1, j]
            if ti == cuda.blockDim.x - 1 and i < m - 1 and j < n:
                shared_h[cuda.blockDim.x + 1, tjp1] = src[i + 1, j]        
            if tj == 0 and j > 0 and j < n and i < m:
                shared_h[tip1, 0] = src[i, j - 1]
            if tj == cuda.blockDim.y - 1 and j < n - 1 and i < m:
                shared_h[tip1, cuda.blockDim.y + 1] = src[i, j + 1]    
            cuda.syncthreads()
            new_val = hij    
            if i > 0 and i < m - 1 and j > 0 and j < n - 1:
                new_val = float32(0.25) * (shared_h[tip1 - 1, tjp1] + shared_h[tip1 + 1, tjp1] + shared_h[tip1 , tjp1 - 1] + shared_h[tip1, tjp1 + 1]) # contraction
            if i < m and j < n:
                dst[i, j] = new_val 
            shared_d[t] = math.fabs(new_val - hij)        
            stride = tpb >> 1
            cuda.syncthreads()
            while stride > 0: # max-reduction        
                if t < stride:
                    shared_d[t] = max(shared_d[t], shared_d[t + stride])
                cuda.syncthreads()
                stride >>= 1
            if t == 0:
                cuda.atomic.max(d, 0, shared_d[0])
        g.sync()
        if b == 0 and t == 0:
            k[0] += 1
            if d[0] <= eps:
                stop_all[0] = True
            else:
                d[0] = float32(0.0)
        g.sync() 
        if stop_all[0]:
            break
        tmp = src
        src = dst
        dst = tmp

def sfwf_mc_cpu_numpy(heights, i, j, n_samples, seed=None, chunk_size=DEFAULT_MC_CPU_NUMPY_CHUNK_SIZE, verbose=True, verbose_gap_percent=0.1, collect_trajectories=False):
    if verbose:
        print(f"SFWF MC CPU NUMPY... [(i, j): {(i, j)}, n_samples: {n_samples:.1e}, seed: {seed}, chunk_size: {chunk_size}]")
    t1 = time.time()
    if seed is None:
        seed = 0
    np.random.seed(seed)      
    m, n = heights.shape    
    actions = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]], dtype=np.int8)    
    Gs = np.empty(n_samples, dtype=np.float32)    
    Ts = np.empty(n_samples, dtype=np.int32)
    trajectories = []
    verbose_gap = max(int(np.round(verbose_gap_percent * n_samples)), 1)
    for k in range(n_samples):
        t = 0
        s = np.array([i, j], dtype=np.int32)
        trajectory = np.array([s]) if collect_trajectories else None
        t1_epi = time.time()
        while True:
            random_chunk = np.random.randint(actions.shape[0], size=chunk_size)
            a = actions[random_chunk]
            sn = s + np.cumsum(a, axis=0)
            terminal_mask = (sn[:, 0] == 0) | (sn[:, 0] == m - 1) | (sn[:, 1] == 0) | (sn[:, 1] == n - 1)
            dt = np.where(terminal_mask)[0]
            if dt.size > 0:
                T = t + dt[0] + 1
                i_T, j_T = sn[dt[0]]
                Gs[k] = heights[i_T, j_T]
                Ts[k] = T
                if collect_trajectories:
                    trajectory = np.r_[trajectory, sn[:dt[0] + 1]]
                    trajectories.append(trajectory)                                
                break
            else:
                t += chunk_size
                s = sn[-1]
                if collect_trajectories:
                    trajectory = np.r_[trajectory, sn]            
        t2_epi = time.time()
        if verbose and (k + 1) % verbose_gap == 0:
            print(f"[{k + 1}/{n_samples}: border reached at t: {Ts[k]}, (i, j): {(int(i_T), int(j_T))}, h: {str(Gs[k])}; trajectory time: {t2_epi - t1_epi} s]")
    h_mean = np.mean(Gs)
    t2 = time.time()
    T_mean = np.mean(Ts) # computation of average trajectory length is not the main task (its time should not be included) 
    if verbose:
        print(f"SFWF MC CPU NUMPY DONE. [h_mean: {str(h_mean)}, T_mean: {T_mean}; time: {t2 - t1} s]")
    return h_mean, T_mean, t2 - t1, trajectories

def sfwf_mc_cuda(heights, i, j, n_samples, seed=None, rpt=DEFAULT_MC_CUDA_RPT, tpb=DEFAULT_MC_CUDA_TPB, verbose=True):
    if verbose:
        print(f"SFWF MC CUDA... [(i, j): {(i, j)}, wanted n_samples: {n_samples:.1e}, seed: {seed}, rpt: {rpt}, tpb: {tpb}]")
    t1 = time.time()
    if seed is None:
        seed = 0
    min_n_generators = (n_samples + rpt - 1) // rpt
    bpg_walk = (min_n_generators + tpb - 1) // tpb
    if verbose:
        print(f"[min_n_generators: {min_n_generators}, bpg_walk: {bpg_walk}]")
        print("[initialization of random generators...]")
    t1_generators = time.time()
    dev_random_generators = create_xoroshiro128p_states(bpg_walk * tpb, seed=seed)
    cuda.synchronize()
    t2_generators = time.time()
    if verbose:
        print(f"[initialization of random generators done; count: {bpg_walk * tpb}, memory: {dev_random_generators.nbytes / 1024**2:.3f} MiB, time: {t2_generators - t1_generators} s]")
    if verbose:
        print(f"[random walks...; bpg_walk: {bpg_walk}, tpb: {tpb}]")
    t1_walk = time.time()        
    dev_h = cuda.to_device(heights)    
    dev_G_means = cuda.device_array(bpg_walk, dtype=np.float32)
    dev_T_means = cuda.device_array(bpg_walk, dtype=np.float32)
    dev_G_mean = cuda.device_array(1, dtype=np.float32)
    dev_T_mean = cuda.device_array(1, dtype=np.float32)        
    sfwf_mc_cuda_walk[bpg_walk, tpb](dev_h, i, j, dev_random_generators, rpt, dev_G_means, dev_T_means)
    cuda.synchronize()
    t2_walk = time.time()
    if verbose:
        print(f"[random walks done; time: {t2_walk - t1_walk} s]")
        print(f"[mean-reduction...; bpg: {1}, tpb: {tpb}]")    
    t1_red = time.time()
    sfwf_mc_cuda_reduce[1, tpb](dev_G_means, dev_T_means, bpg_walk, dev_G_mean, dev_T_mean)    
    h_mean = dev_G_mean.copy_to_host()[0]
    T_mean = dev_T_mean.copy_to_host()[0]
    cuda.synchronize()
    t2_red = time.time()
    if verbose:
        print(f"[mean-reduction done; time: {t2_red - t1_red} s]")
    t2 = time.time()
    t2_t1 = t2 - t1
    t2_t1_without_generators = t2_t1 - (t2_generators - t1_generators)
    if verbose:
        print(f"SFWF MC CUDA DONE. [h_mean: {str(h_mean)}, T_mean: {T_mean}, n_samples de facto: {bpg_walk * tpb * rpt:.1e}; time: {t2 - t1} s, time without generators initalization: {t2_t1_without_generators} s]")
    return h_mean, T_mean, t2_t1_without_generators, None # trajectories not memorized (hence returned)

const_actions_host = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]], dtype=np.int8)
@cuda.jit(void(float32[:, :], int32, int32, xoroshiro128p_type[:], int32, float32[:], float32[:]))    
def sfwf_mc_cuda_walk(h, i, j, random_generators, rpt, G_means, T_means):           
    shared_Gs = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB 
    shared_Ts = cuda.shared.array(512, dtype=int32) # corresponds to DEFAULT_TPB
    const_actions = cuda.const.array_like(const_actions_host)
    t = cuda.threadIdx.x
    t_global = cuda.grid(1)
    tpb = cuda.blockDim.x    
    m, n = h.shape    
    shared_Gs[t] = float32(0.0)
    shared_Ts[t] = int32(0.0)    
    for _ in range(rpt):
        T = int32(0)
        si, sj = i, j
        while not (si == 0 or si == m - 1 or sj == 0 or sj == n - 1):
            rand_action = int8(xoroshiro128p_uniform_float32(random_generators, t_global) * 4)
            si += const_actions[rand_action, 0]
            sj += const_actions[rand_action, 1]
            T += 1
        shared_Gs[t] += h[si, sj]
        shared_Ts[t] += T
    stride = tpb >> 1
    cuda.syncthreads()     
    while stride > 0: # sum-reduction
        if t < stride:
            t_s = t + stride
            shared_Gs[t] += shared_Gs[t_s]
            shared_Ts[t] += shared_Ts[t_s]
        cuda.syncthreads()
        stride >>= 1            
    if t == 0:
        n_samples_in_block = tpb * rpt
        G_means[cuda.blockIdx.x] = shared_Gs[0] / n_samples_in_block 
        T_means[cuda.blockIdx.x] = shared_Ts[0] / n_samples_in_block             

@cuda.jit(void(float32[:], float32[:], int32, float32[:], float32[:]))    
def sfwf_mc_cuda_reduce(G_means, T_means, bpg_walk, G_mean, T_mean):           
    shared_G_means = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB 
    shared_T_means = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    t = cuda.threadIdx.x    
    tpb = cuda.blockDim.x
    ept = (bpg_walk + tpb - 1) // tpb
    shared_G_means[t] = float32(0.0)
    shared_T_means[t] = float32(0.0)
    e = t
    for _ in range(ept):
        if e < bpg_walk:
            shared_G_means[t] += G_means[e]
            shared_T_means[t] += T_means[e]
        e += tpb
    stride = tpb >> 1 # sum-reduction
    cuda.syncthreads()
    while stride > 0:
        if t < stride:
            t_s = t + stride
            shared_G_means[t] += shared_G_means[t_s]
            shared_T_means[t] += shared_T_means[t_s]
        cuda.syncthreads()
        stride >>= 1
    if t == 0:
        G_mean[0] = shared_G_means[0] / bpg_walk
        T_mean[0] = shared_T_means[0] / bpg_walk    
