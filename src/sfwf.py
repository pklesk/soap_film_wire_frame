import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from numba import cuda
from numba import void, int8, int32, float32, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_type
from numba.core.errors import NumbaPerformanceWarning
import math
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
import os
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
from pprint import pprint
from utils import cpu_and_system_props, gpu_props

DEFAULT_TPB = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2
DEFAULT_TPB_SIDE = 16 
DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE = 64
DEFAULT_CONTRACTION_CUDA_LARGE_GRIDSYNC_MAX_BPG = 400
DEFAULT_MC_CUDA_TPB = 64
DEFAULT_MC_CPU_NUMPY_CHUNK_SIZE = 250 # 10**3 (250 for nice plot)
DEFAULT_MC_CPU_NUMPY_VERBOSE_GAP = 10**4
DEFAULT_MC_CUDA_RPT = 10 # repetitions (walks) per thread (owing to this, fewer random generators can be initialized and CUDA blocks scheduled) 

def fourier_sum(t, a, b):
    ks = np.arange(a.shape[0])
    args = 2 * np.pi * ks * t + b
    return np.sum(a * np.sin(args))

def random_wire_frame(fourier_n, fourier_amplitude, border_n, seed=0):
    print(f"RANDOM WIRE FRAME... [fourier_n: {fourier_n}, fourier_amplitude: {fourier_amplitude}, border_n: {border_n}, seed: {seed}]")
    np.random.seed(seed)
    dtype = np.float32
    a = np.random.randn(fourier_n).astype(dtype) * fourier_amplitude
    b = np.random.randn(fourier_n).astype(dtype)
    ts = np.arange(4 * border_n - 4, dtype=dtype) / (4 * border_n - 4)
    border = np.array([fourier_sum(t, a, b) for t in ts])    
    heights = np.zeros((border_n, border_n), dtype=dtype)
    top = [(0, j) for j in range(border_n - 1)]
    right = [(i, border_n - 1) for i in range(0, border_n - 1)]
    bottom = [(border_n - 1, i) for i in range(border_n - 1, 0, -1)]
    left = [(j, 0) for j in range(border_n - 1, 0, -1)]        
    indexes = top + right + bottom + left
    for k, (i, j) in enumerate(indexes):
        heights[i, j] = border[k]
    print
    print("RANDOM WIRE FRAME DONE.")    
    return border, heights
                
def sfwf_plot(border, heights, title="", subtitle=""): 
    figsize = (12, 6)    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[3, 3])
    ax1 = fig.add_subplot(gs[0])
    ax1.set_box_aspect(0.25)
    ax1.axhline(y=0.0, color="k", linewidth=1)     
    ax1.plot(border, color="r", linewidth=2)
    ax1.set_xlabel("BORDER")
    ax1.set_ylabel("HEIGHT")
    ax1.set_title("WIRE FRAME HEIGHTS ALONG BORDER", fontsize=14)
    border_n = len(border)
    corner_indices = [0, border_n // 4, border_n // 2, 3 * border_n // 4, border_n - 1]
    corner_labels = ["A", "B", "C", "D", "A"]
    ax1.scatter(corner_indices, [border[i] for i in corner_indices], color="k", s=20, zorder=3)
    for i, label in enumerate(corner_labels):
        ax1.text(corner_indices[i], border[corner_indices[i]], f" {label}", fontsize=14, color="k")        
    ax2 = fig.add_subplot(gs[1], projection="3d")         
    X1, X2 = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij")
    ax2.plot_surface(X1, X2, heights, cmap="coolwarm", edgecolor="k", linewidth=0.5, alpha=0.75)
    ax2.plot(X1[0, :], X2[0, :], heights[0, :], color="r", linewidth=5) 
    ax2.plot(X1[-1, :], X2[-1, :], heights[-1, :], color="r", linewidth=5)
    ax2.plot(X1[:, 0], X2[:, 0], heights[:, 0], color="r", linewidth=5)
    ax2.plot(X1[:, -1], X2[:, -1], heights[:, -1], color="r", linewidth=5)
    corners_3d = [
        (0, 0, heights[0, 0]),
        (0, heights.shape[1] - 1, heights[0, -1]),
        (heights.shape[0] - 1, heights.shape[1] - 1, heights[-1, -1]),        
        (heights.shape[0] - 1, 0 , heights[-1, 0])        
    ]    
    shift_point = 0.5
    shifts = [
        (-shift_point, -shift_point, shift_point),
        (shift_point, -shift_point, shift_point),
        (shift_point, shift_point, shift_point),
        (shift_point, -shift_point, shift_point)        
    ]
    shift_label = 0.1 * (np.max(heights) - np.min(heights))
    for k, (i, j, h) in enumerate(corners_3d):
        ax2.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=20, depthshade=False, zorder=20)
        ax2.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)                 
    ax2.set_zlabel("HEIGHT")
    ax2.set_box_aspect([1, 1, 0.5])    
    fig.text(0.725, 0.9, title, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    fig.text(0.725, 0.85, subtitle, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()  
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95)
    fig.canvas.manager.set_window_title("SOAP FILM IN WIRE FRAME: CONTRACTION ITERATION")
    plt.show()

def sfwf_plot_large(border, heights, title="", subtitle="", heights_extra=None, title_extra="", subtitle_extra="", ): 
    figsize = (18, 6) 
    fig = plt.figure(figsize=figsize)
    width_ratios=[2, 3, 3]
    gs = GridSpec(1, 3, width_ratios=width_ratios)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_box_aspect(0.25)
    ax1.axhline(y=0.0, color="k", linewidth=1)     
    ax1.plot(border, color="r", linewidth=2)
    ax1.set_xlabel("BORDER")
    ax1.set_ylabel("HEIGHT")
    ax1.set_title("WIRE FRAME HEIGHTS ALONG BORDER", fontsize=14)
    border_n = len(border)
    corner_indices = [0, border_n // 4, border_n // 2, 3 * border_n // 4, border_n - 1]
    corner_labels = ["A", "B", "C", "D", "A"]
    ax1.scatter(corner_indices, [border[i] for i in corner_indices], color="k", s=20, zorder=3)
    for i, label in enumerate(corner_labels):
        ax1.text(corner_indices[i], border[corner_indices[i]], f" {label}", fontsize=14, color="k")        
    ax2 = fig.add_subplot(gs[1], projection="3d")         
    X1, X2 = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij")
    ax2.plot_surface(X1, X2, heights, cmap="coolwarm", edgecolor="k", linewidth=0.5, alpha=0.75)
    ax2.plot(X1[0, :], X2[0, :], heights[0, :], color="r", linewidth=5) 
    ax2.plot(X1[-1, :], X2[-1, :], heights[-1, :], color="r", linewidth=5)
    ax2.plot(X1[:, 0], X2[:, 0], heights[:, 0], color="r", linewidth=5)
    ax2.plot(X1[:, -1], X2[:, -1], heights[:, -1], color="r", linewidth=5)
    corners_3d = [
        (0, 0, heights[0, 0]),
        (0, heights.shape[1] - 1, heights[0, -1]),
        (heights.shape[0] - 1, heights.shape[1] - 1, heights[-1, -1]),        
        (heights.shape[0] - 1, 0 , heights[-1, 0])        
    ] 
    shift_point = 0.5
    shifts = [
        (-shift_point, -shift_point, shift_point),
        (shift_point, -shift_point, shift_point),
        (shift_point, shift_point, shift_point),
        (shift_point, -shift_point, shift_point)        
    ]
    shift_label = 0.1 * (np.max(heights) - np.min(heights))
    for k, (i, j, h) in enumerate(corners_3d):
        ax2.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=20, depthshade=False, zorder=20)
        ax2.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)    
    ax2.set_zlabel("HEIGHT")
    ax2.set_box_aspect([1, 1, 0.5])
    x_shift = (width_ratios[0] + 0.5 * width_ratios[1]) / sum(width_ratios)    
    fig.text(x_shift, 0.9, title, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    fig.text(x_shift, 0.85, subtitle, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    plt.subplots_adjust(wspace=0.5)
    ax3 = fig.add_subplot(gs[2], projection="3d")         
    X1, X2 = np.meshgrid(np.arange(heights_extra.shape[0]), np.arange(heights_extra.shape[1]), indexing="ij")
    ax3.plot_surface(X1, X2, heights_extra, cmap="coolwarm", edgecolor="k", linewidth=0.5, alpha=0.75)
    ax3.plot(X1[0, :], X2[0, :], heights_extra[0, :], color="r", linewidth=5) 
    ax3.plot(X1[-1, :], X2[-1, :], heights_extra[-1, :], color="r", linewidth=5)
    ax3.plot(X1[:, 0], X2[:, 0], heights_extra[:, 0], color="r", linewidth=5)
    ax3.plot(X1[:, -1], X2[:, -1], heights_extra[:, -1], color="r", linewidth=5)
    corners_3d = [
        (0, 0, heights_extra[0, 0]),
        (0, heights_extra.shape[1] - 1, heights_extra[0, -1]),
        (heights_extra.shape[0] - 1, heights_extra.shape[1] - 1, heights_extra[-1, -1]),        
        (heights_extra.shape[0] - 1, 0 , heights_extra[-1, 0]) 
    ]        
    for k, (i, j, h) in enumerate(corners_3d):
        ax3.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=20, depthshade=False, zorder=20)
        ax3.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)                 
    ax3.set_zlabel("HEIGHT")
    ax3.set_box_aspect([1, 1, 0.5])
    x_shift = (width_ratios[0] + width_ratios[1] + 0.5 * width_ratios[2]) / sum(width_ratios)
    titles_shift = 0.025 
    fig.text(x_shift - titles_shift, 0.925, title_extra, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)    
    fig.text(x_shift - titles_shift, 0.875, subtitle_extra, fontsize=15, ha="center", va="bottom") # instead of: ax2.set_title(title)        
    plt.tight_layout()  
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95, wspace=0.05)
    fig.canvas.manager.set_window_title("SOAP FILM IN WIRE FRAME (CONTRACTION ITERATION)")
    plt.show()

def sfwf_plot_mc_trajectories(heights, trajectories, h_mean):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.set_box_aspect([1, 1, 0.5])
    fig.suptitle("MONTE CARLO TRAJECTORIES TOWARDS WIRE FRAME BORDERS")    
    X1, X2 = np.meshgrid(np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij")
    red_edge_color = (1, 0, 0, 0.9)
    ax1.plot(X1[0, :], X2[0, :], heights[0, :], color=red_edge_color, linewidth=3) 
    ax1.plot(X1[-1, :], X2[-1, :], heights[-1, :], color=red_edge_color, linewidth=3)
    ax1.plot(X1[:, 0], X2[:, 0], heights[:, 0], color=red_edge_color, linewidth=3)
    ax1.plot(X1[:, -1], X2[:, -1], heights[:, -1], color=red_edge_color, linewidth=3)    
    # rectangle for domain border at level = 0.0
    ax1.plot([0, heights.shape[0] - 1, heights.shape[0] - 1, 0, 0],
             [0, 0, heights.shape[1] - 1, heights.shape[1] - 1, 0],
             [0, 0, 0, 0, 0], color="k", linewidth=1)
    # plane at level = 0.0 with grid on it
    ax1.plot_surface(X1, X2, np.zeros_like(heights), color="gray", alpha=0.05, rstride=10, cstride=10, zorder=-1)
    for i in range(heights.shape[0]):
        ax1.plot([i, i], [0, heights.shape[1] - 1], [0, 0], color="lightgray", linestyle='-', linewidth=0.5, zorder=-1)
    for j in range(heights.shape[1]):
        ax1.plot([0, heights.shape[0] - 1], [j, j], [0, 0], color="lightgray", linestyle='-', linewidth=0.5, zorder=-1)        
    # trajectories
    colors = ["#9B30FF", "#0033B3", "#009940"]
    for k, traj in enumerate(trajectories):
        color = colors[k % len(colors)]    
        ax1.plot(traj[:, 0], traj[:, 1], np.zeros_like(traj[:, 0]), color=color, linewidth=1.0, alpha=1.0)
        end_i, end_j = traj[-1]    
        ax1.scatter(end_i, end_j, heights[end_i, end_j], color=color, s=15, zorder=10)        
        end_h = heights[int(end_i), int(end_j)]
        ax1.plot([end_i, end_i], [end_j, end_j], [0, end_h], linestyle="-", linewidth=2.0, color=color)
    # segment line for h_mean
    ax1.plot([traj[0, 0], traj[0, 0]], [traj[0, 1], traj[0, 1]], [0, h_mean], linestyle="-", color="k")    
    ax1.scatter(traj[0, 0], traj[0, 1], 0, color="k", s=15, zorder=10)
    ax1.scatter(traj[0, 0], traj[0, 1], h_mean, color="k", s=15, zorder=10)  
    ax1.text(traj[0, 0] - 3.0, traj[0, 1] - 2.0 , 0.1 * h_mean, r"$\widehat{h}$", color="k", fontsize=15, zorder=15)    
    # segment lines for corners
    corners = [(0, 0), (0, heights.shape[1] - 1), (heights.shape[0] - 1, heights.shape[1] - 1), (heights.shape[0] - 1, 0)]
    corner_labels = ["A", "B", "C", "D"] # corner labels
    for i, j in corners:
        corner_h = heights[i, j]
        ax1.plot([i, i], [j, j], [0, corner_h], color="k", linestyle="-", linewidth=1.0, zorder=10)        
    corners_3d = [
        (0, 0, heights[0, 0]),
        (0, heights.shape[1] - 1, heights[0, -1]),
        (heights.shape[0] - 1, heights.shape[1] - 1, heights[-1, -1]),        
        (heights.shape[0] - 1, 0 , heights[-1, 0])        
    ] 
    shift_point = 0.25
    shifts = [
        (-shift_point, -shift_point, shift_point),
        (shift_point, -shift_point, shift_point),
        (shift_point, shift_point, shift_point),
        (shift_point, -shift_point, shift_point)        
    ]
    shift_label = 0.1 * (np.max(heights) - np.min(heights))
    for k, (i, j, h) in enumerate(corners_3d):
        ax1.scatter(i + shifts[k][0], j + shifts[k][1], h + shifts[k][2] * np.sign(h), color="k", s=10, depthshade=False, zorder=20)
        ax1.text(i + shifts[k][0], j + shifts[k][1], h + (shifts[k][2] + shift_label) * np.sign(h), f" {corner_labels[k]}", fontsize=14, color="k", zorder=20)    
    ax1.set_zlabel("HEIGHT")
    plt.tight_layout()  
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.05, right=0.95)
    fig.canvas.manager.set_window_title("SOAP FILM IN WIRE FRAME: MONTE CARLO TRAJECTORIES")    
    plt.show()
        
def sfwf_contraction_cpu_numpy(heights_in, eps, verbose=False):    
    print(f"SFWF CONTRACTION CPU NUMPY... [eps: {eps}]")
    t1 = time.time()
    h = np.copy(heights_in[1:-1, 1:-1]) # inside (no border)
    k = 0    
    d = np.inf
    while True:
        h_l = np.c_[heights_in[1:-1, 0], h[:, :-1]]
        h_r = np.c_[h[:, 1:], heights_in[1:-1, -1]]
        h_t = np.r_[heights_in[0, 1:-1][np.newaxis, :], h[:-1, :]]        
        h_b = np.r_[h[1:, :], heights_in[-1, 1:-1][np.newaxis, :]]        
        h_new = 0.25 * (h_t + h_b + h_l + h_r)
        d = np.max(np.abs(h_new - h))
        h = h_new
        k += 1
        if verbose:
            print(f"[iteration {k} done; d_inf: {d}]")
        if d <= eps:
            break        
    t2 = time.time()
    heights_out = np.copy(heights_in)
    heights_out[1:-1, 1:-1] = h
    print(f"SFWF CONTRACTION CPU NUMPY DONE. [d_inf: {d}, iterations: {k}, time: {t2 - t1} s]")
    return heights_out, d, k, t2 - t1

def sfwf_contraction_cuda_small(heights_in, eps, tpb=DEFAULT_TPB):
    print(f"SFWF CONTRACTION CUDA SMALL... [eps: {eps}, tpb: {tpb}]")
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
    print(f"SFWF CONTRACTION CUDA SMALL DONE. [d_inf: {d}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:, :], float32, float32[:, :], float32[:], int32[:]))    
def sfwf_contraction_cuda_small_job(h_in, eps, h_out, d, k):         
    shared_h = cuda.shared.array((64, 64), dtype=float32) # side corresponds to DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE 
    shared_d = cuda.shared.array((64, 64), dtype=float32) # side corresponds to DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE
    tpb = cuda.blockDim.x
    t = cuda.threadIdx.x
    m, n = h_in.shape
    m_n = int32(m * n)
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
        for q in range(1, ept): # gathering max point-wise differences within first tpb entries of array (preparation before reduction) 
            if t < m_n:
                e = t + q * tpb
                if e < m_n:
                    i_t, j_t = t // n, t % n
                    i_e, j_e = e // n, e % n                            
                    shared_d[i_t, j_t] = max(shared_d[i_t, j_t], shared_d[i_e, j_e])                        
        stride = tpb >> 1  
        cuda.syncthreads()
        while stride > 0: # max-reduction (over 2d shared array)
            if t < stride:
                t_s = t + stride
                if t_s < m_n:
                    i, j = t // n, t % n                
                    i_s, j_s = t_s // n, t_s % n                
                    shared_d[i, j] = max(shared_d[i, j], shared_d[i_s, j_s])
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

def sfwf_contraction_cuda_large_atomicmax(heights_in, eps, lazy_stop_check=100, tpb_side=DEFAULT_TPB_SIDE):
    print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAX... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    d = np.zeros(1, dtype=np.float32)
    dev_d = cuda.to_device(d)   
    tpb = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side      
    bpg = (bpg_i, bpg_j)
    print(f"[bpg: {bpg}]")
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
    print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAX DONE. [d_inf: {d}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:]))    
def sfwf_contraction_cuda_large_atomicmax_reset(d): # called exactly for 1 thread
    d[0] = float32(0.0) 

@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))    
def sfwf_contraction_cuda_large_atomicmax_job(h_in, h_out, d):       
    shared_h_in = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values 
    shared_h_out = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2
    i, j = cuda.grid(2)
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y
    tip1, tjp1 = ti + 1, tj + 1
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    hij = h_in[i, j] if (i < m and j < n) else float32(0.0)
    shared_h_in[tip1, tjp1] = hij
    shared_h_out[tip1, tjp1] = hij
    if ti == 0 and i > 0:
        shared_h_in[0, tjp1] = h_in[i - 1, j]
    elif ti == cuda.blockDim.x - 1 and i < m - 1:
        shared_h_in[cuda.blockDim.x + 1, tjp1] = h_in[i + 1, j]        
    if tj == 0 and j > 0:
        shared_h_in[tip1, 0] = h_in[i, j - 1]
    elif tj == cuda.blockDim.y - 1 and j < n - 1:
        shared_h_in[tip1, cuda.blockDim.y + 1] = h_in[i, j + 1]    
    cuda.syncthreads()    
    if i > 0 and i < m - 1 and j > 0 and j < n - 1 :
        shared_h_out[tip1, tjp1] = float32(0.25) * (shared_h_in[tip1 - 1, tjp1] + shared_h_in[tip1 + 1, tjp1] + shared_h_in[tip1 , tjp1 - 1] + shared_h_in[tip1, tjp1 + 1]) # contraction
    if i < m and j < n:
        h_out[i, j] = shared_h_out[tip1, tjp1]
    cuda.syncthreads()
    shared_d[t] = math.fabs(shared_h_out[tip1, tjp1] - shared_h_in[tip1, tjp1])
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
        
def sfwf_contraction_cuda_large_atomicmax_globalmem(heights_in, eps, lazy_stop_check=100, tpb_side=DEFAULT_TPB_SIDE):
    print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAX GLOBAL MEM... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    d = np.zeros(1, dtype=np.float32)
    dev_d = cuda.to_device(d)   
    tpb = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side      
    bpg = (bpg_i, bpg_j)
    print(f"[bpg: {bpg}]")
    k = 0
    while True:
        sfwf_contraction_cuda_large_atomicmax_globalmem_reset[1, 1](dev_d)
        sfwf_contraction_cuda_large_atomicmax_globalmem_job[bpg, tpb](dev_h_in, dev_h_out, dev_d)
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
    print(f"SFWF CONTRACTION CUDA LARGE ATOMICMAX GLOBALMEM DONE. [d_inf: {d}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:]))    
def sfwf_contraction_cuda_large_atomicmax_globalmem_reset(d): # called exactly for 1 thread
    d[0] = float32(0.0)         

@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))    
def sfwf_contraction_cuda_large_atomicmax_globalmem_job(h_in, h_out, d):       
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2
    i, j = cuda.grid(2)
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y    
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    if i < m and j < n:    
        inside = (i > 0 and i < m - 1 and j > 0 and j < n - 1)    
        if inside:
            h_out[i, j] = float32(0.25) * (h_in[i - 1, j] + h_in[i + 1, j] + h_in[i, j - 1] + h_in[i, j + 1]) # contraction
        else:
            h_out[i, j] = h_in[i, j]                    
    cuda.syncthreads()    
    shared_d[t] = math.fabs(h_out[i, j] - h_in[i, j]) if i < m and j < n else float32(0.0)
    cuda.syncthreads()
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

def sfwf_contraction_cuda_large_gridreducemax(heights_in, eps, lazy_stop_check=100, tpb_side=DEFAULT_TPB_SIDE, tpb_reduce=DEFAULT_TPB):
    print(f"SFWF CONTRACTION CUDA LARGE GRIDREDUCEMAX... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb_side: {tpb_side}, tpb_reduce: {tpb_reduce}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    tpb_job = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side      
    bpg = (bpg_i, bpg_j)
    d = np.zeros(bpg_i * bpg_j, dtype=np.float32)
    dev_d = cuda.to_device(d)       
    print(f"[bpg: {bpg}]")
    k = 0
    while True:
        sfwf_contraction_cuda_large_gridreducemax_reset[1, 1](dev_d)
        sfwf_contraction_cuda_large_gridreducemax_job[bpg, tpb_job](dev_h_in, dev_h_out, dev_d)
        sfwf_contraction_cuda_large_gridreducemax_reduce[1, tpb_reduce](dev_d)
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
    print(f"SFWF ITERATE CONTRACTION CUDA LARGE GRIDREDUCEMAX DONE. [d_inf: {d}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1

@cuda.jit(void(float32[:]))    
def sfwf_contraction_cuda_large_gridreducemax_reset(d):
    d[0] = float32(0.0)   

@cuda.jit(void(float32[:, :], float32[:, :], float32[:]))    
def sfwf_contraction_cuda_large_gridreducemax_job(h_in, h_out, d):         
    shared_h_in = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values 
    shared_h_out = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2
    i, j = cuda.grid(2)
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y
    tip1, tjp1 = ti + 1, tj + 1
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    hij = h_in[i, j] if (i < m and j < n) else float32(0.0)
    shared_h_in[tip1, tjp1] = hij
    shared_h_out[tip1, tjp1] = hij
    if ti == 0 and i > 0:
        shared_h_in[0, tjp1] = h_in[i - 1, j]
    if ti == cuda.blockDim.x - 1 and i < m - 1:
        shared_h_in[cuda.blockDim.x + 1, tjp1] = h_in[i + 1, j]        
    if tj == 0 and j > 0:
        shared_h_in[tip1, 0] = h_in[i, j - 1]
    if tj == cuda.blockDim.y - 1 and j < n - 1:
        shared_h_in[tip1, cuda.blockDim.y + 1] = h_in[i, j + 1]    
    cuda.syncthreads()
    if i > 0 and i < m - 1 and j > 0 and j < n - 1 :
        shared_h_out[tip1, tjp1] = float32(0.25) * (shared_h_in[tip1 - 1, tjp1] + shared_h_in[tip1 + 1, tjp1] + shared_h_in[tip1 , tjp1 - 1] + shared_h_in[tip1, tjp1 + 1]) # contraction
    if i < m and j < n:
        h_out[i, j] = shared_h_out[tip1, tjp1] 
    cuda.syncthreads()
    shared_d[t] = math.fabs(shared_h_out[tip1, tjp1] - shared_h_in[tip1, tjp1])
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
def sfwf_contraction_cuda_large_gridreducemax_reduce(d):         
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
    cuda.syncthreads()
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:    
        d[0] = shared_d[0]

def sfwf_contraction_cuda_large_gridsync(heights_in, eps, tpb_side=DEFAULT_TPB_SIDE, max_bpg_gridsync=DEFAULT_CONTRACTION_CUDA_LARGE_GRIDSYNC_MAX_BPG, discover_max_bpg_gridsync=True):
    print(f"SFWF CONTRACTION CUDA LARGE GRIDSYNC... [eps: {eps}, tpb_side: {tpb_side}, assumed max_bpg_gridsync: {max_bpg_gridsync}, discover_max_bpg_gridsync: {discover_max_bpg_gridsync}]")
    t1 = time.time()
    dev_h_in = cuda.to_device(heights_in)
    dev_h_out = cuda.device_array_like(heights_in)
    tpb = (tpb_side, tpb_side)
    bpg_i = (heights_in.shape[0] + tpb_side - 1) // tpb_side
    bpg_j = (heights_in.shape[1] + tpb_side - 1) // tpb_side
    print(f"[wanted ideal bpg: {(bpg_i, bpg_j)}]")
    if discover_max_bpg_gridsync:
        compiled = sfwf_contraction_cuda_large_gridsync_job.overloads[(float32[:, :], float32, float32[:, :], float32[:], int32[:], boolean[:],)]
        max_bpg_gridsync = compiled.max_cooperative_grid_blocks(tpb)
        print(f"[discovered max_bpg_gridsync: {max_bpg_gridsync}]")
    ratio = min(np.sqrt(max_bpg_gridsync / (bpg_i * bpg_j)), 1.0)
    bpg_i = max(int(np.ceil(ratio * bpg_i)), 1)
    bpg_j = max(int(np.ceil(ratio * bpg_j)), 1)
    if bpg_i * bpg_j > max_bpg_gridsync:
        bpg_i -= 1
    if bpg_i * bpg_j > max_bpg_gridsync:
        bpg_j -= 1
    bpg = (bpg_i, bpg_j)
    tpg_i = bpg_i * tpb_side
    tpg_j = bpg_j * tpb_side
    ept_i = (heights_in.shape[0] + tpg_i - 1) // tpg_i
    ept_j = (heights_in.shape[1] + tpg_j - 1) // tpg_j
    ept = ept_i * ept_j
    print(f"[bpg: {bpg}, ept: {ept}]")
    dev_d = cuda.to_device(np.zeros(1, dtype=np.float32))
    dev_k = cuda.to_device(np.zeros(1, dtype=np.int32))
    dev_stop_all = cuda.to_device(np.zeros(1, dtype=bool))       
    sfwf_contraction_cuda_large_gridsync_job[bpg, tpb](dev_h_in, eps, dev_h_out, dev_d, dev_k, dev_stop_all)     
    heights_out = dev_h_out.copy_to_host()
    d = dev_d.copy_to_host()[0]
    k = dev_k.copy_to_host()[0]
    cuda.synchronize()
    t2 = time.time()
    print(f"SFWF ITERATE CONTRACTION CUDA LARGE GRIDSYNC DONE. [d_inf: {d}, iterations: {k}, time: {t2 - t1} s]")    
    return heights_out, d, k, t2 - t1        

@cuda.jit(void(float32[:, :], float32, float32[:, :], float32[:], int32[:], boolean[:]))    
def sfwf_contraction_cuda_large_gridsync_job(h_in, eps, h_out, d, k, stop_all):           
    shared_h_in = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values 
    shared_h_out = cuda.shared.array((16 + 2, 16 + 2), dtype=float32) # corresponds to DEFAULT_TPB_SIDE + padding for neighbors' values
    shared_d = cuda.shared.array(16**2, dtype=float32) # corresponds to DEFAULT_TPB_SIDE**2    
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y
    i0, j0 = cuda.grid(2)
    b = cuda.blockIdx.x * cuda.blockDim.x + cuda.blockIdx.y # block 1d index
    g = cuda.cg.this_grid() 
    tip1, tjp1 = ti + 1, tj + 1
    t = ti * cuda.blockDim.y + tj 
    m, n = h_in.shape
    tpb = cuda.blockDim.x * cuda.blockDim.y
    tpg_i = cuda.gridDim.x * cuda.blockDim.x
    tpg_j = cuda.gridDim.y * cuda.blockDim.y
    ept_i = (m + tpg_i - 1) // tpg_i
    ept_j = (n + tpg_j - 1) // tpg_j
    ept = ept_i * ept_j
    mg_j = (n + tpg_j - 1) // tpg_j # super-grid dim j
    while True:
        for e in range(ept):
            ei, ej = e // mg_j, e % mg_j
            i = i0 + ei * tpg_i
            j = j0 + ej * tpg_j
            hij = h_in[i, j] if (i < m and j < n) else float32(0.0)
            shared_h_in[tip1, tjp1] = hij
            shared_h_out[tip1, tjp1] = hij
            if ti == 0 and i > 0:
                shared_h_in[0, tjp1] = h_in[i - 1, j]
            if ti == cuda.blockDim.x - 1 and i < m - 1:
                shared_h_in[cuda.blockDim.x + 1, tjp1] = h_in[i + 1, j]        
            if tj == 0 and j > 0:
                shared_h_in[tip1, 0] = h_in[i, j - 1]
            if tj == cuda.blockDim.y - 1 and j < n - 1:
                shared_h_in[tip1, cuda.blockDim.y + 1] = h_in[i, j + 1]    
            cuda.syncthreads()    
            if i > 0 and i < m - 1 and j > 0 and j < n - 1 :
                shared_h_out[tip1, tjp1] = float32(0.25) * (shared_h_in[tip1 - 1, tjp1] + shared_h_in[tip1 + 1, tjp1] + shared_h_in[tip1 , tjp1 - 1] + shared_h_in[tip1, tjp1 + 1]) # contraction
            if i < m and j < n:
                h_out[i, j] = shared_h_out[tip1, tjp1] 
            cuda.syncthreads()
            shared_d[t] = math.fabs(shared_h_out[tip1, tjp1] - shared_h_in[tip1, tjp1])        
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
            if i < m and j < n:
                h_in[i, j] = shared_h_out[tip1, tjp1]
        if b == 0 and t == 0:
            k[0] += 1
            if d[0] <= eps:
                stop_all[0] = True
            else:
                d[0] = float32(0.0)
        g.sync() 
        if stop_all[0]:
            break

def sfwf_mc_cpu_numpy(heights, i, j, n_samples, seed=None, chunk_size=DEFAULT_MC_CPU_NUMPY_CHUNK_SIZE, verbose=False, verbose_gap=DEFAULT_MC_CPU_NUMPY_VERBOSE_GAP, collect_trajectories=False):
    print(f"SFWF MC CPU NUMPY... [(i, j): {(i, j)}, n_samples: {n_samples:.1e}, chunk_size: {chunk_size}, seed: {seed}]")
    t1 = time.time()
    if seed is None:
        seed = 0
    np.random.seed(seed)      
    m, n = heights.shape    
    actions = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]], dtype=np.int8)    
    Gs = np.empty(n_samples, dtype=np.float32)
    Ts = np.empty(n_samples, dtype=np.int32)
    trajectories = []
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
            print(f"[{k + 1}/{n_samples}: border reached at t: {Ts[k]}, (i, j): {(i_T, j_T)}, h: {Gs[k]}; time: {t2_epi - t1_epi} s]")    
    t2 = time.time()
    h_mean = np.mean(Gs)
    T_mean = np.mean(Ts)
    print(f"SFWF MC CPU NUMPY DONE. [h_mean: {h_mean}, T_mean: {T_mean}; time: {t2 - t1} s]")
    return h_mean, T_mean, trajectories

def sfwf_mc_cuda(heights, i, j, n_samples, rpt, seed=None, tpb=DEFAULT_MC_CUDA_TPB):
    print(f"SFWF MC CUDA... [(i, j): {(i, j)}, wanted n_samples: {n_samples:.1e}, seed: {seed}]")
    t1 = time.time()
    if seed is None:
        seed = 0
    min_n_generators = (n_samples + rpt - 1) // rpt
    bpg_walk = (min_n_generators + tpb - 1) // tpb
    print(f"[min_n_generators: {min_n_generators}, bpg_walk: {bpg_walk}, tpb: {tpb}]")
    print("[initialization of random generators...]")
    t1_generators = time.time()
    dev_random_generators = create_xoroshiro128p_states(bpg_walk * tpb, seed=seed)
    t2_generators = time.time()
    print(f"[initialization of random generators done; count: {bpg_walk * tpb}, memory: {dev_random_generators.nbytes / 1024**2:.3f} MB, time: {t2_generators - t1_generators} s]")        
    dev_h = cuda.to_device(heights)    
    dev_G_means = cuda.device_array(bpg_walk, dtype=np.float32)
    dev_T_means = cuda.device_array(bpg_walk, dtype=np.float32)
    dev_G_mean = cuda.device_array(1, dtype=np.float32)
    dev_T_mean = cuda.device_array(1, dtype=np.float32)
    t1_walk = time.time()
    sfwf_mc_cuda_walk[bpg_walk, tpb](dev_h, i, j, dev_random_generators, rpt, dev_G_means, dev_T_means)
    cuda.synchronize()
    t2_walk = time.time()
    print(f"[random walks done; time: {t2_walk - t1_walk} s]")    
    t1_red = time.time()
    sfwf_mc_cuda_reduce[1, tpb](dev_G_means, dev_T_means, bpg_walk, dev_G_mean, dev_T_mean)    
    h_mean = dev_G_mean.copy_to_host()[0]
    T_mean = dev_T_mean.copy_to_host()[0]
    cuda.synchronize()
    t2_red = time.time()
    print(f"[global reduction done; time: {t2_red - t1_red} s]")
    t2 = time.time()
    print(f"SFWF MC CUDA DONE. [h_mean: {h_mean}, T_mean: {T_mean}, n_samples_defacto: {bpg_walk * tpb * rpt:.1e}; time: {t2 - t1} s, time without generators iniitalization: {t2 - t1 - (t2_generators - t1_generators)} s]")
    return h_mean, T_mean

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