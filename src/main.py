import os
NUMPY_SINGLE_THREAD = True
if NUMPY_SINGLE_THREAD:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   
import numpy as np
import time
from utils import cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str
import sys
from sfwf import (
    sfwf_contraction_cpu_numpy,
    sfwf_contraction_cuda_small,
    sfwf_contraction_cuda_large_atomicmax,
    sfwf_contraction_cuda_large_atomicmax_globalmem,
    sfwf_contraction_cuda_large_gridreducemax,
    sfwf_contraction_cuda_large_gridsync,
    sfwf_mc_cpu_numpy,
    sfwf_mc_cuda
)
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
from sfwf_plots import sfwf_plot, sfwf_plot_large, sfwf_plot_mc_trajectories
from pprint import pprint

# global settings                
FOLDER_EXPERIMENTS = "../experiments/"
FOLDER_EXTRAS = "../extras/"

# experiment settings    
SEED = 7 # seeds nice for plots (and experiments): 6, 7, 15
WF_FOURIER_N = 20
WF_FOURIER_AMPLITUDE = 5.0    
WF_BORDER_N = 317
CONTRACTION_EPS = 1e-4
MC_SEED_CPU_NUMPY = 0
MC_SEED_CUDA = 0
MC_SAMPLES = 10**5 
APPROACH_CONTRACTION_CPU_NUMPY = True
APPROACH_CONTRACTION_CUDA_SMALL = False
APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX = True
APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX_GLOBALMEM = True
APPROACH_CONTRACTION_CUDA_LARGE_GRIDREDUCE = True
APPROACH_CONTRACTION_CUDA_LARGE_GRIDSYNC = False
APPROACH_MC_CPU_NUMPY = False
APPROACH_MC_CUDA = False    
        
# auxiliary settings
VERBOSE_HEIGHTS = True
PLOTS = False
PLOT_MC = False
I0, J0 = 12, 36 # starting point for MC random walks; good for plots: 12, 16 with BORDER_N = 64, SAMPLES_MC = 3 if plot to be generated, MC_SEED_CPU_NUMPY = 0

# wire frame related functions
def fourier_sum(t, a, b):
    ks = np.arange(a.shape[0])
    args = 2 * np.pi * ks * t + b
    return np.sum(a * np.sin(args))

def random_wire_frame(fourier_n, fourier_amplitude, border_n, seed=0):    
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
    return border, heights

# --------------------------------------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":        
    experiment_info = {        
        "SEED": SEED, 
        "WF_FOURIER_N": WF_FOURIER_N,
        "WF_FOURIER_AMPLITUDE": WF_FOURIER_AMPLITUDE,
        "WF_BORDER_N":  WF_BORDER_N,
        "NUMPY_SINGLE_THREAD": NUMPY_SINGLE_THREAD,
        "CONTRACTION_EPS": CONTRACTION_EPS,
        "MC_SEED_CPU_NUMPY": MC_SEED_CPU_NUMPY,
        "MC_SEED_CUDA": MC_SEED_CUDA,
        "MC_SAMPLES": MC_SAMPLES,  
        "APPROACH_CONTRACTION_CPU_NUMPY": APPROACH_CONTRACTION_CPU_NUMPY,
        "APPROACH_CONTRACTION_CUDA_SMALL": APPROACH_CONTRACTION_CUDA_SMALL,
        "APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX": APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX,
        "APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX_GLOBALMEM": APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX_GLOBALMEM,
        "APPROACH_CONTRACTION_CUDA_LARGE_GRIDREDUCE": APPROACH_CONTRACTION_CUDA_LARGE_GRIDREDUCE,
        "APPROACH_CONTRACTION_CUDA_LARGE_GRIDSYNC": APPROACH_CONTRACTION_CUDA_LARGE_GRIDSYNC,
        "APPROACH_MC_CPU_NUMPY": APPROACH_MC_CPU_NUMPY,
        "APPROACH_MC_CUDA": APPROACH_MC_CUDA                    
        }        
    c_props = cpu_and_system_props()
    g_props = gpu_props()
    experiment_hs = experiment_hash_str(experiment_info, c_props, g_props)
    
    heights_out = None
    heights_out_ref = None
    time_ref = None            

    logger = Logger(f"{FOLDER_EXPERIMENTS}{experiment_hs}.log")    
    sys.stdout = logger

    t1_main = time.time()
    print("SOAP FILM IN A WIRE FRAME...")
    
    line_separator = 196 * "="   
    print(f"HASH STRING: {experiment_hs}")
    print(line_separator)
    print(f"EXPERIMENT INFO:\n{dict_to_str(experiment_info)}")
    print(line_separator)      
    print("CPU AND SYSTEM:")
    pprint(c_props)
    print("GPU:")
    pprint(g_props)
    print(line_separator)

    border, heights_in = random_wire_frame(WF_FOURIER_N, WF_FOURIER_AMPLITUDE, WF_BORDER_N, seed=SEED)
    print(f"RANDOM WIRE FRAME WITH TOTAL OF POINTS (STATES): {WF_BORDER_N**2}")    
    if PLOTS:
        sfwf_plot(border, heights_in, "WIRE FRAME (INPUT)")
    
    if APPROACH_CONTRACTION_CPU_NUMPY:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cpu_numpy(heights_in, CONTRACTION_EPS, verbose=False)
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_
        if VERBOSE_HEIGHTS:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if PLOTS: 
            method_name = sfwf_contraction_cpu_numpy.__name__
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"            
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)
                
    if APPROACH_CONTRACTION_CUDA_SMALL and BORDER_N <= DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_small(heights_in, CONTRACTION_EPS)
        if VERBOSE_HEIGHTS:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_
        if PLOTS:
            method_name = sfwf_contraction_cuda_small.__name__ 
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"            
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)

    if APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_atomicmax(heights_in, CONTRACTION_EPS)
        if VERBOSE_HEIGHTS:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if PLOTS:
            method_name = sfwf_contraction_cuda_large_atomicmax.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)
    
    if APPROACH_CONTRACTION_CUDA_LARGE_ATOMICMAX_GLOBALMEM:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_atomicmax_globalmem(heights_in, CONTRACTION_EPS) 
        if VERBOSE_HEIGHTS:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if PLOTS:
            method_name = sfwf_contraction_cuda_large_atomicmax_globalmem.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)

    if APPROACH_CONTRACTION_CUDA_LARGE_GRIDREDUCE:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_gridreducemax(heights_in, CONTRACTION_EPS)
        if VERBOSE_HEIGHTS:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")            
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if PLOTS:
            method_name = sfwf_contraction_cuda_large_gridreducemax.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)  

    if APPROACH_CONTRACTION_CUDA_LARGE_GRIDSYNC:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_gridsync(heights_in, CONTRACTION_EPS)
        if VERBOSE_HEIGHTS:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if PLOTS: 
            method_name = sfwf_contraction_cuda_large_gridsync.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}, time: {time_:.3f} s]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)

    if APPROACH_MC_CPU_NUMPY:
        print("---") 
        MC_SAMPLES = 3 # changing to small sample size for plot purposes
        h_mean, T_mean, trajectories = sfwf_mc_cpu_numpy(heights_in, I0, J0, MC_SAMPLES, seed=MC_CPU_NUMPY_SEED, verbose=True, collect_trajectories=True)
        print(f"SINGLE HEIGHT COMPARISON: {heights_out_ref[i0, j0]=} vs {h_mean=}, ABS DIFF: {np.abs(h_mean - heights_out_ref[i0, j0]):.3e}]")
        if PLOT_MC:
            sfwf_plot_mc_trajectories(heights_in, trajectories, h_mean)
        
    if APPROACH_MC_CUDA:
        print("---")
        rpt = DEFAULT_MC_CUDA_RPT
        h_mean, T_mean = sfwf_mc_cuda(heights_in, I0, J0, MC_SAMPLES, rpt, seed=SEED)
        print(f"SINGLE HEIGHT COMPARISON: {heights_out_ref[i0, j0]=} vs {h_mean=}, ABS DIFF: {np.abs(h_mean - heights_out_ref[i0, j0]):.3e}]")
        
    t2_main = time.time()
    print(f"SOAP FILM IN A WIRE FRAME DONE [time: {t2_main - t1_main}].")
    sys.stdout = sys.__stdout__
    logger.logfile.close()    