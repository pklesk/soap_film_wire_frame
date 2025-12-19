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

# global settings                
FOLDER_EXPERIMENTS = "../experiments/"

if __name__ == "__main__":
    print("SOAP FILM IN WIRE FRAME (CONTRACTION ITERATION)...")
        
    seed = 7 # seeds nice for plots (and experiments): 6, 7, 15  
    verbose_system_props = True
    fourier_n = 20
    fourier_amplitude = 5.0
    eps = 1e-4
    border_n = 1000
    plots = False
    plot_mc = False
    verbose_heights = True
    samples_mc = 3 # 3 if plot to be generated else 10**5 or more
    i0, j0 = 12, 36 # starting point for MC random walks; good for plots: 12, 16 with border_n = 64, samples_mc = 3 if plot to be generated
    
    heights_out = None
    heights_out_ref = None
    time_ref = None
    test_contraction_cpu_numpy = False
    test_contraction_cuda_small = False
    test_contraction_cuda_large_atomicmax = True
    test_contraction_cuda_large_atomicmax_globalmem = True
    test_contraction_cuda_large_gridreducemax = True
    test_contraction_cuda_large_gridsync = True
    test_mc_cpu_numpy = False
    test_mc_cuda = False
           
    if verbose_system_props:
        print("CPU AND SYSTEM:")
        pprint(cpu_and_system_props())
        print("GPU:")
        pprint(gpu_props())

    border, heights_in = random_wire_frame(fourier_n, fourier_amplitude, border_n, seed=seed)
    print(f"TOTAL OF POINTS (STATES): {border_n**2}")    
    if plots:
        sfwf_plot(border, heights_in, "WIRE FRAME (INPUT)")
    
    if test_contraction_cpu_numpy:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cpu_numpy(heights_in, eps, verbose=False)
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_
        if verbose_heights:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if plots: 
            method_name = sfwf_contraction_cpu_numpy.__name__
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"            
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)
                
    if test_contraction_cuda_small and border_n <= DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_small(heights_in, eps)
        if verbose_heights:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_
        if plots:
            method_name = sfwf_contraction_cuda_small.__name__ 
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"            
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)

    if test_contraction_cuda_large_atomicmax:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_atomicmax(heights_in, eps)
        if verbose_heights:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if plots:
            method_name = sfwf_contraction_cuda_large_atomicmax.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)
    
    if test_contraction_cuda_large_atomicmax_globalmem:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_atomicmax_globalmem(heights_in, eps) 
        if verbose_heights:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if plots:
            method_name = sfwf_contraction_cuda_large_atomicmax_globalmem.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)

    if test_contraction_cuda_large_gridreducemax:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_gridreducemax(heights_in, eps)
        if verbose_heights:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")            
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if plots:
            method_name = sfwf_contraction_cuda_large_gridreducemax.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)  

    if test_contraction_cuda_large_gridsync:
        print("---")
        heights_out, d, k, time_ = sfwf_contraction_cuda_large_gridsync(heights_in, eps)
        if verbose_heights:
            print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
        if heights_out_ref is not None:
            d_vs_ref = np.max(np.abs(heights_out - heights_out_ref))
            print(f"D_INF VS HEIGHTS REF: {d_vs_ref}")
            print(f"SPEEDUP VS TIME REF: {time_ref / time_}")
        else:
            heights_out_ref = heights_out  
            time_ref = time_            
        if plots: 
            method_name = sfwf_contraction_cuda_large_gridsync.__name__  
            title = f"SHAPE COMPUTED BY: {method_name}"
            subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}, time: {time_:.3f} s]"
            sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)

    if test_mc_cpu_numpy:
        print("---") 
        mc_cpu_numpy_seed = 0 # good for plots: mc_cpu_numpy_seed = 0, (i0, j0) = (12, 16), with border_n = 64, samples_mc = 3
        h_mean, T_mean, trajectories = sfwf_mc_cpu_numpy(heights_in, i0, j0, samples_mc, seed=mc_cpu_numpy_seed, verbose=True, collect_trajectories=True)
        print(f"SINGLE HEIGHT COMPARISON: {heights_out_ref[i0, j0]=} vs {h_mean=}, ABS DIFF: {np.abs(h_mean - heights_out_ref[i0, j0]):.3e}]")
        if plot_mc:
            sfwf_plot_mc_trajectories(heights_in, trajectories, h_mean)
        
    if test_mc_cuda:
        print("---")
        rpt = DEFAULT_MC_CUDA_RPT
        h_mean, T_mean = sfwf_mc_cuda(heights_in, i0, j0, samples_mc, rpt, seed=seed)
        print(f"SINGLE HEIGHT COMPARISON: {heights_out_ref[i0, j0]=} vs {h_mean=}, ABS DIFF: {np.abs(h_mean - heights_out_ref[i0, j0]):.3e}]")
    
    
    print("SOAP FILM IN WIRE FRAME (CONTRACTION ITERATION) DONE.")