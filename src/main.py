__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

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
import sfwf
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
from sfwf_plots import sfwf_plot, sfwf_plot_large, sfwf_plot_mc_trajectories
from pprint import pprint
c_props = cpu_and_system_props()
g_props = gpu_props()

# global settings                
FOLDER_EXPERIMENTS = "../experiments/"
DEFAULT_REPETITIONS = 1

# experiment settings    
SEED = 7 # some seeds nice for plots: {6, 7, 15} with WF_FOURIER_N: 20, WF_FOURIER_AMPLITUDE: 5.0  
WF_FOURIER_N = 20
WF_FOURIER_AMPLITUDE = 5.0    
WF_BORDER_N = 1000
CONTRACTION_EPS = 1e-4
CONTRACTION_PLOTS = False
MC_SEED = 0
MC_SAMPLES = 10**9
MC_I0_J0 = (12, 36) # starting point for MC random walks; good for plots: 12, 16 with BORDER_N = 64, SAMPLES_MC = 3 if plot to be generated, MC_SEED = 0
MC_EXAMPLE_PLOT = False
MC_EXAMPLE_PLOT_SAMPLES = 3 
APPROACHES_CONTRACTION = { # approaches for contraction iteration
    sfwf.sfwf_contraction_cpu_numpy.__name__: (False, sfwf.sfwf_contraction_cpu_numpy, DEFAULT_REPETITIONS, {}),    
    sfwf.sfwf_contraction_cpu_numba_parallel.__name__: (False, sfwf.sfwf_contraction_cpu_numba_parallel, DEFAULT_REPETITIONS, {}),
    sfwf.sfwf_contraction_cuda_small.__name__: (False, sfwf.sfwf_contraction_cuda_small, DEFAULT_REPETITIONS, {"tpb": sfwf.DEFAULT_TPB}),
    sfwf.sfwf_contraction_cuda_large_atomicmax.__name__: (False, sfwf.sfwf_contraction_cuda_large_atomicmax, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE}),    
    sfwf.sfwf_contraction_cuda_large_atomicmaxglosten.__name__: (True, sfwf.sfwf_contraction_cuda_large_atomicmaxglosten, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE}),
    sfwf.sfwf_contraction_cuda_large_hreducemax.__name__: (False, sfwf.sfwf_contraction_cuda_large_hreducemax, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE, "tpb_reduce": sfwf.DEFAULT_TPB}),    
    sfwf.sfwf_contraction_cuda_large_hreducemaxgs.__name__: (False, sfwf.sfwf_contraction_cuda_large_hreducemaxgs, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE, "tpb_reduce": sfwf.DEFAULT_TPB, "cores": g_props["cores_total"]}),    
    sfwf.sfwf_contraction_cuda_large_gridsync.__name__: (False, sfwf.sfwf_contraction_cuda_large_gridsync, DEFAULT_REPETITIONS, {"tpb_side": sfwf.DEFAULT_TPB_SIDE})    
    }
APPROACHES_MC = { # approaches for Monte Carlo simulations
    sfwf.sfwf_mc_cpu_numpy.__name__: (False, sfwf.sfwf_mc_cpu_numpy, DEFAULT_REPETITIONS, {"chunk_size": sfwf.DEFAULT_MC_CPU_NUMPY_CHUNK_SIZE}),
    sfwf.sfwf_mc_cuda.__name__: (False, sfwf.sfwf_mc_cuda, DEFAULT_REPETITIONS, {"rpt": sfwf.DEFAULT_MC_CUDA_RPT, "tpb": sfwf.DEFAULT_MC_CUDA_TPB})
    }            

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

def approaches_info(approaches):
    info = {}
    for key in approaches.keys():
        if approaches[key][0]:
            info[key]  = (approaches[key][0], approaches[key][1].__name__, approaches[key][2], approaches[key][3])
        else:
            info[key] = (approaches[key][0], approaches[key][1].__name__, 0, {})
    return info        

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
        "CONTRACTION_PLOTS": CONTRACTION_PLOTS,
        "MC_SEED": MC_SEED,
        "MC_SAMPLES": MC_SAMPLES, 
        "MC_I0_J0": MC_I0_J0, 
        "MC_EXAMPLE_PLOT": MC_EXAMPLE_PLOT,
        "MC_EXAMPLE_PLOT_SAMPLES": MC_EXAMPLE_PLOT_SAMPLES,
        **approaches_info(APPROACHES_CONTRACTION),
        **approaches_info(APPROACHES_MC)                    
        }                    
    experiment_hs = experiment_hash_str(experiment_info, c_props, g_props)                
    
    logger = Logger(f"{FOLDER_EXPERIMENTS}{experiment_hs}.log")    
    sys.stdout = logger

    t1_main = time.time()
    print("SOAP FILM IN A WIRE FRAME...")    
    line_separator = 244 * "="   
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
    if heights_in.shape[0] >= 5 and heights_in.shape[1] >= 5:
        print(f"HEIGHTS IN[:5, :5]:\n{heights_in[:5, :5]}")        
    if CONTRACTION_PLOTS:
        sfwf_plot(border, heights_in, "WIRE FRAME (INPUT)")
    
    # about to execute contraction iteration approaches
    contraction_ref_approach_name = None
    contraction_ref_heights_out = None
    contraction_ref_time_mean = None
    contraction_times = {}
    contraction_ds = {}
    contraction_ks = {}    
    for index, (approach_name, (approach_on, approach_function, approach_repetitions, approach_extra_params)) in enumerate(APPROACHES_CONTRACTION.items()):
        if approach_on:
            print(line_separator)
            print(f"CONTRACTION ITERATION APPROACH {index + 1}: {approach_name}...", flush=True)
            if approach_name == sfwf.sfwf_contraction_cuda_small.__name__ and WF_BORDER_N > sfwf.DEFAULT_CONTRACTION_CUDA_SMALL_SHARED_SIDE:
                print("[skipping this approach (too large WF_BORDER_N)]")
                continue                 
            for r in range(approach_repetitions):
                print("---")               
                print(f"REPETITION: {r + 1}/{approach_repetitions}:")
                heights_out, d, k, time_ = approach_function(heights_in, eps=CONTRACTION_EPS, **approach_extra_params)
                if approach_name not in contraction_times:                    
                    contraction_times[approach_name] = []
                    contraction_ds[approach_name] = []
                    contraction_ks[approach_name] = []
                contraction_times[approach_name].append(time_)
                contraction_ds[approach_name].append(d)
                contraction_ks[approach_name].append(k)            
            time_mean = np.mean(contraction_times[approach_name])
            time_std = np.std(contraction_times[approach_name])
            d_mean = np.mean(contraction_ds[approach_name])
            k_mean = np.mean(contraction_ks[approach_name])
            if contraction_ref_approach_name is None:
                contraction_ref_approach_name = approach_name
                contraction_ref_heights_out = heights_out
                contraction_ref_time_mean = time_mean                            
            d_vs_ref = np.max(np.abs(heights_out - contraction_ref_heights_out))
            speedup_vs_ref = contraction_ref_time_mean / time_mean
            print("***")
            print("SUMMARY:")
            if heights_out.shape[0] >= 5 and heights_out.shape[1] >= 5:
                print(f"HEIGHTS OUT[:5, :5]:\n{heights_out[:5, :5]}")
            print(f"D_INF OF HEIGHTS VS REF: {str(d_vs_ref)}")
            print(f"D_INF (AT STOP) MEAN: {d_mean}")
            print(f"ITERATIONS MEAN: {k_mean}")                    
            print(f"TIME MEAN: {time_mean} s, STD: {time_std} s, STD_%: {(time_std / time_mean) * 100:.1f}%")                        
            print(f"SPEEDUP VS REF: {speedup_vs_ref}")            
            if CONTRACTION_PLOTS: 
                method_name = approach_function.__name__
                title = f"SHAPE COMPUTED BY: {method_name}"
                subtitle = f"[$d_{{\\infty}}$: {d:.3e}, iterations: {k}]"            
                sfwf_plot_large(border, heights_in, "WIRE FRAME (INPUT)", "", heights_out, title, subtitle)                                                                                                     
    
    # about to execute Monte Carlo approaches
    mc_ref_approach_name = None
    mc_ref_h_mean = None
    mc_ref_time_mean = None
    mc_times = {}    
    for index, (approach_name, (approach_on, approach_function, approach_repetitions, approach_extra_params)) in enumerate(APPROACHES_MC.items()):
        if approach_on:
            print(line_separator)
            print(f"MONTE CARLO APPROACH {index + 1}: {approach_name}...", flush=True)
            if MC_EXAMPLE_PLOT and approach_name == sfwf.sfwf_mc_cpu_numpy.__name__:
                approach_extra_params_plot = approach_extra_params.copy()
                approach_extra_params_plot["collect_trajectories"] = True
                print(f"[making additional example run and plot for only {MC_EXAMPLE_PLOT_SAMPLES} trajectories]")
                h_mean, T_mean, time_, trajectories = approach_function(heights_in, i=MC_I0_J0[0], j=MC_I0_J0[1], n_samples=MC_EXAMPLE_PLOT_SAMPLES, seed=MC_SEED, **approach_extra_params_plot)                                
                sfwf_plot_mc_trajectories(heights_in, trajectories, h_mean)                        
            for r in range(approach_repetitions):
                print("---")               
                print(f"REPETITION: {r + 1}/{approach_repetitions}:")
                h_mean, T_mean, time_, trajectories = approach_function(heights_in, i=MC_I0_J0[0], j=MC_I0_J0[1], n_samples=MC_SAMPLES, seed=MC_SEED, **approach_extra_params)
                if approach_name not in mc_times:
                    mc_times[approach_name] = []
                mc_times[approach_name].append(time_)                        
            time_mean = np.mean(mc_times[approach_name])
            time_std = np.std(mc_times[approach_name])            
            if mc_ref_approach_name is None:
                mc_ref_approach_name = approach_name
                mc_ref_h_mean = h_mean
                mc_ref_time_mean = time_mean                                                            
            time_mean = np.mean(mc_times[approach_name])
            time_std = np.std(mc_times[approach_name])   
            d_vs_ref = np.abs(h_mean - mc_ref_h_mean)
            speedup_vs_ref = mc_ref_time_mean / time_mean
            i0, j0 = MC_I0_J0
            print("***")
            print("SUMMARY:")
            if contraction_ref_heights_out is not None:
                print(f"COMPARISON OF SINGLE HEIGHT VS REF -> h_mean: {str(h_mean)} VS contraction_ref_heights_out[i0, j0]: {str(contraction_ref_heights_out[i0, j0])}, ABS DIFF: {np.abs(h_mean - contraction_ref_heights_out[i0, j0]):.7e}]")            
            print(f"TIME MEAN: {time_mean} s, STD: {time_std} s, STD_%: {(time_std / time_mean) * 100:.1f}%")                        
            print(f"SPEEDUP VS REF: {speedup_vs_ref}")                                                                
    
    print(line_separator)
    print(line_separator)
    print("FINAL SUMMARY:")
    for index, (approach_name, (approach_on, approach_function, approach_repetitions, approach_extra_params)) in enumerate(APPROACHES_CONTRACTION.items()):
        if approach_on:
            if approach_name not in contraction_times:
                print(f"CONTRACTION ITERATION APPROACH {index + 1}: {approach_name} SKIPPED.")
                continue
            reference_info = " (REFERENCE)" if approach_name == contraction_ref_approach_name else ""
            k_mean = np.mean(contraction_ks[approach_name])
            d_mean = np.mean(contraction_ds[approach_name])
            time_mean = np.mean(contraction_times[approach_name])
            time_std = np.std(contraction_times[approach_name])
            speedup = contraction_ref_time_mean / time_mean 
            print(f"CONTRACTION ITERATION APPROACH {index + 1}: {approach_name}{reference_info} -> MEAN ITERATIONS: {k_mean}, MEAN D_INF: {d_mean}, MEAN TIME: {time_mean} s, TIME STD: {time_std} s, STD_%: {(time_std / time_mean) * 100:.1f}%, SPEED-UP: {speedup:.2f}", flush=True)
        else:
            print(f"CONTRACTION ITERATION APPROACH {index + 1}: {approach_name} OFF.")
    for index, (approach_name, (approach_on, approach_function, approach_repetitions, approach_extra_params)) in enumerate(APPROACHES_MC.items()):
        if approach_on:
            reference_info = " (REFERENCE)" if approach_name == mc_ref_approach_name else ""
            time_mean = np.mean(mc_times[approach_name])
            time_std = np.std(mc_times[approach_name])
            speedup = mc_ref_time_mean / time_mean 
            print(f"MONTE CARLO APPROACH {index + 1}: {approach_name}{reference_info} -> MEAN TIME: {time_mean} s, TIME STD: {time_std} s,  STD_%: {(time_std / time_mean) * 100:.1f}%, SPEED-UP: {speedup:.2f}", flush=True)
        else:
            print(f"MONTE CARLO APPROACH {index + 1}: {approach_name} OFF.")
    
    t2_main = time.time()    
    print(f"SOAP FILM IN A WIRE FRAME DONE. [hash string: {experiment_hs}, time: {t2_main - t1_main}]")
    sys.stdout = sys.__stdout__
    logger.logfile.close()
