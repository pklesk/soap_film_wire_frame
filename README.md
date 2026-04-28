# CUDA contraction iteration for soap film in a wire frame
The repository constitutes a part of research on CUDA computational approaches for algorithms based on contraction mapping theorem due to Banach, a.k.a. the fixed-point theorem.

Imagine a wire frame of irregular shape, forming a closed loop, and a soap film surface spanned within it. The task is to compute the shape of that surface knowing the frame shape.
Solution of this problem can be found numerically by means of stencil computations carried out as a *contraction iteration* procedure.
The repository contains six CUDA implementations of such procedures and two referential CPU-based implementations.

<img src="extras/sfwf_contraction.png"/>

Implementations of CUDA kernels have been carried out using Numba - a just-in-time compiler for Python. Numba exposes a programming interface closely mirroring the native CUDA C++ API, and translates kernel functions into its internal representation (Numba IR), which is then lowered via the LLVM and NVVM-based pipeline into PTX and finally JIT-compiled into executable machine code.


# Speed-ups
<img src="extras/sfwf_speedups.png"/>


# Selected experimental results (averages over 10 repetitions)
| no. | approach (design)                               | iters.       | d_inf <br/> (eps: 10<sup>-4</sup>)   | mean time [s] | speed-up     | mean time [s] | speed-up     |
|:----|:------------------------------------------------|:------------:|:------------------------------------:|--------------:|-------------:|--------------:|-------------:|
||||| **RTX 3090**<br/>**(i7-10700 8c/16t)**| **RTX 3090**<br/>**(i7-10700 8c/16t)**| **RTX 5090**<br/>**(Ryzen 9 9950X 16c/32t)**| **RTX 5090**<br/>**(Ryzen 9 9950X 16c/32t)**|
|     | **problem size: ~10<sup>3</sup>** <br/> (domain grid: 32 $\times$ 32, points: 1_024) |||||||
| 1   | `sfwf_contraction`<br/>`_cpu_numpy`             | 1_270        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.017         | $\times$ 1.0 | 0.007         | $\times$ 2.4 |
| 2   | `sfwf_contraction`<br/>`_cpu_numba_parallel`    | 1_270        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.025         | $\times$ 0.7 | 0.020         | $\times$ 0.9 |
| 3   | `sfwf_contraction`<br/>`_cuda_small`            | 1_270        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.006         | $\times$ 2.8 | 0.004         | $\times$ 4.3 |
| 4   | `sfwf_contraction`<br/>`_cuda_large_atomicmax`  | 1_300        | 8.7 $\cdot$ 10<sup>-5</sup>          | 0.072         | $\times$ 0.2 | 0.025         | $\times$ 0.7 |
| 5   | `sfwf_contraction`<br/>`_cuda_large_atomicmaxglosten`| 1_300   | 8.7 $\cdot$ 10<sup>-5</sup>          | 0.072         | $\times$ 0.2 | 0.025         | $\times$ 0.7 |
| 6   | `sfwf_contraction`<br/>`_cuda_large_hreducemax` | 1_300        | 8.7 $\cdot$ 10<sup>-5</sup>          | 0.044         | $\times$ 0.4 | 0.016         | $\times$ 1.1 |
| 7   | `sfwf_contraction`<br/>`_cuda_large_hreducemaxgs`| 1_300       | 8.7 $\cdot$ 10<sup>-5</sup>          | 0.045         | $\times$ 0.4 | 0.016         | $\times$ 1.1 |
| 8   | `sfwf_contraction`<br/>`_cuda_large_gridsync`   | 1_270        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.008         | $\times$ 2.1 | 0.006         | $\times$ 2.8 |
|     | **problem size: ~10<sup>4</sup>** <br/> (domain grid: 100 $\times$ 100, points: 10_000) |||||||
| 9   | `sfwf_contraction`<br/>`_cpu_numpy`             | 8_365        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.262         | $\times$ 1.0 | 0.097         | $\times$ 2.7 |
| 10  | `sfwf_contraction`<br/>`_cpu_numba_parallel`    | 8_365        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.365         | $\times$ 0.7 | 0.163         | $\times$ 1.6 |
| 11  | `sfwf_contraction`<br/>`_cuda_large_atomicmax`  | 8_400        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.473         | $\times$ 0.6 | 0.163         | $\times$ 1.6 |
| 12  | `sfwf_contraction`<br/>`_cuda_large_atomicmaxglosten`| 8_400   | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.468         | $\times$ 0.6 | 0.163         | $\times$ 1.6 |
| 13  | `sfwf_contraction`<br/>`_cuda_large_hreducemax` | 8_400        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.283         | $\times$ 0.9 | 0.101         | $\times$ 2.6 |
| 14  | `sfwf_contraction`<br/>`_cuda_large_hreducemaxgs`| 8_400       | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.284         | $\times$ 0.9 | 0.102         | $\times$ 2.6 |
| 15  | `sfwf_contraction`<br/>`_cuda_large_gridsync`   | 8_365        | 9.9 $\cdot$ 10<sup>-5</sup>          | 0.052         | $\times$ 5.0 | 0.041         | $\times$ 6.4 |
|     | **problem size: ~10<sup>5</sup>** <br/> (domain grid: 317 $\times$ 317, points: 100_489) |||||||
| 16  | `sfwf_contraction`<br/>`_cpu_numpy`             | 38_235       | 10.0 $\cdot$ 10<sup>-5</sup>         | 7.054         | $\times$ 1.0 | 2.888         | $\times$ 2.4 |
| 17  | `sfwf_contraction`<br/>`_cpu_numba_parallel`    | 38_235       | 10.0 $\cdot$ 10<sup>-5</sup>         | 2.664         | $\times$ 2.6 | 1.424         | $\times$ 5.0 |
| 18  | `sfwf_contraction`<br/>`_cuda_large_atomicmax`  | 38_300       | 9.9 $\cdot$ 10<sup>-5</sup>          | 2.055         | $\times$ 3.4 | 0.737         | $\times$ 9.6 |
| 19  | `sfwf_contraction`<br/>`_cuda_large_atomicmaxglosten`| 38_300  | 9.9 $\cdot$ 10<sup>-5</sup>          | 2.065         | $\times$ 3.4 | 0.741         | $\times$ 9.5 |
| 20  | `sfwf_contraction`<br/>`_cuda_large_hreducemax` | 38_300       | 9.9 $\cdot$ 10<sup>-5</sup>          | 1.235         | $\times$ 5.7 | 0.460         | $\times$ 15.3|
| 21  | `sfwf_contraction`<br/>`_cuda_large_hreducemaxgs`| 38_300      | 9.9 $\cdot$ 10<sup>-5</sup>          | 1.244         | $\times$ 5.7 | 0.463         | $\times$ 15.2|
| 22  | `sfwf_contraction`<br/>`_cuda_large_gridsync`   | 38_235       | 10.0 $\cdot$ 10<sup>-5</sup>         | 0.786         | $\times$ 9.0 | 0.334         | $\times$ 21.1|
|     | **problem size: ~10<sup>6</sup>** <br/> (domain grid: 1_000 $\times$ 1_000, points: 1_000_000) |||||||
| 23  | `sfwf_contraction`<br/>`_cpu_numpy`             | 37_329       | 10.0 $\cdot$ 10<sup>-5</sup>         | 71.297        | $\times$ 1.0 | 26.448        | $\times$ 2.7 |
| 24  | `sfwf_contraction`<br/>`_cpu_numba_parallel`    | 37_329       | 10.0 $\cdot$ 10<sup>-5</sup>         | 10.428        | $\times$ 6.8 | 4.847         | $\times$ 14.7|
| 25  | `sfwf_contraction`<br/>`_cuda_large_atomicmax`  | 37_400       | 9.9 $\cdot$ 10<sup>-5</sup>          | 2.076         | $\times$ 34.3| 0.738         | $\times$ 96.6|
| 26  | `sfwf_contraction`<br/>`_cuda_large_atomicmaxglosten`| 37_400  | 9.9 $\cdot$ 10<sup>-5</sup>          | 2.076         | $\times$ 34.3| 0.736         | $\times$ 96.9|
| 27  | `sfwf_contraction`<br/>`_cuda_large_hreducemax` | 37_400       | 9.9 $\cdot$ 10<sup>-5</sup>          | 1.251         | $\times$ 57.0| 0.460         | $\times$ 155.0|
| 28  | `sfwf_contraction`<br/>`_cuda_large_hreducemaxgs`| 37_400      | 9.9 $\cdot$ 10<sup>-5</sup>          | 1.250         | $\times$ 57.0| 0.459         | $\times$ 155.3|
| 29  | `sfwf_contraction`<br/>`_cuda_large_gridsync`   | 37_329       | 10.0 $\cdot$ 10<sup>-5</sup>         | 4.178         | $\times$ 17.1| 0.988         | $\times$ 72.2|
|     | **problem size: ~10<sup>7</sup>** <br/> (domain grid: 3_163 $\times$ 3_163, points: 10_004_569) |||||||
| 30  | `sfwf_contraction`<br/>`_cpu_numpy`             | 71_513       | 9.9 $\cdot$ 10<sup>-5</sup>          | 2_712.149     | $\times$ 1.0 | 1_770.113     | $\times$ 1.5 |
| 31  | `sfwf_contraction`<br/>`_cpu_numba_parallel`    | 71_513       | 9.9 $\cdot$ 10<sup>-5</sup>          | 400.334       | $\times$ 6.8 | 146.057       | $\times$ 18.6|
| 32  | `sfwf_contraction`<br/>`_cuda_large_atomicmax`  | 71_600       | 9.9 $\cdot$ 10<sup>-5</sup>          | 20.209        | $\times$ 134.2| 5.015        | $\times$ 540.8|
| 33  | `sfwf_contraction`<br/>`_cuda_large_atomicmaxglosten`| 71_600  | 9.9 $\cdot$ 10<sup>-5</sup>          | 15.325        | $\times$ 177.0| 4.066        | $\times$ 667.0|
| 34  | `sfwf_contraction`<br/>`_cuda_large_hreducemax` | 71_600       | 9.9 $\cdot$ 10<sup>-5</sup>          | 19.960        | $\times$ 135.9| 4.861        | $\times$ 557.9|
| 35  | `sfwf_contraction`<br/>`_cuda_large_hreducemaxgs`| 71_600      | 9.9 $\cdot$ 10<sup>-5</sup>          | 19.935        | $\times$ 136.0| 4.855        | $\times$ 558.6|
| 36  | `sfwf_contraction`<br/>`_cuda_large_gridsync`   | 71_513       | 10.0 $\cdot$ 10<sup>-5</sup>         | 75.746        | $\times$ 35.8 | 12.724       | $\times$ 213.2|


# Basic usage for default settings
By launching `python main.py` or `python3 main.py`, one executes 10 repetitions of contraction iteration procedures using all CUDA approaches
(and one CPU-based appraoch `_cpu_numba_parallel`) for the program's default settings of `WF_BORDER_N = 1000` - a domain with 10<sup>6</sup> points - and 
`CONTRACTION_EPS = 1e-4` (approach `_cpu_numpy` is turned off by default for shorter execution time). 
This produces the following print-out (abbreviated here):
```bash
SOAP FILM IN A WIRE FRAME...
HASH STRING: 3717747666_44851_901_[7;20;5.0;1000;True;1.0e-04;False;0;1e+09;(12,36);False;3;FTTTTTTT;FF]
=======================================================================================================================================
EXPERIMENT INFO:
{
  SEED: 7,
  WF_FOURIER_N: 20,
  WF_FOURIER_AMPLITUDE: 5.0,
  WF_BORDER_N: 1000,
  NUMPY_SINGLE_THREAD: True,
  CONTRACTION_EPS: 0.0001,
  CONTRACTION_PLOTS: False,
  MC_SEED: 0,
  MC_SAMPLES: 1000000000,
  MC_I0_J0: (12, 36),
  MC_EXAMPLE_PLOT: False,
  MC_EXAMPLE_PLOT_SAMPLES: 3,
  sfwf_contraction_cpu_numpy: (False, 'sfwf_contraction_cpu_numpy', 0, {}),
  sfwf_contraction_cpu_numba_parallel: (True, 'sfwf_contraction_cpu_numba_parallel', 10, {}),
  sfwf_contraction_cuda_small: (True, 'sfwf_contraction_cuda_small', 10, {'tpb': 512}),
  sfwf_contraction_cuda_large_atomicmax: (True, 'sfwf_contraction_cuda_large_atomicmax', 10, {'lazy_stop_check': 100, 'tpb_side': 16}),
  sfwf_contraction_cuda_large_atomicmaxglosten: (True, 'sfwf_contraction_cuda_large_atomicmaxglosten', 10, {'lazy_stop_check': 100, 'tpb_side': 16}),
  sfwf_contraction_cuda_large_hreducemax: (True, 'sfwf_contraction_cuda_large_hreducemax', 10, {'lazy_stop_check': 100, 'tpb_side': 16, 'tpb_reduce': 512}),
  sfwf_contraction_cuda_large_hreducemaxgs: (True, 'sfwf_contraction_cuda_large_hreducemaxgs', 10, {'lazy_stop_check': 100, 'tpb_side': 16, 'tpb_reduce': 512, 'cores': 21760}),
  sfwf_contraction_cuda_large_gridsync: (True, 'sfwf_contraction_cuda_large_gridsync', 10, {'tpb_side': 16}),
  sfwf_mc_cpu_numpy: (False, 'sfwf_mc_cpu_numpy', 0, {}),
  sfwf_mc_cuda: (False, 'sfwf_mc_cuda', 0, {})
}
... [CPU and GPU properties displayed]
... [log of computations]
=======================================================================================================================================
FINAL SUMMARY:
FINAL SUMMARY:
CONTRACTION ITERATION APPROACH 1: sfwf_contraction_cpu_numpy OFF.
CONTRACTION ITERATION APPROACH 2: sfwf_contraction_cpu_numba_parallel (REFERENCE) -> MEAN ITERATIONS: 37329.0, MEAN D_INF: 9.965896606445312e-05, MEAN TIME: 5.125496220588684 s, TIME STD: 0.6529061449150861 s, STD_%: 12.7%, SPEED-UP: 1.00
CONTRACTION ITERATION APPROACH 3: sfwf_contraction_cuda_small SKIPPED.
CONTRACTION ITERATION APPROACH 4: sfwf_contraction_cuda_large_atomicmax -> MEAN ITERATIONS: 37400.0, MEAN D_INF: 9.918212890625e-05, MEAN TIME: 0.721886682510376 s, TIME STD: 0.0068925294070995725 s, STD_%: 1.0%, SPEED-UP: 7.10
CONTRACTION ITERATION APPROACH 5: sfwf_contraction_cuda_large_atomicmaxglosten -> MEAN ITERATIONS: 37400.0, MEAN D_INF: 9.918212890625e-05, MEAN TIME: 0.7257266044616699 s, TIME STD: 0.0030493741505214548 s, STD_%: 0.4%, SPEED-UP: 7.06
CONTRACTION ITERATION APPROACH 6: sfwf_contraction_cuda_large_hreducemax -> MEAN ITERATIONS: 37400.0, MEAN D_INF: 9.918212890625e-05, MEAN TIME: 0.44861438274383547 s, TIME STD: 0.0004875580366133472 s, STD_%: 0.1%, SPEED-UP: 11.43
CONTRACTION ITERATION APPROACH 7: sfwf_contraction_cuda_large_hreducemaxgs -> MEAN ITERATIONS: 37400.0, MEAN D_INF: 9.918212890625e-05, MEAN TIME: 0.45176780223846436 s, TIME STD: 0.0008764222166888117 s, STD_%: 0.2%, SPEED-UP: 11.35
```


# Configuration and other settings
For other executions, one can re-edit the settings in `src/main.py`:
```python
# experiment settings    
SEED = 7 # some seeds nice for plots: {6, 7, 15} with WF_FOURIER_N: 20, WF_FOURIER_AMPLITUDE: 5.0  
WF_FOURIER_N = 20
WF_FOURIER_AMPLITUDE = 5.0    
WF_BORDER_N = 1000
CONTRACTION_EPS = 1e-4
CONTRACTION_PLOTS = False
```

The dictionary `APPROACHES_CONTRACTION` (see below) allows to turn on/off particular approaches and to modify their default parameters.
```python
APPROACHES_CONTRACTION = { # approaches for contraction iteration
    sfwf.sfwf_contraction_cpu_numpy.__name__: (False, sfwf.sfwf_contraction_cpu_numpy, DEFAULT_REPETITIONS, {}),    
    sfwf.sfwf_contraction_cpu_numba_parallel.__name__: (True, sfwf.sfwf_contraction_cpu_numba_parallel, DEFAULT_REPETITIONS, {}),
    sfwf.sfwf_contraction_cuda_small.__name__: (True, sfwf.sfwf_contraction_cuda_small, DEFAULT_REPETITIONS, {"tpb": sfwf.DEFAULT_TPB}),
    sfwf.sfwf_contraction_cuda_large_atomicmax.__name__: (True, sfwf.sfwf_contraction_cuda_large_atomicmax, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE}),    
    sfwf.sfwf_contraction_cuda_large_atomicmaxglosten.__name__: (True, sfwf.sfwf_contraction_cuda_large_atomicmaxglosten, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE}),
    sfwf.sfwf_contraction_cuda_large_hreducemax.__name__: (True, sfwf.sfwf_contraction_cuda_large_hreducemax, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE, "tpb_reduce": sfwf.DEFAULT_TPB}),    
    sfwf.sfwf_contraction_cuda_large_hreducemaxgs.__name__: (True, sfwf.sfwf_contraction_cuda_large_hreducemaxgs, DEFAULT_REPETITIONS, {"lazy_stop_check": sfwf.DEFAULT_LAZY_STOP_CHECK, "tpb_side": sfwf.DEFAULT_TPB_SIDE, "tpb_reduce": sfwf.DEFAULT_TPB, "cores": g_props["cores_total"]}),    
    sfwf.sfwf_contraction_cuda_large_gridsync.__name__: (True, sfwf.sfwf_contraction_cuda_large_gridsync, DEFAULT_REPETITIONS, {"tpb_side": sfwf.DEFAULT_TPB_SIDE})    
    }
```


# Acknowledgements
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.
