# CUDA-based contraction iteration soap film in a wire frame
The repository constitutes a part of research on CUDA computational approaches for algorithms based on contraction mapping theorem due to Banach, a.k.a. the fixed-point theorem.
Imagine a wire frame of irregular shape, forming a closed loop, and a soap film surface spanned within it. The task is to compute the shape of that surface knowing the frame shape. 

TODO img

Implementations of CUDA kernels have been carried out using Numba - a just-in-time compiler for Python. Numba exposes a programming interface closely mirroring the native CUDA C++ API, and translates kernel functions into its internal representation (Numba IR), which is then lowered via the LLVM and NVVM-based pipeline into PTX and finally JIT-compiled into executable machine code.


# Mathematics
TODO

# Speed-ups
TODO


# Selected experimental results (averages over 10 repetitions)
TODO


# Acknowledgements
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.
