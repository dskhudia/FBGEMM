# FBGEMM

## Linux Build: [![CircleCI](https://circleci.com/gh/pytorch/FBGEMM.svg?style=svg)](https://circleci.com/gh/pytorch/FBGEMM)

FBGEMM (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference. 

The library provides efficient low-precision general matrix multiplication for
small batch sizes and support for accuracy-loss minimizing techniques such as
row-wise quantization and outlier-aware quantization. FBGEMM also exploits
fusion opportunities in order to overcome the unique challenges of matrix
multiplication at lower precision with bandwidth-bound operations.

FBGEMM is used as a backend of Caffe2 quantized operators for x86 machines
(https://github.com/pytorch/pytorch/tree/master/caffe2/quantization/server).
We also plan to integrate FBGEMM into PyTorch.

## Examples

The tests (in test folder) and benchmarks (in bench folder) are some great
examples of using FBGEMM. For instance, SpMDMTest test in
test/PackedRequantizeAcc16Test.cc shows how to combine row offset calculations
with packing of A (PackAWithRowOffset), how to pack B matrix (PackBMatrix) and
construct output pipeline (sparse\_matrix\*dense\_matrix --> requantization -->
nop) fused with inner GEMM macro kernel.

## Build Notes
FBGEMM uses the standard CMAKE-based build flow.

### Dependencies
FBGEMM requires gcc 4.9+ and a CPU with support for avx2 instruction set or
higher. It's been tested on Mac OS X and Linux.

+ ###### asmjit
With inner kernels, FBGEMM takes a “one size doesn't fit all” approach, so the
implementation dynamically generates efficient matrix-shape specific vectorized
code using a third-party library called [asmjit][1]. **asmjit is required** to
build FBGEMM. 

+ ###### cpuinfo
FBGEMM detects CPU instruction set support at runtime using cpuinfo library and
dispatches optimized kernels for the detected instruction set. Therefore,
**cpuinfo is required** to detect CPU type.

+ ###### googletest
googletest is required to build and run FBGEMM's tests. **googletest is not
required** if you don't want to run FBGEMM tests. By default, building of tests
is **on**. Turn it off by setting FBGEMM\_BUILD\_TESTS to off.

You can download [asmjit][1], [cpuinfo][2], [googletest][3] and set
ASMJIT\_SRC\_DIR, CPUINFO\_SRC\_DIR, GOOGLETEST\_SOURCE\_DIR respectively for
cmake to find these libraries. If any of these variables is not set, cmake will
try to download that missing library in a folder called third\_party in the
build directory and build it using the downloaded source code.

FBGEMM, in general, does not have any dependency on Intel MKL. However, for
performance comparison, some benchmarks use MKL functions. If MKL is found or
MKL path is provided with INTEL\_MKL\_DIR benchmarks are built with MKL and
performance numbers are reported for MKL functions as well. However, if MKL is
not found, the benchmarks are not built.

General build instructions are as follows:

```
mkdir build && cd build
cmake ..
make
```

To run the tests after building FBGEMM (if tests are built), use the following
command:
```
make test
```

## Installing  FBGEMM
```
make install
```

## How FBGEMM works
For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM please see [our blog][4].

## Full documentation
We have extensively used comments in our source files. The best and up-do-date
documentation is available in the source files.

## Join the FBGEMM community
See the [`CONTRIBUTING`](CONTRIBUTING.md) file for how to help out.

## License
FBGEMM is BSD licensed, as found in the [`LICENSE`](LICENSE) file.


[1]:https://github.com/asmjit/asmjit
[2]:https://github.com/pytorch/cpuinfo
[3]:https://github.com/google/googletest
[4]:https://code.fb.com/ml-applications/fbgemm
