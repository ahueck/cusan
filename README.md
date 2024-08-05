# cusan

CuSan is tool for analyzing and instrumenting CUDA codes to track CUDA domain-specific memory accesses and synchronization semantics in order to find data races between (asynchronous) CUDA calls and the host.

To that end, CuSan uses LLVM to instrument CUDA calls during compilation, to eventually track their semantics in our runtime.
The runtime passes these information appropriately to ThreadSanitizer (packaged with Clang/LLVM) for the final data race analysis.

## Usage

Making use of CuSan consists of two phases:

1. Compile your code with Clang/LLVM (version 14) using one the CuSan compiler wrappers, e.g., `cusan-mpic++`.
This will analyze and instrument the CUDA API appropriately, and link our runtime.
2. Execute the target program with our runtime library to accept the callbacks to do data race analysis with our interface based ThreadSanitizer. You need to use the MPI correctness checker MUST or preload our (very) simple MPI interceptor `libCusanMPIInterceptor.so` for CUDA-aware MPI data race detection.


### Example report
The following is an example report for [03_cuda_to_mpi.c](test/pass/03_cuda_to_mpi.c) of our test suite, excerpt:
```c
   kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    // cudaDeviceSynchronize();  // FIXME: uncomment otherwise a data race happens
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
```
```
==================
WARNING: ThreadSanitizer: data race (pid=689288)
  Read of size 8 at 0x7fb51f200000 by main thread:
    #0 main cusan/test/pass/03_cuda_to_mpi.c:66:5 (03_cuda_to_mpi.c.exe+0x4e8448)

  Previous write of size 8 at 0x7fb51f200000 by thread T6:
    #0 __device_stub__kernel(int*, int) cusan/test/pass/03_cuda_to_mpi.c:23:47 (03_cuda_to_mpi.c.exe+0x4e81ef)

  Thread T6 'cuda_stream' (tid=0, running) created by main thread at:
    #0 __pool_create_fiber_dbg cusan/build/_deps/fiber_pool-src/fiberpool.cpp:538:16 (libCusanFiberpool-d.so+0x1c152)
    #1 main cusan/test/pass/03_cuda_to_mpi.c:59:3 (03_cuda_to_mpi.c.exe+0x4e8331)

SUMMARY: ThreadSanitizer: data race cusan/test/pass/03_cuda_to_mpi.c:66:5 in main
==================
ThreadSanitizer: reported 1 warnings
```

## Building cusan

cusan requires LLVM version 14 and CMake version >= 3.20. Use CMake presets `develop` or `release`
to build.

### Dependencies
CuSan was tested with:
- System modules: `1) gcc/11.2.0 2) cuda/11.5 3) openmpi/4.1.6 4) git/2.40.0 5) python/3.10.10 6) clang/14.0.6`
- External libraries: TypeART (https://github.com/tudasc/TypeART/tree/feat/cuda), FiberPool (optional)
- Testing: llvm-lit, FileCheck
- GPU: Tesla T4 and Tesla V100 (mostly: arch=sm_70)

### Build example

cusan uses CMake to build. Example build recipe (release build, installs to default prefix
`${cusan_SOURCE_DIR}/install/cusan`)

```sh
$> cd cusan
$> cmake --preset release
$> cmake --build build --target install --parallel
```
