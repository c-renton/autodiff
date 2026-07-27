//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Custom main() for autodiff-cudatests (instead of Catch2::Catch2WithMain,
// used by autodiff-cpptests): a CUDA *compiler* being available at build time
// (CMAKE_CUDA_COMPILER) does not imply a CUDA *device* is available at
// runtime -- e.g. CI runners commonly have the CUDA toolkit installed for
// compilation only, with no physical GPU. Without this check, the CUDA test
// cases fail with cudaErrorInsufficientDriver (cudaMalloc/cudaFree etc. all
// failing) despite there being no actual bug. Detect device availability
// before running the Catch2 session and skip gracefully (exit 0) if no
// device is present, rather than running (and failing) tests that
// structurally cannot pass without a GPU.

#ifdef __clang__
#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS // Prevents error: 'uncaught_exceptions' is unavailable: introduced in macOS 10.12 (see discussion at https://github.com/catchorg/Catch2/issues/1218)
#endif

#include <cstdio>
#include <cuda_runtime.h>
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[])
{
    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if(err != cudaSuccess || deviceCount == 0)
    {
        std::printf(
            "No CUDA-capable device detected (cudaGetDeviceCount: %s, device count: %d).\n"
            "Skipping CUDA tests: the CUDA compiler/toolkit is available, but no GPU is present at runtime.\n",
            cudaGetErrorString(err), deviceCount);
        return 0;
    }

    return Catch::Session().run(argc, argv);
}
