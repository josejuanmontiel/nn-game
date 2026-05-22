#include <sycl/sycl.hpp>
#include <iostream>
#include "matrix_sycl.h"

extern "C" void matrix_multiply_sycl(const float* A, const float* B, float* C, int N) {
    sycl::queue q{sycl::gpu_selector_v};

    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    {
        sycl::range<2> matrix_range(N, N);
        sycl::buffer<float, 2> bufA(A, matrix_range);
        sycl::buffer<float, 2> bufB(B, matrix_range);
        sycl::buffer<float, 2> bufC(C, matrix_range);

        q.submit([&](sycl::handler& h) {
            auto accessorA = bufA.get_access<sycl::access::mode::read>(h);
            auto accessorB = bufB.get_access<sycl::access::mode::read>(h);
            auto accessorC = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for(matrix_range, [=](sycl::id<2> id) {
                int row = id[0];
                int col = id[1];
                float sum = 0.0f;
                for (int i = 0; i < N; i++) {
                    sum += accessorA[row][i] * accessorB[i][col];
                }
                accessorC[row][col] = sum;
            });
        });
    }
    q.wait();
}
