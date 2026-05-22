#ifndef MATRIX_SYCL_H
#define MATRIX_SYCL_H

#ifdef __cplusplus
extern "C" {
#endif

void matrix_multiply_sycl(const float* A, const float* B, float* C, int N);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_SYCL_H
