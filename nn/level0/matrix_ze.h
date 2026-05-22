//go:build gpu
#ifndef MATRIX_ZE_H
#define MATRIX_ZE_H

#ifdef __cplusplus
extern "C" {
#endif

void ze_init();
void ze_cleanup();
void ze_matrix_multiply(const float* A, const float* B, float* C, int N);

#ifdef __cplusplus
}
#endif

#endif
