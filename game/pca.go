package game

import (
	"math"
)

// ComputePCA finds the top 3 principal components of a set of points
func ComputePCA(points [][]float64) ProjectionMatrix {
	if len(points) < 3 {
		return IdentityProjection()
	}

	dim := len(points[0])
	// 1. Center the data
	means := make([]float64, dim)
	for _, p := range points {
		for i := 0; i < dim && i < len(p); i++ {
			means[i] += p[i]
		}
	}
	for i := range means {
		means[i] /= float64(len(points))
	}

	centered := make([][]float64, len(points))
	for i, p := range points {
		centered[i] = make([]float64, dim)
		for j := 0; j < dim && j < len(p); j++ {
			centered[i][j] = p[j] - means[j]
		}
	}

	var proj ProjectionMatrix

	// 2. Power Iteration to find top 3 components
	// We'll work on the Covariance matrix C = (X^T * X) / (N-1)
	data := centered

	for comp := 0; comp < 3; comp++ {
		// Start with an initial basis vector
		v := make([]float64, dim)
		if comp < dim {
			v[comp] = 1.0
		} else {
			v[0] = 1.0
		}

		// 15 iterations are usually enough for such small dimensions
		for iter := 0; iter < 15; iter++ {
			// v_next = X^T * (X * v)
			// X * v (N x 1)
			xv := make([]float64, len(data))
			for i := range data {
				for j := 0; j < dim; j++ {
					xv[i] += data[i][j] * v[j]
				}
			}

			// X^T * xv (dim x 1)
			xtxv := make([]float64, dim)
			for j := 0; j < dim; j++ {
				for i := range data {
					xtxv[j] += data[i][j] * xv[i]
				}
			}

			// Normalize
			norm := 0.0
			for _, val := range xtxv { norm += val * val }
			norm = math.Sqrt(norm)
			if norm < 1e-9 { break }
			for j := range v { v[j] = xtxv[j] / norm }
		}

		// Store in projection matrix (only up to 5 dimensions supported by Matrix type)
		for j := 0; j < dim && j < 5; j++ {
			proj[comp][j] = v[j]
		}

		// Deflate: Subtract the projection onto v from the data
		for i := range data {
			projection := 0.0
			for j := 0; j < dim; j++ {
				projection += data[i][j] * v[j]
			}
			for j := 0; j < dim; j++ {
				data[i][j] -= projection * v[j]
			}
		}
	}

	return proj
}
