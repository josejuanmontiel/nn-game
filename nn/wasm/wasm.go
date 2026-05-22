//go:build js

package wasm

import (
	"11-juego-final/nn"
)

type Engine struct{}

// Forward implements the neural network forward pass.
// Optimized with loop unrolling to encourage WASM SIMD vectorization.
func (e *Engine) Forward(layer *nn.Layer, activation nn.ActivationFunc) {
	inputs := layer.Inputs
	for i := 0; i < len(layer.Outputs); i++ {
		sum := layer.Biases[i]
		weights := layer.Weights[i]

		// Help BCE by ensuring slices are checked once
		n := len(inputs)
		if len(weights) < n {
			n = len(weights)
		}

		j := 0
		// 4x Loop Unrolling for Dot Product
		for ; j <= n-4; j += 4 {
			sum += inputs[j] * weights[j]
			sum += inputs[j+1] * weights[j+1]
			sum += inputs[j+2] * weights[j+2]
			sum += inputs[j+3] * weights[j+3]
		}
		// Remainder
		for ; j < n; j++ {
			sum += inputs[j] * weights[j]
		}
		layer.Outputs[i] = activation(sum)
	}
}

func (e *Engine) Backward(layer *nn.Layer, nextLayer *nn.Layer, derivative nn.ActivationFunc) {
	nextErrors := nextLayer.Errors
	numNext := len(nextErrors)

	for j := range layer.Errors {
		err := 0.0
		// Backprop error: sum(next_errors[k] * weight[k][j])
		k := 0
		for ; k <= numNext-4; k += 4 {
			err += nextErrors[k] * nextLayer.Weights[k][j]
			err += nextErrors[k+1] * nextLayer.Weights[k+1][j]
			err += nextErrors[k+2] * nextLayer.Weights[k+2][j]
			err += nextErrors[k+3] * nextLayer.Weights[k+3][j]
		}
		for ; k < numNext; k++ {
			err += nextErrors[k] * nextLayer.Weights[k][j]
		}
		layer.Errors[j] = err * derivative(layer.Outputs[j])
	}
}

func (e *Engine) Update(layer *nn.Layer, lr, momentum float64) {
	inputs := layer.Inputs
	numInputs := len(inputs)

	for i := range layer.Weights {
		errI := layer.Errors[i]
		rowW := layer.Weights[i]
		rowPrev := layer.PrevDeltaW[i]

		j := 0
		for ; j <= numInputs-4; j += 4 {
			// Update Weight j
			grad0 := errI * inputs[j]
			delta0 := lr*grad0 + momentum*rowPrev[j]
			rowW[j] += delta0
			rowPrev[j] = delta0

			// Update Weight j+1
			grad1 := errI * inputs[j+1]
			delta1 := lr*grad1 + momentum*rowPrev[j+1]
			rowW[j+1] += delta1
			rowPrev[j+1] = delta1

			// Update Weight j+2
			grad2 := errI * inputs[j+2]
			delta2 := lr*grad2 + momentum*rowPrev[j+2]
			rowW[j+2] += delta2
			rowPrev[j+2] = delta2

			// Update Weight j+3
			grad3 := errI * inputs[j+3]
			delta3 := lr*grad3 + momentum*rowPrev[j+3]
			rowW[j+3] += delta3
			rowPrev[j+3] = delta3
		}
		for ; j < numInputs; j++ {
			grad := errI * inputs[j]
			delta := lr*grad + momentum*rowPrev[j]
			rowW[j] += delta
			rowPrev[j] = delta
		}

		gradB := errI
		deltaB := lr*gradB + momentum*layer.PrevDeltaB[i]
		layer.Biases[i] += deltaB
		layer.PrevDeltaB[i] = deltaB
	}
}
