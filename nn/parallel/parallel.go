package parallel

import (
	"runtime"
	"sync"
	"11-juego-final/nn"
)

type Engine struct{}

func (e *Engine) Forward(layer *nn.Layer, activation nn.ActivationFunc) {
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	if numCPU > len(layer.Outputs) {
		numCPU = len(layer.Outputs)
	}
	chunkSize := (len(layer.Outputs) + numCPU - 1) / numCPU

	for c := 0; c < numCPU; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > len(layer.Outputs) {
			end = len(layer.Outputs)
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				sum := layer.Biases[i]
				for j := 0; j < len(layer.Inputs); j++ {
					sum += layer.Inputs[j] * layer.Weights[i][j]
				}
				layer.Outputs[i] = activation(sum)
			}
		}(start, end)
	}
	wg.Wait()
}

func (e *Engine) Backward(layer *nn.Layer, nextLayer *nn.Layer, derivative nn.ActivationFunc) {
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	if numCPU > len(layer.Errors) {
		numCPU = len(layer.Errors)
	}
	chunkSize := (len(layer.Errors) + numCPU - 1) / numCPU

	for c := 0; c < numCPU; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > len(layer.Errors) {
			end = len(layer.Errors)
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				err := 0.0
				for k := range nextLayer.Errors {
					err += nextLayer.Errors[k] * nextLayer.Weights[k][j]
				}
				layer.Errors[j] = err * derivative(layer.Outputs[j])
			}
		}(start, end)
	}
	wg.Wait()
}

func (e *Engine) Update(layer *nn.Layer, lr, momentum float64) {
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	if numCPU > len(layer.Weights) {
		numCPU = len(layer.Weights)
	}
	chunkSize := (len(layer.Weights) + numCPU - 1) / numCPU

	copy(layer.LastDelta, layer.Errors)

	for c := 0; c < numCPU; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > len(layer.Weights) {
			end = len(layer.Weights)
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				for j := range layer.Weights[i] {
					grad := layer.Errors[i] * layer.Inputs[j]
					layer.LastGradW[i][j] = grad

					delta := lr*grad + momentum*layer.PrevDeltaW[i][j]
					layer.Weights[i][j] += delta
					layer.PrevDeltaW[i][j] = delta
				}
				gradB := layer.Errors[i]
				layer.LastGradB[i] = gradB

				deltaB := lr*gradB + momentum*layer.PrevDeltaB[i]
				layer.Biases[i] += deltaB
				layer.PrevDeltaB[i] = deltaB
			}
		}(start, end)
	}
	wg.Wait()
}
