package standard

import (
	"11-juego-final/nn"
)

type Engine struct{}

func (e *Engine) Forward(layer *nn.Layer, activation nn.ActivationFunc) {
	for i := 0; i < len(layer.Outputs); i++ {
		sum := layer.Biases[i]
		for j := 0; j < len(layer.Inputs); j++ {
			sum += layer.Inputs[j] * layer.Weights[i][j]
		}
		layer.Outputs[i] = activation(sum)
	}
}

func (e *Engine) Backward(layer *nn.Layer, nextLayer *nn.Layer, derivative nn.ActivationFunc) {
	for j := range layer.Errors {
		err := 0.0
		for k := range nextLayer.Errors {
			err += nextLayer.Errors[k] * nextLayer.Weights[k][j]
		}
		layer.Errors[j] = err * derivative(layer.Outputs[j])
	}
}

func (e *Engine) Update(layer *nn.Layer, lr, momentum float64) {
	copy(layer.LastDelta, layer.Errors)
	for i := range layer.Weights {
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
}
