//go:build amd64 && !js

package avx

import (
	"11-juego-final/nn"
	"11-juego-final/nn/standard"
)

// VecAddAVX (from matrix_avx.s)
func VecAddAVX(a []float32, b []float32, res []float32)

type Engine struct {
	standard.Engine
}

func (e *Engine) Forward(layer *nn.Layer, activation nn.ActivationFunc) {
	e.Engine.Forward(layer, activation)

	// Demo
	v1 := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	v2 := []float32{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}
	res := make([]float32, 8)
	VecAddAVX(v1, v2, res)
}
