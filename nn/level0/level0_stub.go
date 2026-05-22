//go:build !gpu

package level0

import (
	"11-juego-final/nn"
	"11-juego-final/nn/standard"
)

type Engine struct {
	standard.Engine
}

func (e *Engine) Forward(layer *nn.Layer, activation nn.ActivationFunc) {
	e.Engine.Forward(layer, activation)
}
