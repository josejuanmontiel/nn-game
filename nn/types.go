package nn

import (
	"math"
	"math/rand/v2"
)

type ActivationFunc func(float64) float64

// NodeTrace encapsulates the "mathematical explanation" of a single node's calculation.
// This allows different backends (Standard, GoMLX, etc.) to provide their own trace data.
type NodeTrace struct {
	LayerIdx int
	NodeIdx  int
	Inputs   []float64
	Weights  []float64
	Bias     float64
	Z        float64 // Pre-activation sum
	Output   float64 // Final activation
	Delta    float64 // Error signal
	Gradient []float64
}

// Engine defines the low-level math operations for a neural network layer.
type Engine interface {
	Forward(layer *Layer, activation ActivationFunc)
	Backward(layer *Layer, nextLayer *Layer, derivative ActivationFunc)
	Update(layer *Layer, lr, momentum float64)
}

type ILayer interface {
	Forward(input any) any // input/output can be []float64 (Dense) or *Tensor3D (CNN)
	Backward(gradOutput any) any
	Update(lr float64)
	ResetWeights() // To allow global re-init
}

// FallbackEngine is a simple sequential implementation to avoid nil errors.
type FallbackEngine struct{}

func (e *FallbackEngine) Forward(layer *Layer, activation ActivationFunc) {
	for i := 0; i < len(layer.Outputs); i++ {
		sum := layer.Biases[i]
		for j := 0; j < len(layer.Inputs); j++ {
			sum += layer.Inputs[j] * layer.Weights[i][j]
		}
		layer.Outputs[i] = activation(sum)
	}
}

func (e *FallbackEngine) Backward(layer *Layer, nextLayer *Layer, derivative ActivationFunc) {
	for j := range layer.Errors {
		err := 0.0
		for k := range nextLayer.Errors {
			err += nextLayer.Errors[k] * nextLayer.Weights[k][j]
		}
		layer.Errors[j] = err * derivative(layer.Outputs[j])
	}
}

func (e *FallbackEngine) Update(layer *Layer, lr, momentum float64) {
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

type Layer struct {
	Weights [][]float64
	Biases  []float64
	Inputs  []float64
	Outputs []float64
	Errors  []float64
	// Momentum buffers
	PrevDeltaW [][]float64
	PrevDeltaB []float64

	// Visualization buffers for Tracer
	LastDelta []float64
	LastGradW [][]float64
	LastGradB []float64

	// ILayer support
	Activation ActivationFunc
	Derivative ActivationFunc
	Engine     Engine
}

func (l *Layer) Forward(input any) any {
	in := input.([]float64)
	if len(l.Inputs) != len(in) {
		l.Inputs = make([]float64, len(in))
	}
	copy(l.Inputs, in)
	l.Engine.Forward(l, l.Activation)
	return l.Outputs
}

func (l *Layer) Backward(gradOutput any) any {
	grad := gradOutput.([]float64)
	
	// 1. Update this layer's Errors based on the incoming gradient
	for i := range l.Errors {
		l.Errors[i] = grad[i]
	}

	// 2. Compute error for the previous layer to pass it back
	backError := make([]float64, len(l.Inputs))
	for i := range l.Errors {
		for j := range l.Inputs {
			backError[j] += l.Errors[i] * l.Weights[i][j]
		}
	}
	return backError
}

func (l *Layer) Update(lr float64) {
	l.Engine.Update(l, lr, 0.9) // Default momentum
}

func (l *Layer) ResetWeights() {
	inSize := len(l.Inputs)
	outSize := len(l.Outputs)
	if inSize == 0 { inSize = 1 } // Safety
	r := math.Sqrt(6.0/float64(inSize+outSize))
	for i := range l.Weights {
		for j := range l.Weights[i] {
			l.Weights[i][j] = (rand.Float64()*2 - 1) * r
		}
		l.Biases[i] = (rand.Float64()*2 - 1) * 0.01
	}
}

type NeuralNetwork struct {
	Layers         []*Layer
	LayersV2       []ILayer       `json:"-"` // New polymorphic layers
	Activation     ActivationFunc `json:"-"`
	Derivative     ActivationFunc `json:"-"`
	ActivationType string         `json:"activation_type"`
	InSize         int            `json:"in_size"`
	OutSize        int            `json:"out_size"`
	LastOutput     []float64      `json:"-"` // NEW: Store last output for backprop convenience
	WeightScale    float64        `json:"weight_scale"`
	BiasScale      float64        `json:"bias_scale"`
	EngineType     string         `json:"engine_type"`
	Engine         Engine         `json:"-"`
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func Identity(x float64) float64 {
	return x
}

func (nn *NeuralNetwork) Clone() *NeuralNetwork {
	res := &NeuralNetwork{
		Activation:     nn.Activation,
		Derivative:     nn.Derivative,
		ActivationType: nn.ActivationType,
		InSize:         nn.InSize,
		OutSize:        nn.OutSize,
		WeightScale:    nn.WeightScale,
		BiasScale:      nn.BiasScale,
		EngineType:     nn.EngineType,
		Engine:         nn.Engine,
	}
	for _, l := range nn.Layers {
		res.Layers = append(res.Layers, l.Clone())
	}
	return res
}

func (l *Layer) Clone() *Layer {
	res := &Layer{
		Biases:     make([]float64, len(l.Biases)),
		Inputs:     make([]float64, len(l.Inputs)),
		Outputs:    make([]float64, len(l.Outputs)),
		Errors:     make([]float64, len(l.Errors)),
		PrevDeltaB: make([]float64, len(l.PrevDeltaB)),
		LastDelta:  make([]float64, len(l.LastDelta)),
		LastGradB:  make([]float64, len(l.LastGradB)),
	}
	copy(res.Biases, l.Biases)
	copy(res.Inputs, l.Inputs)
	copy(res.Outputs, l.Outputs)
	copy(res.Errors, l.Errors)
	copy(res.PrevDeltaB, l.PrevDeltaB)
	copy(res.LastDelta, l.LastDelta)
	copy(res.LastGradB, l.LastGradB)

	res.Weights = make([][]float64, len(l.Weights))
	res.PrevDeltaW = make([][]float64, len(l.PrevDeltaW))
	res.LastGradW = make([][]float64, len(l.LastGradW))
	for i := range l.Weights {
		res.Weights[i] = make([]float64, len(l.Weights[i]))
		res.PrevDeltaW[i] = make([]float64, len(l.PrevDeltaW[i]))
		res.LastGradW[i] = make([]float64, len(l.LastGradW[i]))
		copy(res.Weights[i], l.Weights[i])
		copy(res.PrevDeltaW[i], l.PrevDeltaW[i])
		copy(res.LastGradW[i], l.LastGradW[i])
	}
	return res
}
