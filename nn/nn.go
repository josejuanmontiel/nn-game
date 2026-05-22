package nn

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

func NewNeuralNetwork(inSize int, hiddenSizes []int, outSize int) *NeuralNetwork {
	nn := &NeuralNetwork{
		InSize:         inSize,
		OutSize:        outSize,
		ActivationType: "tanh",
		Activation:     Tanh,
		Derivative:     func(x float64) float64 { return 1.0 - x*x },
		WeightScale:    1.0,
		BiasScale:      0.01,
		EngineType:     "standard",
		Engine:         &FallbackEngine{}, // Default to avoid nil panics
	}

	allSizes := append([]int{inSize}, hiddenSizes...)
	allSizes = append(allSizes, outSize)

	for i := 0; i < len(allSizes)-1; i++ {
		in, out := allSizes[i], allSizes[i+1]
		layer := &Layer{
			Weights:    make([][]float64, out),
			Biases:     make([]float64, out),
			Inputs:     make([]float64, in),
			Outputs:    make([]float64, out),
			Errors:     make([]float64, out),
			PrevDeltaW: make([][]float64, out),
			PrevDeltaB: make([]float64, out),
			LastDelta:  make([]float64, out),
			LastGradW:  make([][]float64, out),
			LastGradB:  make([]float64, out),
		}
		for j := range layer.Weights {
			layer.Weights[j] = make([]float64, in)
			layer.PrevDeltaW[j] = make([]float64, in)
			layer.LastGradW[j] = make([]float64, in)
		}
		nn.Layers = append(nn.Layers, layer)
	}

	nn.ResetWeights()
	return nn
}

func (nn *NeuralNetwork) AddLayer(nodes int) {
	lastSize := nn.InSize
	if len(nn.Layers) > 1 {
		lastSize = len(nn.Layers[len(nn.Layers)-2].Outputs)
	}
	outLayer := nn.Layers[len(nn.Layers)-1]
	newLayer := &Layer{
		Weights:    make([][]float64, nodes),
		Biases:     make([]float64, nodes),
		Inputs:     make([]float64, lastSize),
		Outputs:    make([]float64, nodes),
		Errors:     make([]float64, nodes),
		PrevDeltaW: make([][]float64, nodes),
		PrevDeltaB: make([]float64, nodes),
		LastDelta:  make([]float64, nodes),
		LastGradW:  make([][]float64, nodes),
		LastGradB:  make([]float64, nodes),
	}
	for i := range newLayer.Weights {
		newLayer.Weights[i] = make([]float64, lastSize)
		newLayer.PrevDeltaW[i] = make([]float64, lastSize)
		newLayer.LastGradW[i] = make([]float64, lastSize)
		for j := range newLayer.Weights[i] {
			if i == j {
				newLayer.Weights[i][j] = 1.0
			}
		}
	}
	outLayer.Inputs = make([]float64, nodes)
	outLayer.Weights = make([][]float64, nn.OutSize)
	outLayer.PrevDeltaW = make([][]float64, nn.OutSize)
	outLayer.LastGradW = make([][]float64, nn.OutSize)
	outLayer.LastGradB = make([]float64, nn.OutSize)
	outLayer.LastDelta = make([]float64, nn.OutSize)
	for i := range outLayer.Weights {
		outLayer.Weights[i] = make([]float64, nodes)
		outLayer.PrevDeltaW[i] = make([]float64, nodes)
		outLayer.LastGradW[i] = make([]float64, nodes)
		for j := range outLayer.Weights[i] {
			if i == j {
				outLayer.Weights[i][j] = 1.0
			}
		}
	}
	nn.Layers = append(nn.Layers[:len(nn.Layers)-1], newLayer, outLayer)
}

func (nn *NeuralNetwork) RemoveLayer() {
	if len(nn.Layers) <= 1 {
		return
	}
	nn.Layers = append(nn.Layers[:len(nn.Layers)-2], nn.Layers[len(nn.Layers)-1])
	outLayer := nn.Layers[len(nn.Layers)-1]
	prevSize := nn.InSize
	if len(nn.Layers) > 1 {
		prevSize = len(nn.Layers[len(nn.Layers)-2].Outputs)
	}
	outLayer.Inputs = make([]float64, prevSize)
	outLayer.Weights = make([][]float64, nn.OutSize)
	outLayer.PrevDeltaW = make([][]float64, nn.OutSize)
	outLayer.LastGradW = make([][]float64, nn.OutSize)
	outLayer.LastGradB = make([]float64, nn.OutSize)
	outLayer.LastDelta = make([]float64, nn.OutSize)
	for i := range outLayer.Weights {
		outLayer.Weights[i] = make([]float64, prevSize)
		outLayer.PrevDeltaW[i] = make([]float64, prevSize)
		outLayer.LastGradW[i] = make([]float64, prevSize)
	}
	nn.ResetWeights()
}

func (nn *NeuralNetwork) ResetWeights() {
	if len(nn.LayersV2) > 0 {
		for _, layer := range nn.LayersV2 {
			layer.ResetWeights()
		}
		return
	}

	for _, layer := range nn.Layers {
		inSize := len(layer.Inputs)
		r := math.Sqrt(6.0/float64(inSize+len(layer.Outputs))) * nn.WeightScale
		for i := range layer.Weights {
			for j := range layer.Weights[i] {
				layer.Weights[i][j] = (rand.Float64()*2 - 1) * r
			}
			layer.Biases[i] = (rand.Float64()*2 - 1) * nn.BiasScale
		}
	}
}

func (nn *NeuralNetwork) Forward(inputs []float64) []float64 {
	if len(nn.LayersV2) > 0 {
		var currentInput any = inputs
		if inputs == nil && len(nn.LastOutput) > 0 {
			return nn.LastOutput
		}
		
		for _, layer := range nn.LayersV2 {
			currentInput = layer.Forward(currentInput)
		}
		// If the last layer is a Flatten layer or similar returning []float64
		var res []float64
		if r, ok := currentInput.([]float64); ok {
			res = r
		} else if tensor, ok := currentInput.(*Tensor3D); ok {
			res = tensor.Data
		}
		
		if len(nn.LastOutput) != len(res) {
			nn.LastOutput = make([]float64, len(res))
		}
		copy(nn.LastOutput, res)
		return nn.LastOutput
	}

	currentInputs := inputs
	for _, layer := range nn.Layers {
		copy(layer.Inputs, currentInputs)
		nn.Engine.Forward(layer, nn.Activation)
		currentInputs = layer.Outputs
	}
	return currentInputs
}

func (nn *NeuralNetwork) GetV2LayerOutputs(inputs []float64) []any {
	if len(nn.LayersV2) == 0 {
		return nil
	}
	var currentInput any = inputs
	outputs := make([]any, len(nn.LayersV2))
	for i, layer := range nn.LayersV2 {
		currentInput = layer.Forward(currentInput)
		outputs[i] = currentInput
	}
	return outputs
}

func (nn *NeuralNetwork) Transform2D(x, y float64) (float64, float64, float64) {
	return nn.TransformPartial([]float64{x, y}, len(nn.Layers))
}

func (nn *NeuralNetwork) Transform(inputs []float64) (float64, float64, float64) {
	return nn.TransformPartial(inputs, len(nn.Layers))
}

func (nn *NeuralNetwork) TransformPartial(inputs []float64, layerIdx int) (float64, float64, float64) {
	if len(nn.LayersV2) > 0 {
		out := nn.Forward(inputs)
		if len(out) >= 3 {
			return out[0], out[1], out[2]
		}
		// Fallback si no hay 3 dimensiones
		res := make([]float64, 3)
		for i := 0; i < len(out) && i < 3; i++ {
			res[i] = out[i]
		}
		return res[0], res[1], res[2]
	}

	// 1. Prepare correctly sized input for the first layer
	firstIn := inputs
	if len(inputs) < nn.InSize {
		firstIn = make([]float64, nn.InSize)
		copy(firstIn, inputs)
	}

	currentInputs := firstIn
	for i := 0; i < layerIdx && i < len(nn.Layers); i++ {
		layer := nn.Layers[i]
		// We use a safe copy instead of pointer swapping to avoid engine panics
		copy(layer.Inputs, currentInputs)
		nn.Engine.Forward(layer, nn.Activation)
		currentInputs = layer.Outputs
	}
	res := make([]float64, 3)
	for i := 0; i < 3 && i < len(currentInputs); i++ {
		res[i] = currentInputs[i]
	}
	// Default to zeros for safety
	return res[0], res[1], res[2]
}

func (nn *NeuralNetwork) GetLayerOutputs(x, y float64, layerIdx int) []float64 {
	// Pad 2D input to match network's InputSize (e.g. 10D)
	inputs := make([]float64, nn.InSize)
	inputs[0] = x
	inputs[1] = y

	currentInputs := inputs
	for i := 0; i < layerIdx && i < len(nn.Layers); i++ {
		layer := nn.Layers[i]
		copy(layer.Inputs, currentInputs)
		nn.Engine.Forward(layer, nn.Activation)
		currentInputs = layer.Outputs
	}
	// Return a copy to be safe
	res := make([]float64, len(currentInputs))
	copy(res, currentInputs)
	return res
}

func (nn *NeuralNetwork) TraceNode(layerIdx, nodeIdx int) NodeTrace {
	if layerIdx < 1 || layerIdx > len(nn.Layers) {
		return NodeTrace{}
	}
	layer := nn.Layers[layerIdx-1]
	if nodeIdx < 0 || nodeIdx >= len(layer.Outputs) {
		return NodeTrace{}
	}

	numInputs := len(layer.Inputs)
	numWeights := 0
	if nodeIdx < len(layer.Weights) {
		numWeights = len(layer.Weights[nodeIdx])
	}

	trace := NodeTrace{
		LayerIdx: layerIdx,
		NodeIdx:  nodeIdx,
		Inputs:   make([]float64, numInputs),
		Weights:  make([]float64, numWeights),
		Bias:     0,
		Output:   layer.Outputs[nodeIdx],
		Delta:    0,
		Gradient: make([]float64, numWeights),
	}

	copy(trace.Inputs, layer.Inputs)
	if nodeIdx < len(layer.Weights) {
		copy(trace.Weights, layer.Weights[nodeIdx])
	}
	if nodeIdx < len(layer.Biases) {
		trace.Bias = layer.Biases[nodeIdx]
	}
	if nodeIdx < len(layer.LastDelta) {
		trace.Delta = layer.LastDelta[nodeIdx]
	}
	if nodeIdx < len(layer.LastGradW) {
		copy(trace.Gradient, layer.LastGradW[nodeIdx])
	}

	z := trace.Bias
	if len(trace.Weights) == len(trace.Inputs) {
		for i := range trace.Inputs {
			z += trace.Inputs[i] * trace.Weights[i]
		}
	} else {
		z = trace.Output // Approximation for proxy layers
	}
	trace.Z = z

	return trace
}

func (nn *NeuralNetwork) TrainStep(inputs []float64, targets []float64, lr float64) {
	nn.Forward(inputs)
	nn.Backpropagate(targets, lr)
}

func (nn *NeuralNetwork) Backpropagate(targets []float64, lr float64) {
	if len(nn.LayersV2) > 0 {
		// Calculate initial gradient (target - output)
		// We use the cached LastOutput from the previous Forward pass
		if len(nn.LastOutput) == 0 {
			return // Should not happen if Forward was called
		}
		
		grad := make([]float64, len(targets))
		for i := range grad {
			grad[i] = targets[i] - nn.LastOutput[i]
		}

		var currentGrad any = grad
		for i := len(nn.LayersV2) - 1; i >= 0; i-- {
			currentGrad = nn.LayersV2[i].Backward(currentGrad)
		}

		for _, layer := range nn.LayersV2 {
			layer.Update(lr)
		}
		return
	}

	if len(nn.Layers) == 0 || len(targets) == 0 {
		return
	}
	lastLayer := nn.Layers[len(nn.Layers)-1]
	for i := range lastLayer.Errors {
		if i >= len(targets) {
			break
		}
		lastLayer.Errors[i] = (targets[i] - lastLayer.Outputs[i]) * nn.Derivative(lastLayer.Outputs[i])
	}
	for i := len(nn.Layers) - 2; i >= 0; i-- {
		nn.Engine.Backward(nn.Layers[i], nn.Layers[i+1], nn.Derivative)
	}
	momentum := 0.9
	for _, layer := range nn.Layers {
		nn.Engine.Update(layer, lr, momentum)
	}
}

func (nn *NeuralNetwork) Save(filename string) error {
	data, err := json.MarshalIndent(nn, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func LoadFromFile(filename string) (*NeuralNetwork, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var nn NeuralNetwork
	err = json.Unmarshal(data, &nn)
	if err != nil {
		return nil, err
	}
	nn.SetActivation(nn.ActivationType)
	return &nn, nil
}

func (nn *NeuralNetwork) SetEngine(engine Engine, engineType string) {
	nn.Engine = engine
	nn.EngineType = engineType
}

func (nn *NeuralNetwork) SetActivation(actType string) {
	nn.ActivationType = actType
	switch actType {
	case "relu":
		nn.Activation = ReLU
		nn.Derivative = func(x float64) float64 {
			if x > 0 {
				return 1
			}
			return 0
		}
	case "tanh":
		nn.Activation = Tanh
		nn.Derivative = func(x float64) float64 { return 1.0 - x*x }
	case "sigmoid":
		nn.Activation = Sigmoid
		nn.Derivative = func(x float64) float64 { return x * (1 - x) }
	case "identity":
		nn.Activation = Identity
		nn.Derivative = func(x float64) float64 { return 1 }
	default:
		nn.Activation = Tanh
		nn.Derivative = func(x float64) float64 { return 1.0 - x*x }
	}
}

func (nn *NeuralNetwork) Summary() string {
	res := ""
	for l, layer := range nn.Layers {
		res += fmt.Sprintf("L%d[", l+1)
		for i := range layer.Biases {
			res += fmt.Sprintf("B%.2f ", layer.Biases[i])
		}
		res += "| "
		for i := range layer.Weights {
			for j := range layer.Weights[i] {
				res += fmt.Sprintf("W%.2f ", layer.Weights[i][j])
			}
		}
		res += "] "
	}
	return res
}
