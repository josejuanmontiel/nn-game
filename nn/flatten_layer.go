package nn

type FlattenLayer struct {
	LastInput *Tensor3D
}

func NewFlattenLayer() *FlattenLayer {
	return &FlattenLayer{}
}

func (l *FlattenLayer) Forward(input any) any {
	in := input.(*Tensor3D)
	l.LastInput = in
	// Simple copy of input data to 1D slice
	output := make([]float64, len(in.Data))
	copy(output, in.Data)
	return output
}

func (l *FlattenLayer) Backward(gradOutput any) any {
	gradOut := gradOutput.([]float64)
	gradInput := NewTensor3D(l.LastInput.Channels, l.LastInput.Height, l.LastInput.Width)
	copy(gradInput.Data, gradOut)
	return gradInput
}

func (l *FlattenLayer) Update(lr float64) {
	// Nothing to update
}

func (l *FlattenLayer) ResetWeights() {
	// Nothing to reset
}
