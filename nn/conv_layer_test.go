package nn

import (
	"testing"
)

func TestConvLayer_Forward(t *testing.T) {
	// 1 channel 3x3 input
	input := NewTensor3D(1, 3, 3)
	for i := range input.Data {
		input.Data[i] = float64(i + 1)
	}
	/*
	Input:
	1 2 3
	4 5 6
	7 8 9
	*/

	// 1 filter 2x2, stride 1, no padding
	conv := NewConvLayer(1, 1, 2, 1, 0)
	
	// Set identity-like kernel for predictable result
	// [[1, 0], [0, 0]]
	conv.Kernels[0][0].Set(0, 0, 1.0)
	conv.Kernels[0][0].Set(0, 1, 0.0)
	conv.Kernels[0][0].Set(1, 0, 0.0)
	conv.Kernels[0][0].Set(1, 1, 0.0)
	conv.Biases[0] = 0.0

	outputRaw := conv.Forward(input)
	output := outputRaw.(*Tensor3D)

	// Expected output size: (3-2)/1 + 1 = 2
	if output.Height != 2 || output.Width != 2 {
		t.Errorf("Expected 2x2 output, got %dx%d", output.Height, output.Width)
	}

	// Expected values:
	// Top-left: 1*1 + 2*0 + 4*0 + 5*0 = 1
	// Top-right: 2*1 + 3*0 + 5*0 + 6*0 = 2
	// Bottom-left: 4*1 + 5*0 + 7*0 + 8*0 = 4
	// Bottom-right: 5*1 + 6*0 + 8*0 + 9*0 = 5
	
	expected := []float64{1, 2, 4, 5}
	for i, val := range output.Data {
		if val != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}
