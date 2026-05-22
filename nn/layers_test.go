package nn

import "testing"

func TestPoolLayer_Forward(t *testing.T) {
	// 1 channel 4x4 input
	input := NewTensor3D(1, 4, 4)
	input.Data = []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	// 2x2 MaxPool, stride 2
	pool := NewPoolLayer(2, 2)
	outputRaw := pool.Forward(input)
	output := outputRaw.(*Tensor3D)

	// Expected output size: 2x2
	if output.Height != 2 || output.Width != 2 {
		t.Errorf("Expected 2x2 output, got %dx%d", output.Height, output.Width)
	}

	// Expected values:
	// Top-left block: max(1,2,5,6) = 6
	// Top-right block: max(3,4,7,8) = 8
	// Bottom-left block: max(9,10,13,14) = 14
	// Bottom-right block: max(11,12,15,16) = 16
	
	expected := []float64{6, 8, 14, 16}
	for i, val := range output.Data {
		if val != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], val)
		}
	}
}

func TestFlattenLayer_Forward(t *testing.T) {
	// 2 channels 2x2 input
	input := NewTensor3D(2, 2, 2)
	for i := range input.Data {
		input.Data[i] = float64(i)
	}

	flatten := NewFlattenLayer()
	outputRaw := flatten.Forward(input)
	output := outputRaw.([]float64)

	// Expected 1D vector of size 2*2*2 = 8
	if len(output) != 8 {
		t.Errorf("Expected length 8, got %d", len(output))
	}

	for i, val := range output {
		if val != float64(i) {
			t.Errorf("At index %d: expected %f, got %f", i, float64(i), val)
		}
	}
}
