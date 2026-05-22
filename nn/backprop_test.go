package nn

import "testing"

func TestConvLayer_Backprop(t *testing.T) {
	// Simple 3x3 input
	input := NewTensor3D(1, 3, 3)
	for i := range input.Data {
		input.Data[i] = 1.0
	}

	conv := NewConvLayer(1, 1, 2, 1, 0)
	// Initial weights
	w0 := conv.Kernels[0][0].At(0, 0)

	// Forward
	_ = conv.Forward(input)

	// Create a dummy gradient (we want the weights to decrease)
	gradOut := NewTensor3D(1, 2, 2)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0
	}

	// Backward
	gradInputRaw := conv.Backward(gradOut)
	_ = gradInputRaw.(*Tensor3D)

	// Update
	conv.Update(0.1)

	// Verify weight changed
	w1 := conv.Kernels[0][0].At(0, 0)
	if w1 == w0 {
		t.Errorf("Weight did not change after update")
	}
	if w1 > w0 {
		t.Errorf("Weight should have decreased, got %f -> %f", w0, w1)
	}
}

func TestPoolLayer_Backprop(t *testing.T) {
	// 4x4 input where max is in top-left of each 2x2 block
	input := NewTensor3D(1, 4, 4)
	input.Data = []float64{
		10, 0, 10, 0,
		0, 0, 0, 0,
		10, 0, 10, 0,
		0, 0, 0, 0,
	}

	pool := NewPoolLayer(2, 2)
	_ = pool.Forward(input)

	// Gradient coming back: all ones
	gradOut := NewTensor3D(1, 2, 2)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0
	}

	gradInRaw := pool.Backward(gradOut)
	gradIn := gradInRaw.(*Tensor3D)

	// Expected gradIn: 1.0 at max positions (0,0), (0,2), (2,0), (2,2), 0.0 elsewhere
	if gradIn.At(0, 0, 0) != 1.0 || gradIn.At(0, 0, 1) != 0.0 {
		t.Errorf("Gradient flow incorrect in PoolLayer")
	}
}
