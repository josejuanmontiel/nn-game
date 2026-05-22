package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type ConvLayer struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int

	// Kernels: OutChannels x InChannels x KernelSize x KernelSize
	Kernels [][]*mat.Dense 
	Biases  []float64

	// Buffers for training
	LastInput   *Tensor3D
	LastOutput  *Tensor3D
	gradKernels [][]*mat.Dense
	gradBiases  []float64
}

func NewConvLayer(inC, outC, kSize, stride, padding int) *ConvLayer {
	layer := &ConvLayer{
		InChannels:  inC,
		OutChannels: outC,
		KernelSize:  kSize,
		Stride:      stride,
		Padding:     padding,
		Kernels:     make([][]*mat.Dense, outC),
		Biases:      make([]float64, outC),
	}

	for i := 0; i < outC; i++ {
		layer.Kernels[i] = make([]*mat.Dense, inC)
		for j := 0; j < inC; j++ {
			data := make([]float64, kSize*kSize)
			// Xavier/Glorot Initialization
			r := 1.0 / float64(kSize*kSize*inC)
			for k := range data {
				data[k] = (rand.Float64()*2 - 1) * r
			}
			layer.Kernels[i][j] = mat.NewDense(kSize, kSize, data)
		}
		layer.Biases[i] = (rand.Float64()*2 - 1) * 0.01
	}

	return layer
}

func (l *ConvLayer) Forward(input any) any {
	var in *Tensor3D
	if v, ok := input.(*Tensor3D); ok {
		in = v
	} else if v, ok := input.([]float64); ok {
		// Assuming 8x8 input for shapes level
		in = NewTensor3D(1, 8, 8)
		copy(in.Data, v)
	} else {
		panic(fmt.Sprintf("ConvLayer.Forward: expected *Tensor3D or []float64, got %T", input))
	}

	l.LastInput = in
	
	outH := (in.Height + 2*l.Padding - l.KernelSize) / l.Stride + 1
	outW := (in.Width + 2*l.Padding - l.KernelSize) / l.Stride + 1
	
	output := NewTensor3D(l.OutChannels, outH, outW)
	
	for f := 0; f < l.OutChannels; f++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				sum := l.Biases[f]
				
				for c := 0; c < l.InChannels; c++ {
					kernel := l.Kernels[f][c]
					for ky := 0; ky < l.KernelSize; ky++ {
						for kx := 0; kx < l.KernelSize; kx++ {
							iy := y*l.Stride + ky - l.Padding
							ix := x*l.Stride + kx - l.Padding
							
							if iy >= 0 && iy < in.Height && ix >= 0 && ix < in.Width {
								sum += in.At(c, iy, ix) * kernel.At(ky, kx)
							}
						}
					}
				}
				output.Set(f, y, x, sum)
			}
		}
	}
	
	l.LastOutput = output
	return output
}

func (l *ConvLayer) Backward(gradOutput any) any {
	var gradOut *Tensor3D
	if v, ok := gradOutput.(*Tensor3D); ok {
		gradOut = v
	} else if v, ok := gradOutput.([]float64); ok {
		// Flattened grad coming back from Dense layer
		outH := (l.LastInput.Height + 2*l.Padding - l.KernelSize) / l.Stride + 1
		outW := (l.LastInput.Width + 2*l.Padding - l.KernelSize) / l.Stride + 1
		gradOut = NewTensor3D(l.OutChannels, outH, outW)
		copy(gradOut.Data, v)
	} else {
		panic(fmt.Sprintf("ConvLayer.Backward: expected *Tensor3D or []float64, got %T", gradOutput))
	}
	gradInput := NewTensor3D(l.InChannels, l.LastInput.Height, l.LastInput.Width)
	
	// We store gradients internally to be used during Update
	l.gradKernels = make([][]*mat.Dense, l.OutChannels)
	l.gradBiases = make([]float64, l.OutChannels)

	for f := 0; f < l.OutChannels; f++ {
		l.gradKernels[f] = make([]*mat.Dense, l.InChannels)
		for c := 0; c < l.InChannels; c++ {
			l.gradKernels[f][c] = mat.NewDense(l.KernelSize, l.KernelSize, nil)
		}
	}

	for f := 0; f < l.OutChannels; f++ {
		for y := 0; y < gradOut.Height; y++ {
			for x := 0; x < gradOut.Width; x++ {
				gradOutVal := gradOut.At(f, y, x)
				l.gradBiases[f] += gradOutVal

				for c := 0; c < l.InChannels; c++ {
					for ky := 0; ky < l.KernelSize; ky++ {
						for kx := 0; kx < l.KernelSize; kx++ {
							iy := y*l.Stride + ky - l.Padding
							ix := x*l.Stride + kx - l.Padding

							if iy >= 0 && iy < l.LastInput.Height && ix >= 0 && ix < l.LastInput.Width {
								// Grad wrt Weights
								oldGradK := l.gradKernels[f][c].At(ky, kx)
								l.gradKernels[f][c].Set(ky, kx, oldGradK + l.LastInput.At(c, iy, ix)*gradOutVal)

								// Grad wrt Input
								oldGradIn := gradInput.At(c, iy, ix)
								gradInput.Set(c, iy, ix, oldGradIn + l.Kernels[f][c].At(ky, kx)*gradOutVal)
							}
						}
					}
				}
			}
		}
	}

	return gradInput
}

func (l *ConvLayer) Update(lr float64) {
	for f := 0; f < l.OutChannels; f++ {
		l.Biases[f] -= lr * l.gradBiases[f]
		for c := 0; c < l.InChannels; c++ {
			for r := 0; r < l.KernelSize; r++ {
				for col := 0; col < l.KernelSize; col++ {
					updatedWeight := l.Kernels[f][c].At(r, col) - lr*l.gradKernels[f][c].At(r, col)
					l.Kernels[f][c].Set(r, col, updatedWeight)
				}
			}
		}
	}
	// Clear gradients for next pass
	l.gradKernels = nil
	l.gradBiases = nil
}

func (l *ConvLayer) ResetWeights() {
	inSize := l.InChannels * l.KernelSize * l.KernelSize
	outSize := l.OutChannels * l.KernelSize * l.KernelSize
	r := math.Sqrt(6.0 / float64(inSize+outSize))

	for f := 0; f < l.OutChannels; f++ {
		for c := 0; c < l.InChannels; c++ {
			for i := 0; i < l.KernelSize; i++ {
				for j := 0; j < l.KernelSize; j++ {
					l.Kernels[f][c].Set(i, j, (rand.Float64()*2-1)*r)
				}
			}
		}
		l.Biases[f] = 0
	}
}
