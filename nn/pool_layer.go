package nn

type PoolLayer struct {
	Size   int
	Stride int
	// MaxIndices stores the 1D index in the input patch for each output element
	MaxIndices []int 
	LastInput  *Tensor3D
}

func NewPoolLayer(size, stride int) *PoolLayer {
	return &PoolLayer{
		Size:   size,
		Stride: stride,
	}
}

func (l *PoolLayer) Forward(input any) any {
	in := input.(*Tensor3D)
	l.LastInput = in
	outH := (in.Height-l.Size)/l.Stride + 1
	outW := (in.Width-l.Size)/l.Stride + 1
	
	output := NewTensor3D(in.Channels, outH, outW)
	l.MaxIndices = make([]int, in.Channels * outH * outW)
	
	for c := 0; c < in.Channels; c++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				maxVal := -1e18
				maxIdx := -1
				
				for py := 0; py < l.Size; py++ {
					for px := 0; px < l.Size; px++ {
						iy := y*l.Stride + py
						ix := x*l.Stride + px
						
						val := in.At(c, iy, ix)
						if val > maxVal {
							maxVal = val
							maxIdx = py*l.Size + px
						}
					}
				}
				output.Set(c, y, x, maxVal)
				l.MaxIndices[c*outH*outW + y*outW + x] = maxIdx
			}
		}
	}
	
	return output
}

func (l *PoolLayer) Backward(gradOutput any) any {
	gradOut := gradOutput.(*Tensor3D)
	gradInput := NewTensor3D(l.LastInput.Channels, l.LastInput.Height, l.LastInput.Width)
	
	outH := gradOut.Height
	outW := gradOut.Width
	
	for c := 0; c < l.LastInput.Channels; c++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				maxIdx := l.MaxIndices[c*outH*outW + y*outW + x]
				py := maxIdx / l.Size
				px := maxIdx % l.Size
				
				iy := y*l.Stride + py
				ix := x*l.Stride + px
				
				val := gradOut.At(c, y, x)
				gradInput.Data[c*gradInput.Height*gradInput.Width + iy*gradInput.Width + ix] += val
			}
		}
	}
	
	return gradInput
}

func (l *PoolLayer) Update(lr float64) {
	// Nothing to update for baseline PoolLayer
}

func (l *PoolLayer) ResetWeights() {
	// Nothing to reset
}
