package nn

import (
	"gonum.org/v1/gonum/mat"
)

// Tensor3D represents a 3D volume (Channels, Height, Width)
type Tensor3D struct {
	Channels int
	Height   int
	Width    int
	Data     []float64
}

func NewTensor3D(c, h, w int) *Tensor3D {
	return &Tensor3D{
		Channels: c,
		Height:   h,
		Width:    w,
		Data:     make([]float64, c*h*w),
	}
}

// At returns the value at (c, y, x)
func (t *Tensor3D) At(c, y, x int) float64 {
	return t.Data[c*t.Height*t.Width+y*t.Width+x]
}

// Set sets the value at (c, y, x)
func (t *Tensor3D) Set(c, y, x int, val float64) {
	t.Data[c*t.Height*t.Width+y*t.Width+x] = val
}

// ToMatrix returns a Gonum matrix representation of a single channel
func (t *Tensor3D) ToMatrix(c int) *mat.Dense {
	offset := c * t.Height * t.Width
	return mat.NewDense(t.Height, t.Width, t.Data[offset : offset+t.Height*t.Width])
}
