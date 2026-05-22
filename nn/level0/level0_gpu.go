//go:build gpu

package level0

/*
#cgo LDFLAGS: -lze_loader
#cgo CXXFLAGS: -I.
#include "matrix_ze.h"
*/
import "C"
import (
	"11-juego-final/nn"
	"11-juego-final/nn/standard"
	"unsafe"
)

type Engine struct {
	standard.Engine
}

func (e *Engine) Forward(layer *nn.Layer, activation nn.ActivationFunc) {
	// For small matrices (neurons < 16), GPU overhead is too high.
	// We'll use standard for small layers and GPU for larger ones as a demo.
	if len(layer.Outputs) < 16 {
		e.Engine.Forward(layer, activation)
		return
	}

	// This is a simplified demo of GPU MatMul integration.
	// In a real implementation, we'd handle non-square matrices and USM persistence better.
	N := len(layer.Outputs)
	
	// Prepare flattened inputs/weights if needed, or just call the C++ function
	// For this starfield game, layers are tiny (4-32 nodes), 
	// so MatMul on GPU is actually slower than CPU!
	// But we implement it for the "wow" factor and future scalability.
	
	// Dummy call to demonstrate linkage
	C.ze_matrix_multiply((*C.float)(unsafe.Pointer(&layer.Inputs[0])), 
		(*C.float)(unsafe.Pointer(&layer.Weights[0][0])), 
		(*C.float)(unsafe.Pointer(&layer.Outputs[0])), 
		C.int(N))
}

func init() {
	C.ze_init()
}
