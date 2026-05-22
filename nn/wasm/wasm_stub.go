//go:build !js

package wasm

import (
	"11-juego-final/nn/standard"
)

// Engine is a fallback to standard engine for non-JS platforms
type Engine struct {
	standard.Engine
}
