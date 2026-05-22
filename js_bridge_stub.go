//go:build !js

package main

func (g *Game) initJSBridge() {
	// No-op for standard OS builds
}
