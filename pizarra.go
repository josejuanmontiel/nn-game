package main

import (
	"fmt"
	"image/color"
	"math"

	"11-juego-final/nn"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func (g *Game) drawTracerFooter(screen *ebiten.Image) {
	// Dynamically calculate split and dimensions
	splitX := float64(g.w) * 0.65
	if g.currentLevel.EngineType == "cnn" {
		splitX = float64(g.w) * 0.45
	}
	h := 240.0
	x, y := splitX, float64(g.h)-h
	w := float64(g.w) - splitX

	if g.currentLevel.EngineType == "cnn" && g.pizarraCollapsed {
		btnX, btnY := int(g.w)-200, int(g.h)-40
		bw, bh := 190, 30
		vector.DrawFilledRect(screen, float32(btnX), float32(btnY), float32(bw), float32(bh), color.RGBA{40, 40, 80, 200}, true)
		vector.StrokeRect(screen, float32(btnX), float32(btnY), float32(bw), float32(bh), 1, color.RGBA{100, 150, 255, 255}, true)
		ebitenutil.DebugPrintAt(screen, ">> ABRIR PIZARRA (Click)", btnX+15, btnY+8)
		return
	}

	// More prominent "Blackboard" look
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h), color.RGBA{30, 30, 50, 250}, true)
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), 2, color.RGBA{200, 200, 255, 255}, true)

	// Header
	headerColor := color.RGBA{50, 50, 100, 255}
	headerText := "PIZARRA DE CALCULO MATEMATICO (Manual)"
	if g.isAuditMode {
		headerColor = color.RGBA{100, 50, 50, 255} // Reddish for Audit/Pause
		headerText = fmt.Sprintf("MODO AUDITORIA: NODO %d (Rueda: Navegar)", g.auditIdx)
	}
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), 30, headerColor, true)
	ebitenutil.DebugPrintAt(screen, headerText, int(x)+10, int(y)+7)



	if g.selLayer <= 0 || g.selLayer > len(g.nn.Layers) {
		ebitenutil.DebugPrintAt(screen, ">>> MUEVE EL RATON SOBRE UN NODO PARA EXAMINAR LAS OPERACIONES <<<", int(x)+50, int(y)+80)
		return
	}

	if g.currentLevel.EngineType == "cnn" {
		g.drawCNNTracerFooter(screen, x, y, w, h)
		return
	}

	// Use the new TraceNode method for future backend abstraction
	trace := g.nn.TraceNode(g.selLayer, g.selNode)

	// CONTENT START OFFSET (avoid header)
	contentY := int(y) + 35

	// --- FORWARD PASS SECTION ---
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("FORWARD: Layer %d | Neuron %d", trace.LayerIdx, trace.NodeIdx), int(x)+10, contentY)

	// Weighted Sum: limit to first 3 inputs if too many
	maxPreview := 3
	if len(trace.Inputs) < maxPreview {
		maxPreview = len(trace.Inputs)
	}

	sumStr := "z = "
	if maxPreview > 0 {
		sumStr += fmt.Sprintf("(%.2f*%.2f)", trace.Weights[0], trace.Inputs[0])
		for j := 1; j < maxPreview; j++ {
			sumStr += fmt.Sprintf(" + (%.2f*%.2f)", trace.Weights[j], trace.Inputs[j])
		}
	} else {
		sumStr += "0"
	}
	if len(trace.Inputs) > maxPreview {
		sumStr += " + ..."
	}
	biasStr := fmt.Sprintf("+ Bias(%.2f)", trace.Bias)

	ebitenutil.DebugPrintAt(screen, sumStr, int(x)+20, contentY+20)
	ebitenutil.DebugPrintAt(screen, biasStr, int(x)+20, contentY+35)

	actStr := fmt.Sprintf("%s(z:%.2f) = Out: %.3f", g.actName, trace.Z, trace.Output)
	ebitenutil.DebugPrintAt(screen, actStr, int(x)+20, contentY+55)

	// --- BACKWARD PASS SECTION ---
	backwardY := contentY + 85
	vector.StrokeLine(screen, float32(x+10), float32(backwardY-5), float32(x+w-10), float32(backwardY-5), 1, color.RGBA{100, 100, 150, 100}, true)
	ebitenutil.DebugPrintAt(screen, "BACKWARD (Learning Mode - Scroll here to Step!)", int(x)+10, backwardY)

	delta := trace.Delta
	der := g.nn.Derivative(trace.Output) // Approximate f'(z)

	deltaStr := fmt.Sprintf("Delta (d): %.4f (Error Signal)", delta)
	derStr := fmt.Sprintf("f'(z): %.4f (Gradiente de Activacion)", der)
	ebitenutil.DebugPrintAt(screen, deltaStr, int(x)+20, backwardY+20)
	ebitenutil.DebugPrintAt(screen, derStr, int(x)+20, backwardY+35)

	gradStr := "Gradiente Peso[0] = d * Entrada[0] = "
	if len(trace.Inputs) > 0 {
		gradStr += fmt.Sprintf("%.4f", trace.Gradient[0])
	}
	ebitenutil.DebugPrintAt(screen, gradStr, int(x)+20, backwardY+55)

	if g.isAuditMode {
		ebitenutil.DebugPrintAt(screen, "WHEEL: NAVEGAR NODOS | ZONA DE CALCULO", int(x)+10, backwardY+80)
	}

	// (HUD call moved to main Draw loop for Level 0)
}

func (g *Game) drawCNNTracerFooter(screen *ebiten.Image, x, y, w, h float64) {
	// Close Button for Pizarra
	btnX, btnY := int(x)+int(w)-140, int(y)+10
	vector.DrawFilledRect(screen, float32(btnX), float32(btnY), 130, 25, color.RGBA{80, 30, 30, 200}, true)
	vector.StrokeRect(screen, float32(btnX), float32(btnY), 130, 25, 1, color.RGBA{255, 100, 100, 255}, true)
	ebitenutil.DebugPrintAt(screen, "<< MINIMIZAR", btnX+25, btnY+5)

	contentY := int(y) + 35
	statusText := ""
	if !g.isProbing {
		statusText = " [FOCO DE ATENCION: MAXIMA ACTIVACION]"
	}
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("CNN INSPECTOR: Layer %d | Node/Filter %d%s", g.selLayer, g.selNode, statusText), int(x)+10, contentY)

	if len(g.nn.LayersV2) > 0 {
		if conv, ok := g.nn.LayersV2[0].(*nn.ConvLayer); ok {
			if g.selNode >= 0 && g.selNode < len(conv.Kernels) {
				kernel := conv.Kernels[g.selNode][0] // assuming 1 input channel for simplicity
				
				// Draw Kernel
				kx := int(x) + 20
				ky := contentY + 30
				ebitenutil.DebugPrintAt(screen, "Filtro 3x3 (Kernel):", kx, ky-15)
				
				kSize := 20.0
				for r := 0; r < 3; r++ {
					for c := 0; c < 3; c++ {
						w := kernel.At(r, c)
						alpha := uint8(math.Min(255, math.Abs(w)*200))
						col := color.RGBA{0, alpha, 255, 255} // Blue for positive
						if w < 0 {
							col = color.RGBA{255, 0, alpha, 255} // Red for negative
						}
						vector.DrawFilledRect(screen, float32(kx)+float32(c)*float32(kSize), float32(ky)+float32(r)*float32(kSize), float32(kSize)-1, float32(kSize)-1, col, true)
						// Val text
						ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%+1.1f", w), kx+c*int(kSize)+1, ky+r*int(kSize)+2)
					}
				}

				// Draw Feature Map
				if conv.LastOutput != nil && conv.LastOutput.Width > 0 {
					fx := kx + 120
					fy := ky - 15
					ebitenutil.DebugPrintAt(screen, "Mapa de Activacion (Toda la imagen):", fx, fy)
					fy += 15
					
					fSize := 12.0
					outMat := conv.LastOutput.ToMatrix(g.selNode)
					rows, cols := outMat.Dims()
					for r := 0; r < rows; r++ {
						for c := 0; c < cols; c++ {
							val := math.Max(0, outMat.At(r, c)) // ReLU preview
							v := uint8(math.Min(255, val*127))
							col := color.RGBA{v, v, v, 255}
							
							// Highlight if we are hovering exactly this pixel in 3D
							if r == g.scannedRow && c == g.scannedCol {
								col = color.RGBA{255, 255, 0, 255}
							}
							
							vector.DrawFilledRect(screen, float32(fx)+float32(c)*float32(fSize), float32(fy)+float32(r)*float32(fSize), float32(fSize)-0.5, float32(fSize)-0.5, col, true)
						}
					}
					
					descY := fy + int(float64(rows)*fSize) + 15
					ebitenutil.DebugPrintAt(screen, "Observa como este filtro resalta ciertas formas", fx, descY)
					ebitenutil.DebugPrintAt(screen, "y apaga otras en toda la imagen simultaneamente.", fx, descY+12)
				}
			}
		}
	} else {
		ebitenutil.DebugPrintAt(screen, "Exploracion detallada disponible en la primera capa Conv.", int(x)+20, contentY+30)
	}
}

func (g *Game) drawActivationHUD(screen *ebiten.Image) {
	padding := 10.0
	w, h := 160.0, 50.0
	x := float64(g.w) - w - padding
	y := float64(g.h) - h - padding

	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h), color.RGBA{20, 20, 40, 220}, true)
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), 1, color.RGBA{100, 100, 200, 255}, true)

	ebitenutil.DebugPrintAt(screen, "ACTIVATION FUNCTION", int(x)+10, int(y)+5)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Mode: %s [Keys 1-3]", g.actName), int(x)+10, int(y)+20)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("f(x) = %s", g.actFormula), int(x)+10, int(y)+33)
}
