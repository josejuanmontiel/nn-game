package game

import (
	"fmt"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func DrawParallelCoordinates(screen *ebiten.Image, wg *WarpGrid, highlight []float64) {
	// Draw 5 vertical axes on the Right Panel (or overlay)
	// Let's use the right half of the screen
	startX := 50.0
	endX := float64(wg.width) - 50.0
	startY := 100.0
	endY := float64(wg.height) - 150.0
	
	numNodes := 0
	if wg.NN != nil && len(wg.NN.Layers) > 0 {
		numNodes = len(wg.NN.Layers[len(wg.NN.Layers)-1].Outputs)
	}
	if numNodes < 2 { numNodes = 2 }
	
	dist := (endX - startX) / float64(numNodes-1)

	// Draw Axes
	for i := 0; i < numNodes; i++ {
		x := startX + float64(i)*dist
		vector.StrokeLine(screen, float32(x), float32(startY), float32(x), float32(endY), 1, color.RGBA{100, 100, 100, 255}, true)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("H%d", i), int(x)-5, int(startY)-20)
	}

	// Draw Grid Lines (Parallel Coordinates)
	for r := 0; r <= wg.rows; r += 2 { // Sample every 2 to maintain perf
		for c := 0; c <= wg.cols; c += 2 {
			inputX := (float64(c)/float64(wg.cols) - 0.5) * 3.0
			inputY := (float64(r)/float64(wg.rows) - 0.5) * 3.0
			hidden := wg.NN.GetLayerOutputs(inputX, inputY, wg.VisibleLayers)

			prevX, prevY := 0.0, 0.0
			for i := 0; i < len(hidden) && i < numNodes; i++ {
				val := hidden[i]
				normY := startY + (1.0-(val+1.0)/2.0)*(endY-startY)
				currX := startX + float64(i)*dist
				
				if i > 0 {
					vector.StrokeLine(screen, float32(prevX), float32(prevY), float32(currX), float32(normY), 1, color.RGBA{0, 255, 100, 30}, true)
				}
				prevX, prevY = currX, normY
			}
		}
	}

	// HIGHLIGHT (e.g. Player)
	if highlight != nil {
		prevX, prevY := 0.0, 0.0
		for i := 0; i < len(highlight) && i < numNodes; i++ {
			val := highlight[i]
			normY := startY + (1.0-(val+1.0)/2.0)*(endY-startY)
			currX := startX + float64(i)*dist
			if i > 0 {
				vector.StrokeLine(screen, float32(prevX), float32(prevY), float32(currX), float32(normY), 3, color.RGBA{255, 255, 0, 255}, true)
			}
			prevX, prevY = currX, normY
		}
	}
}

func DrawSPLOM(screen *ebiten.Image, wg *WarpGrid) {
	// 5x5 Scatter Plot Matrix
	size := 80.0
	padding := 10.0
	startX := 20.0
	startY := 50.0

	numNodes := 0
	if wg.NN != nil && len(wg.NN.Layers) > 0 {
		numNodes = len(wg.NN.Layers[len(wg.NN.Layers)-1].Outputs)
	}
	if numNodes < 1 { numNodes = 1 }

	for i := 0; i < numNodes; i++ {
		for j := 0; j < numNodes; j++ {
			px := startX + float64(j)*(size+padding)
			py := startY + float64(i)*(size+padding)

			// Box
			vector.StrokeRect(screen, float32(px), float32(py), float32(size), float32(size), 1, color.RGBA{60, 60, 60, 255}, true)
			
			// Highlight current 3D axes if any
			// (Simple heuristic: if Projection matches [i][j])
			
			// Sample points
			for r := 0; r <= wg.rows; r += 4 {
				for c := 0; c <= wg.cols; c += 4 {
					inputX := (float64(c)/float64(wg.cols) - 0.5) * 3.0
					inputY := (float64(r)/float64(wg.rows) - 0.5) * 3.0
					hidden := wg.NN.GetLayerOutputs(inputX, inputY, wg.VisibleLayers)
					
					hx := px + (hidden[j]+1.0)/2.0*size
					hy := py + (1.0-(hidden[i]+1.0)/2.0)*size
					screen.Set(int(hx), int(hy), color.RGBA{0, 255, 100, 150})
				}
			}
		}
	}
}

func DrawSchlegel(screen *ebiten.Image, wg *WarpGrid) {
	// Implementation for a 5D hypercube wireframe projection
	// For now, let's draw a centered recursive cube effect as a placeholder
	centerX := float64(wg.width) * 0.5
	centerY := float64(wg.height) * 0.5
	
	ebitenutil.DebugPrintAt(screen, "HI-SPACE TOPOLOGY (Schlegel)", int(centerX)-80, int(centerY)-150)
	
	// Draw a series of nested projected boxes
	for i := 0; i < 5; i++ {
		s := 250.0 - float64(i)*40
		c := uint8(255 - i*40)
		vector.StrokeRect(screen, float32(centerX-s/2), float32(centerY-s/2), float32(s), float32(s), 2, color.RGBA{c, c, 255, 255}, true)
	}
}
