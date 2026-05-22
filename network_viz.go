package main

import (
	"11-juego-final/nn"
	"fmt"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"gonum.org/v1/gonum/mat"
)

func (g *Game) handleNetworkInteraction() {
	mx, my := ebiten.CursorPosition()
	fmx, fmy := float64(mx), float64(my)

	splitX := float64(g.w) * 0.65
	// Only interact if cursor is in the right panel
	if fmx < splitX {
		return
	}

	layerSpacing := 65.0
	nodeSpacing := 35.0

	g.selLayer = -1
	g.selNode = -1
	g.selPrev = -1

	// Hover check
	for l := 0; l <= len(g.nn.Layers); l++ {
		ly := g.netY + (float64(l)-float64(len(g.nn.Layers))/2.0)*layerSpacing*g.netZoom
		numNodes := 0
		if l == 0 {
			numNodes = 2 // input
		} else {
			numNodes = len(g.nn.Layers[l-1].Outputs)
		}
		offsetX := g.netX - (float64(numNodes)-1.0)*nodeSpacing*g.netZoom/2.0

		for i := 0; i < numNodes; i++ {
			nx, ny := offsetX+float64(i)*nodeSpacing*g.netZoom, ly
			dist := math.Sqrt(math.Pow(nx-fmx, 2) + math.Pow(ny-fmy, 2))
			if dist < 12*g.netZoom {
				g.selLayer = l
				g.selNode = i
				return
			}
		}
	}

	for l := 1; l <= len(g.nn.Layers); l++ {
		ly := g.netY + (float64(l)-float64(len(g.nn.Layers))/2.0)*layerSpacing*g.netZoom
		numNodes := len(g.nn.Layers[l-1].Outputs)
		offsetX := g.netX - (float64(numNodes)-1.0)*nodeSpacing*g.netZoom/2.0

		prevLY := ly - layerSpacing*g.netZoom
		var prevNumNodes int
		if l == 1 {
			prevNumNodes = 2
		} else {
			prevNumNodes = len(g.nn.Layers[l-2].Outputs)
		}
		prevOffsetX := g.netX - (float64(prevNumNodes)-1.0)*nodeSpacing*g.netZoom/2.0

		for i := 0; i < numNodes; i++ {
			for j := 0; j < prevNumNodes; j++ {
				cx := ((prevOffsetX + float64(j)*nodeSpacing*g.netZoom) + (offsetX + float64(i)*nodeSpacing*g.netZoom)) / 2
				cy := (prevLY + ly) / 2
				dist := math.Sqrt(math.Pow(cx-fmx, 2) + math.Pow(cy-fmy, 2))
				if dist < 8*g.netZoom {
					g.selLayer = l
					g.selNode = i
					g.selPrev = j
					return
				}
			}
		}
	}
}

func (g *Game) applyAdjustment() {
	if g.selLayer <= 0 {
		return
	}

	diff := 0.0
	// Use Arrow keys or Mouse Wheel
	if ebiten.IsKeyPressed(ebiten.KeyUp) || ebiten.IsKeyPressed(ebiten.KeyArrowUp) {
		diff = 0.02
	}
	if ebiten.IsKeyPressed(ebiten.KeyDown) || ebiten.IsKeyPressed(ebiten.KeyArrowDown) {
		diff = -0.02
	}

	_, wy := ebiten.Wheel()
	if wy != 0 {
		diff = wy * 0.05
	}

	if diff == 0 {
		return
	}

	layer := g.nn.Layers[g.selLayer-1]
	if g.selPrev == -1 {
		if g.selNode < len(layer.Biases) {
			layer.Biases[g.selNode] += diff
			msg := fmt.Sprintf("Adjusted Bias L%d N%d: %.3f", g.selLayer, g.selNode, layer.Biases[g.selNode])
			g.mode = msg
			if inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) || inpututil.IsKeyJustPressed(ebiten.KeyArrowDown) {
				g.addLog(msg)
				g.saveSnapshot()
			}
		}
	} else {
		if g.selNode < len(layer.Weights) && g.selPrev < len(layer.Weights[g.selNode]) {
			layer.Weights[g.selNode][g.selPrev] += diff
			msg := fmt.Sprintf("Adjusted Weight L%d N%d<-N%d: %.3f", g.selLayer, g.selNode, g.selPrev, layer.Weights[g.selNode][g.selPrev])
			g.mode = msg
			if inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) || inpututil.IsKeyJustPressed(ebiten.KeyArrowDown) {
				g.addLog(msg)
				g.saveSnapshot()
			}
		}
	}
}

func (g *Game) drawNetwork(screen *ebiten.Image) {
	if g.currentLevel.EngineType == "cnn" {
		g.drawCNNExplainer(screen)
		return
	}

	splitX := float64(g.w) * 0.65
	// Screen mask for right panel
	vector.DrawFilledRect(screen, float32(splitX), 0, float32(g.w-int(splitX)), float32(g.h), color.RGBA{10, 10, 20, 255}, true)
	vector.StrokeRect(screen, float32(splitX), 0, float32(g.w-int(splitX)), float32(g.h), 2, color.RGBA{100, 100, 100, 255}, true)
	ebitenutil.DebugPrintAt(screen, "NETWORK ARCHITECTURE (DRAG & WHEEL)", int(splitX)+10, 10)

	layerSpacing := 65.0
	nodeSpacing := 35.0

	// Draw weights first
	for l := 1; l <= len(g.nn.Layers); l++ {
		ly := g.netY + (float64(l)-float64(len(g.nn.Layers))/2.0)*layerSpacing*g.netZoom
		numNodes := len(g.nn.Layers[l-1].Outputs)
		offsetX := g.netX - (float64(numNodes)-1.0)*nodeSpacing*g.netZoom/2.0

		prevLY := ly - layerSpacing*g.netZoom
		var prevNumNodes int
		if l == 1 {
			prevNumNodes = 2
		} else {
			prevNumNodes = len(g.nn.Layers[l-2].Outputs)
		}
		prevOffsetX := g.netX - (float64(prevNumNodes)-1.0)*nodeSpacing*g.netZoom/2.0
		layer := g.nn.Layers[l-1]
		if len(layer.Weights) == 0 {
			continue
		}
		for i := 0; i < numNodes; i++ {
			nx, ny := offsetX+float64(i)*nodeSpacing*g.netZoom, ly
			for j := 0; j < prevNumNodes; j++ {
				pnx, pny := prevOffsetX+float64(j)*nodeSpacing*g.netZoom, prevLY
				w := layer.Weights[i][j]
				c := color.RGBA{100, 100, 255, 100}
				if w < 0 {
					c = color.RGBA{255, 100, 100, 100}
				}
				thickness := float32(math.Abs(w) * 2 * g.netZoom)
				if thickness < 0.5 {
					thickness = 0.5
				}
				if thickness > 5 {
					thickness = 5
				}

				// Audit/Selection Highlight
				if g.selLayer == l && g.selNode == i {
					c.A = 255 // Full opacity for active node connections
					thickness *= 1.5
				}

				if g.selLayer == l && g.selNode == i && g.selPrev == j {
					c = color.RGBA{255, 255, 0, 255}
					thickness *= 1.5
				}
				// clip to right panel
				if pnx > splitX && nx > splitX {
					vector.StrokeLine(screen, float32(pnx), float32(pny), float32(nx), float32(ny), thickness, c, true)
				}
			}
		}
	}

	// Draw nodes
	for l := 0; l <= len(g.nn.Layers); l++ {
		ly := g.netY + (float64(l)-float64(len(g.nn.Layers))/2.0)*layerSpacing*g.netZoom
		numNodes := 0
		if l == 0 {
			numNodes = 2 // input
		} else {
			numNodes = len(g.nn.Layers[l-1].Outputs)
		}
		offsetX := g.netX - (float64(numNodes)-1.0)*nodeSpacing*g.netZoom/2.0

		for i := 0; i < numNodes; i++ {
			nx, ny := offsetX+float64(i)*nodeSpacing*g.netZoom, ly
			if nx < splitX {
				continue
			}
			c := color.RGBA{150, 150, 200, 255}
			size := float32(5 * g.netZoom)

			if l > 0 {
				out := g.nn.Layers[l-1].Outputs[i]
				val := uint8(math.Max(0, math.Min(255, 127+out*127)))
				c = color.RGBA{val, val, 255, 255}
			}

			// SPECIAL CNN NODE VISUALIZATION: Show Feature Maps in Nodes
			isCNNNode := g.currentLevel.EngineType == "cnn" && l == 1 // Conv Layer
			var outMat *mat.Dense
			if isCNNNode && len(g.nn.LayersV2) > 0 {
				if conv, ok := g.nn.LayersV2[0].(*nn.ConvLayer); ok {
					if conv.LastOutput != nil && i < conv.OutChannels {
						// Render the spatial activation (feature map) as the node itself
						outMat = conv.LastOutput.ToMatrix(i)
					}
				}
			}

			if outMat != nil {
				// We display the node's activation purely based on the patch currently being hovered by the user.
				var val float64
				if g.isProbing {
					rows, cols := outMat.Dims()
					r, c := g.scannedRow, g.scannedCol
					if r >= 0 && c >= 0 && r < rows && c < cols {
						val = math.Max(0, outMat.At(r, c)) // ReLU 
					}
				}

				// If not probing, it stays mostly dark or we could show average. 
				// We'll just show the specific value of the scanned patch to be didactic.
				v := uint8(math.Min(255, val*127))
				
				// Use a distinctive cyan/blue tint to differentiate Conv nodes from Dense nodes
				c = color.RGBA{v/3, v, v, 255} 

				// Draw slightly larger circle for Conv nodes
				vector.DrawFilledCircle(screen, float32(nx), float32(ny), size*1.5, c, true)
				vector.StrokeCircle(screen, float32(nx), float32(ny), size*1.5, 1, color.RGBA{150, 150, 255, 200}, true)
			} else {
				vector.DrawFilledCircle(screen, float32(nx), float32(ny), size, c, true)
			}

			if g.selLayer == l && g.selNode == i && g.selPrev == -1 {
				c = color.RGBA{255, 255, 200, 255}
				size = float32(10 * g.netZoom)
				// Glow/Aura for Audit Mode
				if g.isAuditMode {
					vector.DrawFilledCircle(screen, float32(nx), float32(ny), size*1.4, color.RGBA{255, 255, 100, 80}, true)
				}
				// Display value with dark background for contrast
				if l > 0 {
					// Use TraceNode for mathematical visualization
					trace := g.nn.TraceNode(l, i)
					valStr := fmt.Sprintf("Bias: %.3f", trace.Bias)

					traceStr := fmt.Sprintf("Σ(w*x) %.2f + b %.2f = z %.2f", trace.Z-trace.Bias, trace.Bias, trace.Z)
					actStr := fmt.Sprintf("%s(z) = %.2f", g.actName, trace.Output)

					tx, ty := int(nx)-30, int(ny)+20 // Move text BELOW node
					if ty > g.h-100 {
						ty = int(ny) - 60 // Flip if too low
					}
					// Ensure horizontal safety
					if tx+185 > g.w-10 {
						tx = g.w - 185 - 10
					}

					vector.DrawFilledRect(screen, float32(tx)-2, float32(ty)-2, 185, 45, color.RGBA{0, 0, 0, 220}, true)
					ebitenutil.DebugPrintAt(screen, valStr, tx, ty)
					ebitenutil.DebugPrintAt(screen, traceStr, tx, ty+15)
					ebitenutil.DebugPrintAt(screen, actStr, tx, ty+30)
				} else {
					tx, ty := int(nx)-20, int(ny)-20
					ebitenutil.DebugPrintAt(screen, "Input", tx, ty)
				}
			}
			vector.DrawFilledCircle(screen, float32(nx), float32(ny), size, c, true)
		}
	}

	if g.selLayer > 0 && g.selLayer <= len(g.nn.Layers) && g.selPrev != -1 {
		layer := g.nn.Layers[g.selLayer-1]
		if len(layer.Weights) > 0 {
			if g.selNode < len(layer.Weights) && g.selPrev < len(layer.Weights[g.selNode]) {
				val := layer.Weights[g.selNode][g.selPrev]
				mx, my := ebiten.CursorPosition()
				valStr := fmt.Sprintf("Weight: %.3f", val)
				vector.DrawFilledRect(screen, float32(mx+10)-2, float32(my+10)-2, float32(len(valStr)*6)+4, 12, color.RGBA{0, 0, 0, 200}, true)
				ebitenutil.DebugPrintAt(screen, valStr, mx+10, my+10)
			}
		}
	}
}

func (g *Game) drawCNNExplainer(screen *ebiten.Image) {
	splitX := float64(g.w) * 0.45 // Expandimos hacia la izquierda
	vector.DrawFilledRect(screen, float32(splitX), 0, float32(g.w-int(splitX)), float32(g.h), color.RGBA{15, 15, 25, 255}, true)
	vector.StrokeRect(screen, float32(splitX), 0, float32(g.w-int(splitX)), float32(g.h), 2, color.RGBA{100, 100, 100, 255}, true)
	ebitenutil.DebugPrintAt(screen, "CNN EXPLAINER VISUALIZATION (Aharley Style)", int(splitX)+10, 10)

	if len(g.nn.LayersV2) == 0 {
		return
	}

	// Detemine inputs
	var input []float64
	if len(g.convInput) >= 64 {
		input = g.convInput
	} else if g.starfield != nil && len(g.starfield.Stars) > 0 {
		input = g.starfield.Stars[0].RawData
	}

	if len(input) < 64 {
		ebitenutil.DebugPrintAt(screen, "Esperando datos... Juega con el ratón o espera el Auto-Train.", int(splitX)+50, 50)
		return
	}
	
	// Actualizar outputs si no estamos flotando activamente
	if !g.isProbing {
		g.nn.GetV2LayerOutputs(input)
	}

	baseCx := splitX + 160.0
	baseCy := float64(g.h) * 0.7

	var conv *nn.ConvLayer
	if c, ok := g.nn.LayersV2[0].(*nn.ConvLayer); ok {
		conv = c
	}

	mx, my := ebiten.CursorPosition()
	fmx, fmy := float64(mx), float64(my)
	
	// Funciones matemáticas para proyectar y des-proyectar
	getIsoScreenPos := func(r, c float64, cx, cy float64, gridW, gridH, cellSize float64) (float64, float64) {
		x := c*cellSize - gridW/2
		y := r*cellSize - gridH/2
		cos := math.Cos(-math.Pi / 4)
		sin := math.Sin(-math.Pi / 4)
		xr := x*cos - y*sin
		yr := x*sin + y*cos
		return cx + xr, cy + yr*0.5
	}

	getIsoCell := func(mx, my float64, cx, cy float64, gridW, gridH, cellSize float64) (int, int) {
		x := mx - cx
		y := (my - cy) * 2.0 // Invert scale Y
		cos := math.Cos(math.Pi / 4)
		sin := math.Sin(math.Pi / 4)
		xr := x*cos - y*sin
		yr := x*sin + y*cos
		xr += gridW / 2
		yr += gridH / 2
		return int(math.Floor(yr / cellSize)), int(math.Floor(xr / cellSize))
	}

	hoverChannel := -1
	hoverR := -1
	hoverC := -1

	outCellSize := 14.0
	stackSpacing := 50.0
	outCx := baseCx + 220.0
	
	// Detectar hover sobre un feature map
	if conv != nil && conv.LastOutput != nil && fmx > splitX {
		outRows, outCols := conv.LastOutput.Height, conv.LastOutput.Width
		gridW := float64(outCols) * outCellSize
		gridH := float64(outRows) * outCellSize
		
		for i := conv.OutChannels - 1; i >= 0; i-- { // De arriba a abajo visualmente
			cy := baseCy - float64(i)*stackSpacing
			hr, hc := getIsoCell(fmx, fmy, outCx, cy, gridW, gridH, outCellSize)
			if hr >= 0 && hc >= 0 && hr < outRows && hc < outCols {
				hoverChannel = i
				hoverR = hr
				hoverC = hc
				
				// Actualizar solo si no estamos navegando con la rueda manualmente
				if !g.isAuditMode {
					g.selLayer = 1
					g.selNode = i
					g.scannedRow = hr
					g.scannedCol = hc
				}
				
				break // Stop on the top-most intercepted layer
			}
		}
	}

	// Si no hemos interceptado un celda 3D con el ratón, usamos el Foco de Atención
	if hoverChannel == -1 && conv != nil && conv.LastOutput != nil {
		if g.isAuditMode {
			// MANUAL AUDIT / SCROLL MODE: Force frustum to user's wheel-selected node
			hoverChannel = g.selNode
			if hoverChannel < 0 {
				hoverChannel = 0
			}
			if hoverChannel >= conv.OutChannels {
				hoverChannel = conv.OutChannels - 1
			}

			// Local focus: Find highest activation within this specific filter
			maxVal := -99999.0
			outMat := conv.LastOutput.ToMatrix(hoverChannel)
			rows, cols := outMat.Dims()
			simHr, simHc := g.scannedRow, g.scannedCol
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					if val := outMat.At(r, c); val > maxVal {
						maxVal = val
						simHr = r
						simHc = c
					}
				}
			}
			hoverR = simHr
			hoverC = simHc
			g.scannedRow = simHr
			g.scannedCol = simHc

		} else {
			// AUTOMATIC MODE: Global Attention Focus across all filters
			maxVal := -99999.0
			simHr, simHc, simChannel := 0, 0, 0

			for i := 0; i < conv.OutChannels; i++ {
				outMat := conv.LastOutput.ToMatrix(i)
				rows, cols := outMat.Dims()
				for r := 0; r < rows; r++ {
					for c := 0; c < cols; c++ {
						if val := outMat.At(r, c); val > maxVal {
							maxVal = val
							simHr = r
							simHc = c
							simChannel = i
						}
					}
				}
			}

			hoverChannel = simChannel
			hoverR = simHr
			hoverC = simHc

			g.selLayer = 1
			g.selNode = simChannel
			g.scannedRow = simHr
			g.scannedCol = simHc
		}
	}

	// 1. Dibujar Helper Functions
	drawIsoGrid := func(data *mat.Dense, cx, cy float64, cellSize float64, r, gCol, b uint8, label string, isHovered bool, isInput bool) {
		rows, cols := data.Dims()
		gridW := float64(cols) * cellSize
		gridH := float64(rows) * cellSize

		tmp := ebiten.NewImage(int(gridW), int(gridH))
		
		for rr := 0; rr < rows; rr++ {
			for cc := 0; cc < cols; cc++ {
				val := data.At(rr, cc)
				if val < 0 && !isInput { val = 0 } // relu solo aparente
				
				v := uint8(math.Min(255, math.Abs(val)*127 + 40))
				if isInput && val < 0.1 { v = 20 }
				
				c := color.RGBA{uint8(float64(r)*float64(v)/255), uint8(float64(gCol)*float64(v)/255), uint8(float64(b)*float64(v)/255), 255}
				
				// Highlight target cell if hovered
				if isHovered && rr == hoverR && cc == hoverC {
					c = color.RGBA{255, 255, 0, 255} // Amarillo puro
				}

				// Si es input y hay hover, resaltar el receptivo (3x3)
				if isInput && hoverChannel != -1 {
					if rr >= hoverR && rr < hoverR+conv.KernelSize && cc >= hoverC && cc < hoverC+conv.KernelSize {
						alpha := c.A
						c = color.RGBA{255, 255, 0, alpha} // Tinte amarillo
					}
				}
				
				// Reducir un píxel para la grilla
				vector.DrawFilledRect(tmp, float32(cc)*float32(cellSize), float32(rr)*float32(cellSize), float32(cellSize)-1, float32(cellSize)-1, c, true)
			}
		}

		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(-gridW/2, -gridH/2)
		op.GeoM.Rotate(-math.Pi / 4)
		op.GeoM.Scale(1.0, 0.5)
		op.GeoM.Translate(cx, cy)
		screen.DrawImage(tmp, op)
		
		tx, ty := getIsoScreenPos(float64(rows), 0, cx, cy, gridW, gridH, cellSize)
		ebitenutil.DebugPrintAt(screen, label, int(tx)-20, int(ty)+10)
	}

	// 2. Dibujar Feature Maps (Output Conv1)
	if conv != nil && conv.LastOutput != nil {
		for i := 0; i < conv.OutChannels; i++ {
			outMat := conv.LastOutput.ToMatrix(i)
			cy := baseCy - float64(i)*stackSpacing
			isChannelHovered := (hoverChannel == i)
			drawIsoGrid(outMat, outCx, cy, outCellSize, 100, 200, 255, fmt.Sprintf("Filtro %d", i), isChannelHovered, false)
		}
	}

	// 3. Dibujar Input Layer
	inMat := mat.NewDense(8, 8, input)
	inCellSize := 20.0
	drawIsoGrid(inMat, baseCx, baseCy, inCellSize, 255, 255, 255, "Input(8x8)", false, true)

	// 4. Dibujar Conexiones (Frustum de Convolución)
	if hoverChannel != -1 && conv != nil {
		inGridW := 8.0 * inCellSize
		inGridH := 8.0 * inCellSize
		
		outRows, outCols := conv.LastOutput.Height, conv.LastOutput.Width
		outGridW := float64(outCols) * outCellSize
		outGridH := float64(outRows) * outCellSize
		
		cy := baseCy - float64(hoverChannel)*stackSpacing
		
		// Coordenadas del píxel objetivo
		tx, ty := getIsoScreenPos(float64(hoverR)+0.5, float64(hoverC)+0.5, outCx, cy, outGridW, outGridH, outCellSize)
		vector.DrawFilledCircle(screen, float32(tx), float32(ty), 4, color.RGBA{255, 255, 0, 255}, true)
		
		// Conectar con las 4 esquinas del Sliding Window en Input
		inR := float64(hoverR)
		inC := float64(hoverC)
		kSize := float64(conv.KernelSize)
		
		p1x, p1y := getIsoScreenPos(inR, inC, baseCx, baseCy, inGridW, inGridH, inCellSize)
		p2x, p2y := getIsoScreenPos(inR, inC+kSize, baseCx, baseCy, inGridW, inGridH, inCellSize)
		p3x, p3y := getIsoScreenPos(inR+kSize, inC+kSize, baseCx, baseCy, inGridW, inGridH, inCellSize)
		p4x, p4y := getIsoScreenPos(inR+kSize, inC, baseCx, baseCy, inGridW, inGridH, inCellSize)
		
		cLine := color.RGBA{255, 255, 0, 100}
		vector.StrokeLine(screen, float32(p1x), float32(p1y), float32(tx), float32(ty), 1, cLine, true)
		vector.StrokeLine(screen, float32(p2x), float32(p2y), float32(tx), float32(ty), 1, cLine, true)
		vector.StrokeLine(screen, float32(p3x), float32(p3y), float32(tx), float32(ty), 1, cLine, true)
		vector.StrokeLine(screen, float32(p4x), float32(p4y), float32(tx), float32(ty), 1, cLine, true)
		
		// Dibujar borde alrededor del Sliding Window en 3D
		cBorder := color.RGBA{255, 255, 0, 255}
		vector.StrokeLine(screen, float32(p1x), float32(p1y), float32(p2x), float32(p2y), 2, cBorder, true)
		vector.StrokeLine(screen, float32(p2x), float32(p2y), float32(p3x), float32(p3y), 2, cBorder, true)
		vector.StrokeLine(screen, float32(p3x), float32(p3y), float32(p4x), float32(p4y), 2, cBorder, true)
		vector.StrokeLine(screen, float32(p4x), float32(p4y), float32(p1x), float32(p1y), 2, cBorder, true)
	}

	// Dense Layers Visualization (Flattened)
	denseStartX := outCx + 160.0
	ebitenutil.DebugPrintAt(screen, "Flatten & Dense -> Clases", int(denseStartX)-40, int(baseCy)-100)
	
	for l := 1; l < len(g.nn.Layers); l++ {
		nodes := len(g.nn.Layers[l-1].Outputs)
		lx := denseStartX + float64(l-1)*80.0
		
		// Mapear los círculos de la red densa
		nodeSpacing := 14.0
		offsetY := baseCy - (float64(nodes)-1.0)*nodeSpacing/2.0
		
		for i := 0; i < nodes; i++ {
			ny := offsetY + float64(i)*nodeSpacing
			v := uint8(math.Min(255, math.Abs(g.nn.Layers[l-1].Outputs[i])*127 + 50))
			c := color.RGBA{v, v, v, 255}
			
			// Highlight final class
			if l == len(g.nn.Layers)-1 {
				if i == 0 { c = color.RGBA{v, v, v, 255} } // Class 0 (X)
				if i == 1 { c = color.RGBA{v, v, 255, 255} } // Class 1 (O)
			}
			
			vector.DrawFilledCircle(screen, float32(lx), float32(ny), 5, c, true)
		}
	}
}
