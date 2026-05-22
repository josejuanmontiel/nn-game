package game

import (
	"11-juego-final/nn"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

type WarpGrid struct {
	NN            *nn.NeuralNetwork
	GhostNN       *nn.NeuralNetwork
	rows, cols    int
	width, height float64
	Zoom          float64
	CamX, CamY    float64
	Pitch, Yaw    float64 // 3D Perspective
	MorphFactor   float64 // 0 for flat, 1 for warped
	VisibleLayers int     // How many layers to apply
	Mode          VizMode
	Projection    ProjectionMatrix
	Variances     []float64
	Highlight     []float64
}

func NewWarpGrid(network *nn.NeuralNetwork, w, h float64, r, c int) *WarpGrid {
	return &WarpGrid{
		NN:          network,
		rows:        r,
		cols:        c,
		width:       w,
		height:      h,
		Zoom:        1.0,
		CamX:        0,
		CamY:        0,
		Pitch:       0,
		Yaw:         0,
		MorphFactor: 1.0,
		Mode:        VizModeFixed,
		Projection:  IdentityProjection(),
	}
}
func (wg *WarpGrid) UpdateSize(w, h float64) {
	wg.width = w
	wg.height = h
}

func (wg *WarpGrid) UpdateProjection() {
	// Sample grid points to find activity
	var points [][]float64
	numNodes := 0
	for r := 0; r <= wg.rows; r++ {
		for c := 0; c <= wg.cols; c++ {
			inputX := (float64(c)/float64(wg.cols) - 0.5) * 3.0
			inputY := (float64(r)/float64(wg.rows) - 0.5) * 3.0
			hidden := wg.NN.GetLayerOutputs(inputX, inputY, wg.VisibleLayers)
			if numNodes == 0 { numNodes = len(hidden) }
			points = append(points, hidden)
		}
	}

	if wg.Mode == VizModePCA || wg.Mode == VizModeGlyph {
		wg.Projection = ComputePCA(points)
	}

	// MONITOR: Dynamic Variance Detection
	variances := make([]float64, numNodes)
	means := make([]float64, numNodes)
	for _, p := range points {
		for i := 0; i < numNodes && i < len(p); i++ {
			means[i] += p[i]
		}
	}
	for i := range means { means[i] /= float64(len(points)) }
	for _, p := range points {
		for i := 0; i < numNodes && i < len(p); i++ {
			diff := p[i] - means[i]
			variances[i] += diff * diff
		}
	}
	// Normalization not strictly needed for comparison, but good for reporting
	for i := range variances { variances[i] = math.Sqrt(variances[i] / float64(len(points))) }
	wg.Variances = variances

	// If Dim 4 or 5 are significantly more active than visible dimensions, suggest switch
	numVisible := 0
	sumVisible := 0.0
	for i := 0; i < 3 && i < numNodes; i++ {
		sumVisible += variances[i]
		numVisible++
	}
	if numVisible > 0 {
		avgActive := sumVisible / float64(numVisible)
		for i := 3; i < 5 && i < numNodes; i++ {
			if variances[i] > avgActive*2.0 {
				// Potential alert handled in main.go
			}
		}
	}
}

func (wg *WarpGrid) Draw(screen *ebiten.Image) {
	centerX, centerY := wg.width/2, wg.height/2

	// Helper to get color with depth shading
	getDepthColor := func(z float64) color.RGBA {
		// Simple depth factor based on height/z and tilt
		// vz is mapped to alpha
		alpha := 180 + z*100
		if alpha > 255 {
			alpha = 255
		}
		if alpha < 30 {
			alpha = 30
		}
		return color.RGBA{100, 100, 255, uint8(alpha)}
	}

	// Standard Mesh Drawing (Only in Fixed, PCA, or Glyph modes)
	if wg.Mode == VizModeFixed || wg.Mode == VizModePCA || wg.Mode == VizModeGlyph {
		// Draw horizontal lines
		for r := 0; r <= wg.rows; r++ {
			for c := 0; c < wg.cols; c++ {
				x1, y1, z1 := wg.getTransformed(float64(c), float64(r), centerX, centerY)
				x2, y2, _ := wg.getTransformed(float64(c+1), float64(r), centerX, centerY)

				vector.StrokeLine(screen, float32(x1), float32(y1), float32(x2), float32(y2), 1, getDepthColor(z1), true)
			}
		}

		// Draw vertical lines
		for c := 0; c <= wg.cols; c++ {
			for r := 0; r < wg.rows; r++ {
				x1, y1, z1 := wg.getTransformed(float64(c), float64(r), centerX, centerY)
				x2, y2, _ := wg.getTransformed(float64(c), float64(r+1), centerX, centerY)

				vector.StrokeLine(screen, float32(x1), float32(y1), float32(x2), float32(y2), 1, getDepthColor(z1), true)
			}
		}
	}

	// NEW: Draw Glyphs at vertices if mode is Glyph
	if wg.Mode == VizModeGlyph {
		for r := 0; r <= wg.rows; r++ {
			for c := 0; c <= wg.cols; c++ {
				x, y, z := wg.getTransformed(float64(c), float64(r), centerX, centerY)
				
				// Get Dim 4 and 5
				inputX := (float64(c)/float64(wg.cols) - 0.5) * 3.0
				inputY := (float64(r)/float64(wg.rows) - 0.5) * 3.0
				hidden := wg.NN.GetLayerOutputs(inputX, inputY, wg.VisibleLayers)
				
				dim4 := 0.0
				dim5 := 0.0
				if len(hidden) >= 5 {
					dim4 = (hidden[3] + 1.0) / 2.0 // [0, 1]
					dim5 = (hidden[4] + 1.0) / 2.0 // [0, 1]
				}

				// Size controlled by Dim 4 (Spherical context)
				size := float32(2.0 + dim4*8.0)
				// Alpha or Rugosity controlled by Dim 5
				alpha := uint8(100 + dim5*155)
				c := getDepthColor(z)
				c.A = alpha
				
				vector.DrawFilledCircle(screen, float32(x), float32(y), size, c, true)
			}
		}
	}

	// NEW: Draw Ghost Manifold if GhostNN exists (Level 0 Visualization)
	if wg.GhostNN != nil {
		ghostColor := color.RGBA{100, 100, 100, 60} // Very faint grey
		for r := 0; r <= wg.rows; r++ {
			for c := 0; c < wg.cols; c++ {
				x1, y1, _ := wg.getTransformedGhost(float64(c), float64(r), centerX, centerY)
				x2, y2, _ := wg.getTransformedGhost(float64(c+1), float64(r), centerX, centerY)
				vector.StrokeLine(screen, float32(x1), float32(y1), float32(x2), float32(y2), 1, ghostColor, true)
			}
		}
		for c := 0; c <= wg.cols; c++ {
			for r := 0; r < wg.rows; r++ {
				x1, y1, _ := wg.getTransformedGhost(float64(c), float64(r), centerX, centerY)
				x2, y2, _ := wg.getTransformedGhost(float64(c), float64(r+1), centerX, centerY)
				vector.StrokeLine(screen, float32(x1), float32(y1), float32(x2), float32(y2), 1, ghostColor, true)
			}
		}
	}

	// Specialized Visualization Overlays
	switch wg.Mode {
	case VizModeParallel:
		DrawParallelCoordinates(screen, wg, wg.Highlight)
	case VizModeSPLOM:
		DrawSPLOM(screen, wg)
	case VizModeSchlegel:
		DrawSchlegel(screen, wg)
	}
}

func (wg *WarpGrid) getTransformedGhost(gridC, gridR float64, centerX, centerY float64) (float64, float64, float64) {
	inputX := (gridC/float64(wg.cols) - 0.5) * 3.0
	inputY := (gridR/float64(wg.rows) - 0.5) * 3.0
	hidden := wg.GhostNN.GetLayerOutputs(inputX, inputY, wg.VisibleLayers)
	
	var tx, ty, tz float64
	for i := 0; i < 3; i++ {
		val := 0.0
		for j := 0; j < 5 && j < len(hidden); j++ {
			val += wg.Projection[i][j] * hidden[j]
		}
		if i == 0 { tx = val } else if i == 1 { ty = val } else { tz = val }
	}

	mx := inputX*(1-wg.MorphFactor) + tx*wg.MorphFactor
	my := inputY*(1-wg.MorphFactor) + ty*wg.MorphFactor
	mz := 0*(1-wg.MorphFactor) + tz*wg.MorphFactor
	return wg.Project3D(mx, my, mz, centerX, centerY)
}

func (wg *WarpGrid) getTransformed(gridC, gridR float64, centerX, centerY float64) (float64, float64, float64) {
	// Map grid coords to [-1.5, 1.5] space
	inputX := (gridC/float64(wg.cols) - 0.5) * 3.0
	inputY := (gridR/float64(wg.rows) - 0.5) * 3.0

	// Get 5D hidden state
	hidden := wg.NN.GetLayerOutputs(inputX, inputY, wg.VisibleLayers)
	
	// Project 5D -> 3D
	var tx, ty, tz float64
	for i := 0; i < 3; i++ {
		val := 0.0
		for j := 0; j < 5 && j < len(hidden); j++ {
			val += wg.Projection[i][j] * hidden[j]
		}
		if i == 0 { tx = val } else if i == 1 { ty = val } else { tz = val }
	}

	// Morph: interpolate between flat 2D (inputX, inputY, 0) and warped 3D (tx, ty, tz)
	mx := inputX*(1-wg.MorphFactor) + tx*wg.MorphFactor
	my := inputY*(1-wg.MorphFactor) + ty*wg.MorphFactor
	mz := 0*(1-wg.MorphFactor) + tz*wg.MorphFactor

	return wg.Project3D(mx, my, mz, centerX, centerY)
}

func (wg *WarpGrid) GetTransformedScreen(sx, sy float64) (float64, float64, float64) {
	inputX := (sx/wg.width - 0.5) * 3.0
	inputY := (sy/wg.height - 0.5) * 3.0
	return wg.GetTransformedScreenFromWorld(inputX, inputY, 0)
}

func (wg *WarpGrid) GetTransformedScreenFromWorld(wx, wy, wz float64) (float64, float64, float64) {
	centerX, centerY := wg.width/2, wg.height/2

	// NN only takes the full raw data, but for visualization of Level Zero,
	// we use the projected 3D coords as inputs to see how the manifold evolves from them.
	// NOTE: This assumes the input layer is 3D for visualization purposes.
	// Real high-D stars will use sf.Update() with RawData.
	hidden := wg.NN.GetLayerOutputs(wx, wy, wg.VisibleLayers)

	var tx, ty, tz float64
	for i := 0; i < 3; i++ {
		val := 0.0
		for j := 0; j < 5 && j < len(hidden); j++ {
			val += wg.Projection[i][j] * hidden[j]
		}
		if i == 0 {
			tx = val
		} else if i == 1 {
			ty = val
		} else {
			tz = val
		}
	}

	mx := wx*(1-wg.MorphFactor) + tx*wg.MorphFactor
	my := wy*(1-wg.MorphFactor) + ty*wg.MorphFactor
	mz := wz*(1-wg.MorphFactor) + tz*wg.MorphFactor

	return wg.Project3D(mx, my, mz, centerX, centerY)
}

// Project3D returns (screenX, screenY, rawRelativeZ for shading)
func (wg *WarpGrid) Project3D(x, y, z float64, cx, cy float64) (float64, float64, float64) {
	scale := 150.0 * wg.Zoom

	// 1. Move point relative to screen center AND apply panning
	vx := x*scale + wg.CamX
	vy := y*scale + wg.CamY
	vz := z * scale

	// 2. Rotation (Yaw) around the visual center
	cosY, sinY := math.Cos(wg.Yaw), math.Sin(wg.Yaw)
	rx := vx*cosY - vy*sinY
	ry := vx*sinY + vy*cosY

	// 3. Tilt (Pitch)
	cosP, sinP := math.Cos(wg.Pitch), math.Sin(wg.Pitch)
	finalX := rx
	finalY := (ry * cosP) - (vz * sinP)

	// rawRelativeZ for shading (depth from camera)
	// Closer is higher positive, further is lower/negative
	rawZ := (ry * sinP) + (vz * cosP)

	return cx + finalX, cy + finalY, rawZ / 150.0
}
func (wg *WarpGrid) GetInputAt(sx, sy float64) []float64 {
	// Let's use the level's class logic to determine if it's an X or O
	// and generate a clean version for the animation.
	// We use the current level's class logic if available
	class := 0
	if wg.NN != nil {
		// Use a dummy level 6 if needed, or better, the current level logic
		// For Level 6, GetClassAt identifies X or O
		class = Level{ID: 6}.GetClassAt(sx, sy) 
	}
	
	pattern := make([]float64, 64)
	if class == 0 {
		for j := 0; j < 8; j++ {
			pattern[j*8+j] = 1.0
			pattern[j*8+(7-j)] = 1.0
		}
	} else {
		for j := 2; j < 6; j++ {
			pattern[0*8+j] = 1.0
			pattern[7*8+j] = 1.0
			pattern[j*8+0] = 1.0
			pattern[j*8+7] = 1.0
		}
	}
	return pattern
}
