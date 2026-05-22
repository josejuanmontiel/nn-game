package main

import (
	_ "embed"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"time"
	"math/rand/v2"

	"11-juego-final/game"
	"11-juego-final/nn"
	"11-juego-final/nn/avx"
	"11-juego-final/nn/level0"
	"11-juego-final/nn/oneapi"
	"11-juego-final/nn/parallel"
	"11-juego-final/nn/standard"
	"11-juego-final/nn/wasm"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

//go:embed levels.json
var defaultLevelsJSON []byte

const (
	screenWidth  = 1200
	screenHeight = 600

	focusNone  = 0
	focusDepth = 1
	focusWidth = 2
)

type Game struct {
	nn           *nn.NeuralNetwork
	ghostNN      *nn.NeuralNetwork
	grid         *game.WarpGrid
	player       *game.Player
	currentLevel game.Level
	starfield    *game.Starfield
	w, h         int // Current screen dimensions
	mode         string
	accuracy     float64
	history      []float64
	isWin        bool
	isTraining   bool
	isSimulating bool
	showBase     bool

	// Log and history
	showLog       bool
	logEntries    []string
	logOffset     int
	snapshotCount int
	iterCount     int

	// Audit Mode
	isAuditMode bool
	auditIdx    int
	isHoverNet  bool
	isHoverGrid bool

	// Simulation settings
	simSpeed    float64
	tickCounter int

	// Mouse state
	prevMouseX, prevMouseY int

	// Selection state
	selLayer int
	selNode  int
	selPrev  int

	// UI State
	showIntro     bool
	showTension   bool
	targetMorph   float64
	visibleLayers int

	// View State
	netX, netY    float64
	netZoom       float64
	isDraggingNet bool
	lastNetMouseX int
	lastNetMouseY int
	dragStartX    float64
	dragStartY    float64

	// Metadata
	actName    string
	actFormula string

	// Interactive Inputs
	focusBox   int
	inputDepth string
	inputWidth string

	// NEW: Probe Launching
	shots []*Shot

	ghostTimer  int
	demoStepIdx int

	// CNN Animation (Manim-style)
	isAnimatingConv bool
	convFrame       int
	convLayerIdx    int
	convChannelIdx  int
	convInput       []float64
	userIsDrawing    bool
	padCollapsed     bool
	pizarraCollapsed bool

	// CNN Interactive Probing
	isProbing    bool
	probeInput   []float64
	probeOutputs []any // Store outputs of each V2 layer
	scannedPatch []float64
	scannedRow   int
	scannedCol   int
}

type Shot struct {
	X, Y    float64
	TX, TY  float64
	Speed   float64
	Class   int
	Arrived bool
}

func (g *Game) Update() error {
	if g.showIntro {
		if inpututil.IsKeyJustPressed(ebiten.KeyEnter) || inpututil.IsKeyJustPressed(ebiten.KeyKPEnter) || ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
			g.showIntro = false
		}
		return nil
	}

	if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		g.ResetGame()
	}
	g.updateCNNAnimation()
	g.updateCNNProbing()

	if g.showLog {
		_, wy := ebiten.Wheel()
		if wy != 0 {
			g.logOffset -= int(wy)
			g.clampLogOffset()
		}
		if inpututil.IsKeyJustPressed(ebiten.KeyUp) {
			g.logOffset--
			g.clampLogOffset()
		}
		if inpututil.IsKeyJustPressed(ebiten.KeyDown) {
			g.logOffset++
			g.clampLogOffset()
		}
		return nil // Block EVERYTHING else while log is open
	}

	mx, my := ebiten.CursorPosition()
	fmx, fmy := float64(mx), float64(my)

	splitX := float64(g.w) * 0.65
	pizarraH := 240.0
	splitY := float64(g.h) - pizarraH
	fH := float64(g.h)

	// Panel-specific Navigation
	if fmx > splitX && fmy < splitY { // Network Panel (Right) - excluding Pizarra
		if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
			g.isDraggingNet = true
			g.dragStartX = fmx
			g.dragStartY = fmy
		}
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) && g.isDraggingNet {
			dx := fmx - g.dragStartX
			dy := fmy - g.dragStartY
			g.netX += dx
			g.netY += dy
			g.dragStartX = fmx
			g.dragStartY = fmy
		}
		_, wy := ebiten.Wheel()
		if wy != 0 {
			g.netZoom *= math.Pow(1.1, wy)
			g.netZoom = math.Max(0.1, math.Min(5.0, g.netZoom))
		}
	} else if fmx <= splitX && g.currentLevel.EngineType != "cnn" { // Manifold Panel (Left)
		// Camera Zoom (Mouse Wheel)
		_, wy := ebiten.Wheel()
		if wy != 0 {
			k := 1.1
			if wy < 0 {
				k = 1.0 / 1.1
			}
			g.grid.Zoom *= k
			// Center zoom on mouse relative to center of left panel
			centerX := splitX / 2.0
			centerY := fH / 2.0
			g.grid.CamX -= (fmx - centerX - g.grid.CamX) * (k - 1)
			g.grid.CamY -= (fmy - centerY - g.grid.CamY) * (k - 1)
		}
		// Camera Pan (Right Click Drag)
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight) {
			dx := mx - g.prevMouseX
			dy := my - g.prevMouseY
			g.grid.CamX += float64(dx)
			g.grid.CamY += float64(dy)
		}
		// Camera Rotate (Left Click Drag)
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
			dx := mx - g.prevMouseX
			dy := my - g.prevMouseY
			g.grid.Yaw += float64(dx) * 0.01
			g.grid.Pitch += float64(dy) * 0.01
			halfPi := 3.14159 / 2.2
			if g.grid.Pitch > halfPi {
				g.grid.Pitch = halfPi
			}
			if g.grid.Pitch < -halfPi {
				g.grid.Pitch = -halfPi
			}
		}
	}

	if !ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		g.isDraggingNet = false
	}

	if inpututil.IsKeyJustPressed(ebiten.KeyC) {
		g.centerCamera()
	}

	if inpututil.IsKeyJustPressed(ebiten.KeyF) {
		ebiten.SetFullscreen(!ebiten.IsFullscreen())
	}

	g.handleInput()

	// NEW: Probe Shooting
	if fmx < splitX && ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) && ebiten.IsKeyPressed(ebiten.KeyShift) {
		// Launch from player (or center if no player?)
		class := g.currentLevel.GetClassAt(fmx, fmy)
		g.shots = append(g.shots, &Shot{
			X:     g.player.X,
			Y:     g.player.Y,
			TX:    fmx,
			TY:    fmy,
			Speed: 5.0,
			Class: class,
		})

		// Trigger CNN Animation
		if g.currentLevel.EngineType == "cnn" && !g.isAnimatingConv {
			g.isAnimatingConv = true
			g.convFrame = 0
			g.convChannelIdx = 0
			g.convInput = g.grid.GetInputAt(fmx, fmy)
		}
	}

	// Update Shots
	for i := 0; i < len(g.shots); i++ {
		s := g.shots[i]
		dx := s.TX - s.X
		dy := s.TY - s.Y
		dist := math.Sqrt(dx*dx + dy*dy)
		if dist < s.Speed {
			s.Arrived = true
			g.starfield.AddStar(s.TX, s.TY, s.Class)
			g.addLog(fmt.Sprintf("Probe Collapsed: Class %d at (%.0f, %.0f)", s.Class, s.TX, s.TY))
			// Remove
			g.shots = append(g.shots[:i], g.shots[i+1:]...)
			i--
		} else {
			s.X += (dx / dist) * s.Speed
			s.Y += (dy / dist) * s.Speed
		}
	}

	mx, my = ebiten.CursorPosition()
	g.prevMouseX, g.prevMouseY = mx, my

	// NEW: Global Pause (Space)
	if ebiten.IsKeyPressed(ebiten.KeySpace) {
		g.isTraining = false
		g.isSimulating = false
		g.mode = "PAUSED"
	}
	g.showBase = ebiten.IsKeyPressed(ebiten.KeyB)

	// Activation Function Power-ups
	if ebiten.IsKeyPressed(ebiten.KeyDigit1) {
		g.nn.SetActivation("relu")
		g.actName = "ReLU"
		g.actFormula = "max(0, x)"
	}
	if ebiten.IsKeyPressed(ebiten.KeyDigit2) {
		g.nn.SetActivation("tanh")
		g.actName = "Tanh"
		g.actFormula = "tanh(x)"
	}
	if ebiten.IsKeyPressed(ebiten.KeyDigit3) {
		g.nn.SetActivation("identity")
		g.actName = "Identity"
		g.actFormula = "x"
	}

	// Dynamic Layers (+ / - keys or P / M)
	if inpututil.IsKeyJustPressed(ebiten.KeyEqual) || inpututil.IsKeyJustPressed(ebiten.KeyKPAdd) || inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.nn.AddLayer(5)
		g.mode = fmt.Sprintf("Deep Depth: %d Layers", len(g.nn.Layers))
		g.resetAuditSelection()
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyMinus) || inpututil.IsKeyJustPressed(ebiten.KeyKPSubtract) || inpututil.IsKeyJustPressed(ebiten.KeyM) {
		g.nn.RemoveLayer()
		g.mode = fmt.Sprintf("Deep Depth: %d Layers", len(g.nn.Layers))
		g.resetAuditSelection()
	}

	// Cheat/Jump keys: N or J to next level
	if inpututil.IsKeyJustPressed(ebiten.KeyN) || inpututil.IsKeyJustPressed(ebiten.KeyJ) {
		// Find current index
		currIdx := -1
		for i, l := range game.Levels {
			if l.ID == g.currentLevel.ID {
				currIdx = i
				break
			}
		}
		if currIdx != -1 && currIdx < len(game.Levels)-1 {
			g.currentLevel = game.Levels[currIdx+1]
			g.addLog(fmt.Sprintf("--- JUMPED TO Level %d: %s ---", g.currentLevel.ID, g.currentLevel.Name))
			g.ResetGame()
		}
	} else {
		g.player.Update(g.grid)
		g.starfield.Update(g.grid, g.currentLevel.NumClasses)
	}

	if inpututil.IsKeyJustPressed(ebiten.KeyK) {
		g.saveSnapshot()
	}

	// NEW: Automatic Audit Mode ONLY on Pizarra (calculation zone) hover
	g.isAuditMode = fmx >= splitX && fmy >= splitY
	g.isHoverGrid = fmx < splitX && fmy < fH
	g.isHoverNet = fmx >= splitX && fmy < splitY

	if g.isAuditMode {
		// Navigation logic remains focused here
	}

	if g.isAuditMode {
		_, wy := ebiten.Wheel()
		if wy != 0 {
			// Navigate through all hidden+output nodes
			totalNodes := 0
			for _, l := range g.nn.Layers {
				totalNodes += len(l.Outputs)
			}

			if wy > 0 {
				g.auditIdx++
			} else {
				g.auditIdx--
			}

			// Wrap / Clamp
			if g.auditIdx < 0 {
				g.auditIdx = totalNodes - 1
			}
			if g.auditIdx >= totalNodes {
				g.auditIdx = 0
			}

			// Map global auditIdx back to selLayer and selNode
			curr := 0
			for lIdx, l := range g.nn.Layers {
				if g.auditIdx < curr+len(l.Outputs) {
					g.selLayer = lIdx + 1
					g.selNode = g.auditIdx - curr
					g.selPrev = -1 // Focus on node/bias by default
					break
				}
				curr += len(l.Outputs)
			}
		}
	} else if g.isHoverNet {
		g.handleNetworkInteraction()
		g.applyAdjustment()
	} else if g.isHoverGrid {
		// Hovering over the starfield/grid zone
	}

	if g.isSimulating {
		g.mode = "Intelligent Simulation: Auto-Ordering..."
	} else {
		g.mode = "Standard Orbit"
	}

	acc := g.starfield.GetAccuracy()
	g.accuracy = g.accuracy*0.9 + acc*0.1

	// Add to history every 60 frames
	if ebiten.ActualTPS() > 0 && int(ebiten.ActualFPS())%60 == 0 {
		g.history = append(g.history, g.accuracy)
		if len(g.history) > 100 {
			g.history = g.history[1:]
		}
	}

	// Auto-Training / Simulation Logic
	if (g.isTraining || g.isSimulating) && !g.isWin && !g.isAuditMode {
		// Both modes respect simSpeed for pedagogical observation
		g.autoTrain(g.simSpeed)

		// Throttled Snapshot: Every 15 frames (~4 times per second)
		g.tickCounter++
		if g.tickCounter%15 == 0 {
			g.takeSnapshot()
		}
	} else {
		g.tickCounter = 0
	}

	// Don't auto-win if already won
	if g.accuracy >= g.currentLevel.TargetAcc && !g.isWin {
		g.isWin = true
		g.mode = "CONVERGENCE ACHIEVED: LEVEL COMPLETE"
		g.addLog(fmt.Sprintf("CONVERGENCE ACHIEVED! Final Accuracy: %.2f%%", g.accuracy*100))
		g.takeSnapshot()

		// Auto-Save Matrix
		filename := fmt.Sprintf("net_L%d_acc%.0f.json", g.currentLevel.ID, g.accuracy*100)
		if err := g.nn.Save(filename); err == nil {
			g.addLog(fmt.Sprintf("Matrix Auto-Saved: %s", filename))
		}

		g.addLog("You can still inspect the network or press 'N' for next level.")
	}

	// Simulation Toggle (S Key)
	if inpututil.IsKeyJustPressed(ebiten.KeyS) {
		g.isSimulating = !g.isSimulating
		if g.isSimulating {
			g.isTraining = false
		}
	}

	// Training Toggle (T Key)
	if inpututil.IsKeyJustPressed(ebiten.KeyT) {
		g.isTraining = !g.isTraining
		if g.isTraining {
			g.isSimulating = false
		}
	}

	// Visualization Mode (X key)
	if inpututil.IsKeyJustPressed(ebiten.KeyX) {
		g.grid.Mode = (g.grid.Mode + 1) % 6 // 6 modes defined in viz_types.go
		if g.grid.Mode == 0 {               // Reset to Fixed
			g.grid.Projection = game.IdentityProjection()
		}
		g.addLog(fmt.Sprintf("Visual Mode: %s", g.grid.Mode.String()))
	}

	// Periodic PCA Update (every 60 frames if in PCA mode)
	if g.grid.Mode == game.VizModePCA || g.grid.Mode == game.VizModeGlyph {
		if g.tickCounter%60 == 0 {
			g.grid.UpdateProjection()
		}
	}

	// Update Highlight for Visualizer (Player pos)
	px := (g.player.X/float64(g.w/2) - 0.5) * 2.0
	py := (g.player.Y/float64(g.h) - 0.5) * 2.0
	g.grid.Highlight = g.nn.GetLayerOutputs(px, py, g.grid.VisibleLayers)

	// Variance Alert: If higher dims become very active
	if len(g.grid.Variances) > 3 {
		sumVisible := 0.0
		numVisible := 0
		for i := 0; i < 3 && i < len(g.grid.Variances); i++ {
			sumVisible += g.grid.Variances[i]
			numVisible++
		}
		if numVisible > 0 {
			avgActive := sumVisible / float64(numVisible)
			alert := false
			for i := 3; i < len(g.grid.Variances); i++ {
				if g.grid.Variances[i] > avgActive*2.0 {
					alert = true
					break
				}
			}
			if alert && g.tickCounter%120 < 60 {
				g.mode = "WARNING: HI-DIM ACTIVITY DETECTED (TRY PCA/GLYPH MODE)"
			}
		}
	}

	// Manual Step (Enter)
	if inpututil.IsKeyJustPressed(ebiten.KeyEnter) || inpututil.IsKeyJustPressed(ebiten.KeyKPEnter) {
		g.autoTrain(0.05) // Stronger step for manual
		g.mode = "Manual Evolution Step"
		g.addLog("Manual Training Step Executed [0.05]")
		g.takeSnapshot()
	}

	// Synchronize visible layers to grid
	g.grid.VisibleLayers = g.visibleLayers

	// Explanatory Controls
	if inpututil.IsKeyJustPressed(ebiten.KeyV) {
		if g.targetMorph == 1.0 {
			g.targetMorph = 0.0
			g.addLog("Releasing space: Morphing back to 2D")
		} else {
			g.targetMorph = 1.0
			g.addLog("Stretching space: Morphing to Neural Manifold")
		}
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyB) {
		g.showTension = !g.showTension
		g.addLog(fmt.Sprintf("Tension Lines: %v", g.showTension))
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyPageUp) {
		g.visibleLayers++
		if g.visibleLayers > len(g.nn.Layers) {
			g.visibleLayers = len(g.nn.Layers)
		}
		g.addLog(fmt.Sprintf("Visible Depth: %d Layers", g.visibleLayers))
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyPageDown) {
		g.visibleLayers--
		if g.visibleLayers < 0 {
			g.visibleLayers = 0
		}
		g.addLog(fmt.Sprintf("Visible Depth: %d Layers", g.visibleLayers))
	}

	// Morph Animation
	if g.grid.MorphFactor < g.targetMorph {
		g.grid.MorphFactor += 0.02
		if g.grid.MorphFactor > g.targetMorph {
			g.grid.MorphFactor = g.targetMorph
		}
	} else if g.grid.MorphFactor > g.targetMorph {
		g.grid.MorphFactor -= 0.02
		if g.grid.MorphFactor < g.targetMorph {
			g.grid.MorphFactor = g.targetMorph
		}
	}

	// Simulation Speed Control
	if inpututil.IsKeyJustPressed(ebiten.KeyLeftBracket) {
		g.simSpeed *= 0.5
		if g.simSpeed < 0.0001 {
			g.simSpeed = 0.0001
		}
		g.addLog(fmt.Sprintf("Sim Speed: %.5f", g.simSpeed))
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyRightBracket) {
		g.simSpeed *= 2.0
		if g.simSpeed > 0.5 {
			g.simSpeed = 0.5
		}
		g.addLog(fmt.Sprintf("Sim Speed: %.5f", g.simSpeed))
	}

	// Update Ghost Timer
	if g.ghostNN != nil {
		g.ghostTimer--
		if g.ghostTimer <= 0 {
			g.ghostNN = nil
			g.grid.GhostNN = nil
		}
	}

	return nil
}

func (g *Game) autoTrain(lr float64) {
	if g.accuracy > 0.999 || g.starfield == nil || len(g.starfield.Stars) == 0 || g.currentLevel.NumClasses <= 0 {
		return
	}

	targets := game.Get3DTargets(0.8, g.currentLevel.NumClasses)
	if len(targets) == 0 {
		return
	}

	// Pick more random stars to accelerate convergence for complex levels
	for i := 0; i < 24; i++ {
		idx := rand.IntN(len(g.starfield.Stars))
		s := g.starfield.Stars[idx]

		if s.Class >= len(targets) {
			continue
		}
		target := targets[s.Class]

		g.nn.TrainStep(s.RawData, target, lr)
	}
	g.iterCount++
}

func (g *Game) ResetGame() {
	// Create network and apply level-specific settings
	if g.currentLevel.EngineType == "cnn" {
		// Special CNN Architecture for Shapes level
		// Input: 8x8 = 64
		nnv2 := &nn.NeuralNetwork{
			InSize:         64,
			OutSize:        3,
			ActivationType: "tanh",
			Activation:     nn.Tanh,
			Derivative:     func(x float64) float64 { return 1.0 - x*x },
			WeightScale:    1.0,
			BiasScale:      0.0,
			EngineType:     "cnn",
			Engine:         &nn.FallbackEngine{},
		}

		// Layer 1: Conv 1->8 channels, 3x3 kernel, stride 1, padding 1
		conv := nn.NewConvLayer(1, 8, 3, 1, 1)
		// Layer 2: Max Pool 2x2, stride 2
		pool := nn.NewPoolLayer(2, 2)
		// Layer 3: Flatten
		flatten := nn.NewFlattenLayer()
		// Layer 4: Dense 128 -> 32 (HiddenSizes[0])
		h1Size := 32
		if len(g.currentLevel.HiddenSizes) > 0 {
			h1Size = g.currentLevel.HiddenSizes[0]
		}
		dense1 := &nn.Layer{
			Weights:    make([][]float64, h1Size),
			Biases:     make([]float64, h1Size),
			Inputs:     make([]float64, 128),
			Outputs:    make([]float64, h1Size),
			Errors:     make([]float64, h1Size),
			PrevDeltaW: make([][]float64, h1Size),
			PrevDeltaB: make([]float64, h1Size),
			LastDelta:  make([]float64, h1Size),
			LastGradW:  make([][]float64, h1Size),
			LastGradB:  make([]float64, h1Size),
			Activation: nn.Tanh,
			Derivative: func(x float64) float64 { return 1.0 - x*x },
			Engine:     nnv2.Engine,
		}
		for i := range dense1.Weights {
			dense1.Weights[i] = make([]float64, 128)
			dense1.PrevDeltaW[i] = make([]float64, 128)
			dense1.LastGradW[i] = make([]float64, 128)
		}
		// Layer 5: Dense 32 -> 3 (Output) - ALWAYS use 3 for Manifold compatibility
		denseOut := &nn.Layer{
			Weights:    make([][]float64, 3),
			Biases:     make([]float64, 3),
			Inputs:     make([]float64, h1Size),
			Outputs:    make([]float64, 3),
			Errors:     make([]float64, 3),
			PrevDeltaW: make([][]float64, 3),
			PrevDeltaB: make([]float64, 3),
			LastDelta:  make([]float64, 3),
			LastGradW:  make([][]float64, 3),
			LastGradB:  make([]float64, 3),
			Activation: nn.Tanh,
			Derivative: func(x float64) float64 { return 1.0 - x*x },
			Engine:     nnv2.Engine,
		}
		for i := range denseOut.Weights {
			denseOut.Weights[i] = make([]float64, h1Size)
			denseOut.PrevDeltaW[i] = make([]float64, h1Size)
			denseOut.LastGradW[i] = make([]float64, h1Size)
		}

		nnv2.LayersV2 = []nn.ILayer{conv, pool, flatten, dense1, denseOut}
		g.nn = nnv2
		
		// Fill original Layers slice with proxy layers for HUD/Viz compatibility
		// We omit Pool and Flatten from the visual diagram to prevent massive node counts (128 nodes)
		// from drawing off-screen and causing "double overlapping points".
		// This keeps the diagram focused on Conv -> Dense -> Out.
		g.nn.Layers = []*nn.Layer{
			{Outputs: make([]float64, 8), Biases: make([]float64, 8)}, // Conv (8 filters)
			dense1,                                                    // Dense1 (32 nodes)
			denseOut,                                                  // DenseOut (3 nodes)
		}
	} else {
		g.nn = nn.NewNeuralNetwork(g.currentLevel.InputDims, g.currentLevel.HiddenSizes, g.currentLevel.NumClasses)
		g.nn.WeightScale = g.currentLevel.WeightInitScale
		g.nn.BiasScale = g.currentLevel.BiasInitScale

		// Initialize the selected math engine
		engine := g.getEngine(g.currentLevel.EngineType)
		g.nn.SetEngine(engine, g.currentLevel.EngineType)
	}

	g.nn.ResetWeights()

	// Load HUD VizMode
	switch g.currentLevel.VizMode {
	case "PCA":
		g.grid.Mode = game.VizModePCA
	case "Parallel":
		g.grid.Mode = game.VizModeParallel
	case "SPLOM":
		g.grid.Mode = game.VizModeSPLOM
	case "Schlegel":
		g.grid.Mode = game.VizModeSchlegel
	case "Glyph":
		g.grid.Mode = game.VizModeGlyph
	default:
		g.grid.Mode = game.VizModeFixed
	}
	g.grid.Projection = game.IdentityProjection()

	g.isAuditMode = g.currentLevel.StartPaused
	g.ghostNN = nil
	g.ghostTimer = 0

	g.addLog(fmt.Sprintf("GAME: Reset Level %d (%s) - Engine: %s [Paused: %v]", g.currentLevel.ID, g.currentLevel.Name, g.currentLevel.EngineType, g.isAuditMode))

	g.player.X = (float64(g.w) * 0.65) / 2.0 // Center of left panel
	g.player.Y = float64(g.h) / 2.0
	g.accuracy = 0
	g.history = []float64{0}
	g.isWin = false
	g.isTraining = false
	g.padCollapsed = true
	g.pizarraCollapsed = false
	g.simSpeed = 0.0001 // Ultra Slow for observation
	g.isSimulating = false
	g.iterCount = 0
	g.mode = "Space Reset"
	g.grid.Zoom = 1.0
	g.grid.CamX = 0
	g.grid.CamY = 0
	g.grid.NN = g.nn // Update grid's NN reference
	splitX := float64(g.w) * 0.65
	g.starfield = game.NewStarfield(g.currentLevel, 200, splitX, float64(g.h))
	g.showIntro = true
	g.selLayer = -1
	g.selNode = -1
	g.selPrev = -1
	g.targetMorph = 1.0
	g.grid.MorphFactor = 1.0
	g.visibleLayers = len(g.nn.Layers)
	g.showTension = false

	g.netX = float64(g.w) * 0.825
	g.netY = float64(g.h) / 2.0
	g.netZoom = 1.0
	g.isDraggingNet = false

	// Clear ghost state
	g.ghostNN = nil
	g.ghostTimer = 0

	// Update grid references
	g.grid.NN = g.nn
	g.grid.GhostNN = g.ghostNN

	// Demo Initialization
	if g.currentLevel.ID == 0 && len(g.currentLevel.DemoSteps) > 0 {
		g.demoStepIdx = 0
		g.ApplyDemoStep(0)
	}
}

func (g *Game) ApplyDemoStep(idx int) {
	if idx < 0 || idx >= len(g.currentLevel.DemoSteps) {
		return
	}
	step := g.currentLevel.DemoSteps[idx]
	layer0 := g.nn.Layers[0]
	if len(step.Weights) > 0 {
		for i := range layer0.Weights {
			if i < len(step.Weights) {
				copy(layer0.Weights[i], step.Weights[i])
			}
		}
	}
	if len(step.Biases) > 0 {
		copy(layer0.Biases, step.Biases)
	}
	g.addLog(fmt.Sprintf("DEMO: %s", step.Name))
}

func (g *Game) AdvanceDemoStep() {
	if len(g.currentLevel.DemoSteps) == 0 {
		return
	}
	g.ghostNN = g.nn.Clone()
	g.grid.GhostNN = g.ghostNN
	g.ghostTimer = 180 // 3 seconds for demo transitions

	g.demoStepIdx = (g.demoStepIdx + 1) % len(g.currentLevel.DemoSteps)
	g.ApplyDemoStep(g.demoStepIdx)
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Deep Space Background
	screen.Fill(color.RGBA{5, 5, 20, 255})

	// HUD info
	// Redundant top-left HUD removed as it overlaps with drawStats

	// Draw game elements regardless of win to allow post-win inspection
	if g.showBase {
		g.drawBaseGrid(screen)
	} else {
		g.grid.Draw(screen)
	}

	if g.showLog {
		g.drawLog(screen)
		return
	}

	if g.showIntro {
		g.drawIntro(screen)
		return
	}

	// NEW: Manifold is now on the left half (0-600)
	// Grid and starfield already use internal logic that needs to be aware of the center.
	// We'll pass the center (300, 300) to them if needed, or adjust their logic.

	// Geometric World Elements (Only in Mesh/PCA/Glyph modes)
	isCNN := g.currentLevel.EngineType == "cnn"

	if g.grid.Mode == game.VizModeFixed || g.grid.Mode == game.VizModePCA || g.grid.Mode == game.VizModeGlyph {
		if !isCNN {
			g.drawLegend(screen)
			g.starfield.Draw(screen, g.grid)
			if g.showTension {
				g.drawTension(screen)
			}
		}
	}

	// NEW: Architecture Map, Decision Boundaries, and Class Targets
	if !isCNN && (g.grid.Mode == game.VizModeFixed || g.grid.Mode == game.VizModePCA || g.grid.Mode == game.VizModeGlyph) {
		// 1. Static Base Grid (XY Plane)
		g.drawBaseGrid(screen)

		// 2. Decision Boundaries (Only in 2D view)
		if g.grid.MorphFactor < 0.5 {
			g.drawBoundaries(screen)
		}

		g.drawTargets(screen)
		g.player.Draw(screen, g.grid)
		g.drawProbe(screen)
	}

	// Draw win overlay if needed
	if g.isWin {
		// Semi-transparent overlay for text readability
		vector.DrawFilledRect(screen, 50, 240, 500, 100, color.RGBA{0, 0, 0, 180}, true)
		ebitenutil.DebugPrintAt(screen, "CONVERGENCE ACHIEVED!", 220, 250)
		ebitenutil.DebugPrintAt(screen, g.currentLevel.Name+" Cleared!", 180, 270)
		ebitenutil.DebugPrintAt(screen, "Press 'N' for next level or 'R' to reset.", 160, 300)
	}

	// NEW: Draw knob interaction ranges and ghost player in base space
	if !isCNN {
		g.drawControlZones(screen)
	}
	g.drawHistory(screen)
	g.drawNetwork(screen) // NOW PANNABLE ON THE RIGHT
	if !isCNN && (g.grid.Mode == game.VizModeFixed || g.grid.Mode == game.VizModePCA || g.grid.Mode == game.VizModeGlyph) {
		g.drawAxis(screen)
	}

	// HUDs and Pizarra (On top of everything)
	g.drawStats(screen)
	g.drawActivationHUD(screen)
	if g.currentLevel.EngineType == "cnn" {
		g.drawCNNDrawingPad(screen)
		g.drawTracerFooter(screen) // Recuperamos la pizarra!
	} else {
		g.grid.Draw(screen)
		g.player.Draw(screen, g.grid)
		g.drawTracerFooter(screen)
		g.drawKernels(screen)
	}
}

func (g *Game) centerCamera() {
	g.grid.Zoom = 1.0
	g.grid.CamX = 0
	g.grid.CamY = 0
	g.grid.Pitch = 0
	g.grid.Yaw = 0
}

func (g *Game) drawAxis(screen *ebiten.Image) {
	splitX := float64(g.w) * 0.65
	centerX, centerY := splitX/2.0, float64(g.h)/2.0
	project := func(mx, my float64) (float64, float64, float64) {
		tx, ty, tz := g.nn.Transform2D(mx, my)
		return g.grid.Project3D(tx, ty, tz, centerX, centerY)
	}
	ox, oy, _ := project(0, 0)
	xx, xy, _ := project(0.5, 0)
	yx, yy, _ := project(0, 0.5)

	// For Z, rotate slightly to see the Z vector
	_, _, tz_orig := g.nn.Transform2D(0, 0)
	zx, zy, _ := g.grid.Project3D(0, 0, tz_orig+0.5, centerX, centerY)

	vector.StrokeLine(screen, float32(ox), float32(oy), float32(xx), float32(xy), 2, color.RGBA{255, 50, 50, 200}, true)
	vector.StrokeLine(screen, float32(ox), float32(oy), float32(yx), float32(yy), 2, color.RGBA{50, 255, 50, 200}, true)
	vector.StrokeLine(screen, float32(ox), float32(oy), float32(zx), float32(zy), 2, color.RGBA{50, 50, 255, 200}, true)
}

func (g *Game) drawTargets(screen *ebiten.Image) {
	splitX := float64(g.w) * 0.65
	centerX, centerY := splitX/2.0, float64(g.h)/2.0
	// Draw the 5 3D class centers as pulsating spheres
	targets := game.Get3DTargets(0.8, g.currentLevel.NumClasses)
	for i, t := range targets {
		// Project 3D coordinate to screen center of left panel
		tx, ty, tz := g.grid.Project3D(t[0], t[1], t[2], centerX, centerY)

		depthFactor := 1.0 + tz*0.5
		if depthFactor < 0.1 {
			continue
		}

		c := game.ClassColors[i]
		size := float32(8.0 * depthFactor)

		// Draw target sphere
		vector.DrawFilledCircle(screen, float32(tx), float32(ty), size, c, true)
		vector.StrokeCircle(screen, float32(tx), float32(ty), size+2, 1, color.White, true)

		// Label
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("C%d", i), int(tx)-8, int(ty)-8)
	}
}

func (g *Game) drawProbe(screen *ebiten.Image) {
	splitX := float64(g.w) * 0.65
	fH := float64(g.h)
	centerX, centerY := splitX/2.0, fH/2.0

	// The player is the "New Element". Let's show its classification.
	bx := (g.player.X/splitX - 0.5) * 2.0
	by := (g.player.Y/fH - 0.5) * 2.0

	// Forward pass for player position
	tx, ty, tz := g.nn.Transform2D(bx, by)

	// Project to screen space
	stx, sty, stz := g.grid.Project3D(tx, ty, tz, centerX, centerY)

	// Find closest class
	targets := game.Get3DTargets(0.8, g.currentLevel.NumClasses)
	minDist := 1000.0
	closestClass := -1
	for i, t := range targets {
		d := math.Sqrt(math.Pow(tx-t[0], 2) + math.Pow(ty-t[1], 2) + math.Pow(tz-t[2], 2))
		if d < minDist {
			minDist = d
			closestClass = i
		}
	}

	if closestClass != -1 {
		c := game.ClassColors[closestClass]

		// 1. Draw projection dot in 3D space
		depthFactor := 1.0 + stz*0.5
		vector.DrawFilledCircle(screen, float32(stx), float32(sty), float32(4*depthFactor), c, true)
		vector.StrokeCircle(screen, float32(stx), float32(sty), float32(6*depthFactor), 1, color.White, true)

		// 2. Draw line to closest target
		target3D := targets[closestClass]
		ttx, tty, _ := g.grid.Project3D(target3D[0], target3D[1], target3D[2], centerX, centerY)
		vector.StrokeLine(screen, float32(stx), float32(sty), float32(ttx), float32(tty), 1, color.RGBA{c.R, c.G, c.B, 100}, true)

		// 3. Draw classification halo around screen player
		vector.StrokeCircle(screen, float32(g.player.X), float32(g.player.Y), 12, 2, c, true)

		// (Removed legacy CNN logic that drew an 8x8 pattern near the cursor)
	}
}

func (g *Game) drawArchitecture(screen *ebiten.Image) {
	// Keep it simple as we have the new drawNetwork
	x, y := 30, 400
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Layers: %d", len(g.nn.Layers)), x, y)
}

func (g *Game) drawBaseGrid(screen *ebiten.Image) {
	splitX := float32(g.w) * 0.65
	fH := float32(g.h)
	// Simplified static grid
	for i := 0; i <= 20; i++ {
		x := float32(i) * splitX / 20
		vector.StrokeLine(screen, x, 0, x, fH, 1, color.RGBA{50, 50, 50, 255}, true)
		y := float32(i) * fH / 20
		vector.StrokeLine(screen, 0, y, splitX, y, 1, color.RGBA{50, 50, 50, 255}, true)
	}
}

func (g *Game) drawHistory(screen *ebiten.Image) {
	// Show it even if it's mostly empty to guide the user
	graphW, graphH := 180.0, 60.0
	graphX, graphY := float64(g.w)-graphW-20.0, 30.0

	// Draw box
	vector.DrawFilledRect(screen, float32(graphX), float32(graphY), float32(graphW), float32(graphH), color.RGBA{0, 0, 0, 150}, true)
	vector.StrokeRect(screen, float32(graphX), float32(graphY), float32(graphW), float32(graphH), 1, color.RGBA{100, 100, 100, 255}, true)
	ebitenutil.DebugPrintAt(screen, "Classification Progression", int(graphX), int(graphY)-15)

	for i := 1; i < len(g.history); i++ {
		x1 := graphX + float64(i-1)*graphW/100.0
		y1 := graphY + graphH - g.history[i-1]*graphH
		x2 := graphX + float64(i)*graphW/100.0
		y2 := graphY + graphH - g.history[i]*graphH
		vector.StrokeLine(screen, float32(x1), float32(y1), float32(x2), float32(y2), 2, color.RGBA{0, 255, 100, 255}, true)
	}
}

func (g *Game) drawControlZones(screen *ebiten.Image) {
	vector.DrawFilledCircle(screen, float32(g.player.X), float32(g.player.Y), 4, color.RGBA{255, 255, 255, 100}, true)
}

func (g *Game) drawLegend(screen *ebiten.Image) {
	startX, startY := 20.0, float64(g.h)-150.0 // Move to bottom left of manifold panel
	ebitenutil.DebugPrintAt(screen, "CLASS COLOR GUIDE", int(startX), int(startY)-20)

	for i := 0; i < g.currentLevel.NumClasses; i++ {
		c := game.ClassColors[i]
		y := startY + float64(i)*25
		vector.DrawFilledRect(screen, float32(startX), float32(y), 15, 15, c, true)
		vector.StrokeRect(screen, float32(startX), float32(y), 15, 15, 1, color.White, true)
		label := fmt.Sprintf("Class %d Target", i)
		if g.currentLevel.EngineType == "cnn" {
			if i == 0 {
				label = "Class 0: Pattern 'X'"
			} else if i == 1 {
				label = "Class 1: Pattern 'O'"
			}
		}
		ebitenutil.DebugPrintAt(screen, label, int(startX)+25, int(y))
	}

	// Network guide
	ny := startY + float64(g.currentLevel.NumClasses)*25 + 15
	ebitenutil.DebugPrintAt(screen, "NETWORK GUIDE", int(startX), int(ny))
	vector.StrokeLine(screen, float32(startX), float32(ny+25), float32(startX+20), float32(ny+25), 2, color.RGBA{100, 100, 255, 255}, true)
	ebitenutil.DebugPrintAt(screen, "+ Weight", int(startX)+25, int(ny)+18)
	vector.StrokeLine(screen, float32(startX), float32(ny+45), float32(startX+20), float32(ny+45), 2, color.RGBA{255, 100, 100, 255}, true)
	ebitenutil.DebugPrintAt(screen, "- Weight", int(startX)+25, int(ny)+38)
}

func (g *Game) drawKernels(screen *ebiten.Image) {
	if g.currentLevel.EngineType != "cnn" {
		return
	}

	// Find the first ConvLayer
	var conv *nn.ConvLayer
	for _, l := range g.nn.LayersV2 {
		if c, ok := l.(*nn.ConvLayer); ok {
			conv = c
			break
		}
	}
	if conv == nil {
		return
	}

	// Draw kernels in a column on the left
	startX, startY := 10, 150
	cellSize := 4
	spacing := 10
	
	ebitenutil.DebugPrintAt(screen, "V1 FILTERS (8x8)", startX, startY-20)
	
	for i := 0; i < conv.OutChannels; i++ {
		// Kernels[outChannel][inChannel]
		k := conv.Kernels[i][0]
		kx, ky := startX, startY + i*(conv.KernelSize*cellSize + spacing)
		
		// Background
		vector.DrawFilledRect(screen, float32(kx)-1, float32(ky)-1, float32(conv.KernelSize*cellSize)+2, float32(conv.KernelSize*cellSize)+2, color.RGBA{0, 0, 0, 200}, true)
		
		for r := 0; r < conv.KernelSize; r++ {
			for c := 0; c < conv.KernelSize; c++ {
				val := k.At(r, c)
				// Normalize visualization (simple shift/scale for visibility)
				v := uint8(math.Max(0, math.Min(255, (val+0.2)*255*3)))
				col := color.RGBA{v, v, v, 255}
				vector.DrawFilledRect(screen, float32(kx+c*cellSize), float32(ky+r*cellSize), float32(cellSize)-1, float32(cellSize)-1, col, true)
			}
		}
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("K%d", i), kx + conv.KernelSize*cellSize + 5, ky)
	}
}

func (g *Game) addLog(msg string) {
	g.logEntries = append(g.logEntries, msg)
	if len(g.logEntries) > 1000 { // Even more capacity
		g.logEntries = g.logEntries[1:]
	}
	g.scrollToBottom()
}

func (g *Game) scrollToBottom() {
	maxScroll := len(g.logEntries) - 25
	if maxScroll < 0 {
		maxScroll = 0
	}
	g.logOffset = maxScroll
}

func (g *Game) clampLogOffset() {
	if g.logOffset < 0 {
		g.logOffset = 0
	}
	maxScroll := len(g.logEntries) - 25
	if maxScroll < 0 {
		maxScroll = 0
	}
	if g.logOffset > maxScroll {
		g.logOffset = maxScroll
	}
}

func (g *Game) takeSnapshot() {
	g.snapshotCount++
	summary := g.nn.Summary()
	g.addLog(fmt.Sprintf("#%d SNAPSHOT: %s", g.snapshotCount, summary))
	g.mode = fmt.Sprintf("Snapshot #%d Captured", g.snapshotCount)
}

func (g *Game) drawLog(screen *ebiten.Image) {
	// Overlay centered
	w, h := 740.0, 540.0
	x, y := (float64(g.w)-w)/2, (float64(g.h)-h)/2
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h), color.RGBA{0, 0, 0, 240}, true)
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), 1, color.RGBA{100, 255, 100, 255}, true)

	ebitenutil.DebugPrintAt(screen, "NEURAL EVOLUTION LOG / HISTORY", int(x)+270, int(y)+20)
	ebitenutil.DebugPrintAt(screen, "Press 'L' to exit / Wheel or Arrows to Scroll", int(x)+230, int(y)+40)

	visibleEntries := 25
	for i := 0; i < visibleEntries && i+g.logOffset < len(g.logEntries); i++ {
		idx := i + g.logOffset
		g.drawColoredEntry(screen, fmt.Sprintf("[%03d] %s", idx+1, g.logEntries[idx]), int(x)+20, int(y)+80+i*18)
	}
}

func (g *Game) drawColoredEntry(screen *ebiten.Image, text string, x, y int) {
	words := []string{}
	current := ""
	for _, char := range text {
		if char == ' ' || char == '[' || char == ']' || char == '|' {
			if current != "" {
				words = append(words, current)
				current = ""
			}
			words = append(words, string(char))
		} else {
			current += string(char)
		}
	}
	if current != "" {
		words = append(words, current)
	}

	curX := x
	for _, word := range words {
		// If it's a number like 0.00 or -0.00
		var val float64
		if _, err := fmt.Sscanf(word, "%f", &val); err == nil && (len(word) > 2 || word == "0") {
			// Draw background color
			col := color.RGBA{50, 50, 50, 255}
			if val > 0.01 {
				col = color.RGBA{20, 20, 100, 255} // Dim Blue
			} else if val < -0.01 {
				col = color.RGBA{100, 20, 20, 255} // Dim Red
			}
			vector.DrawFilledRect(screen, float32(curX), float32(y), float32(len(word)*6), 14, col, true)
		} else if word == "SNAPSHOT:" {
			vector.DrawFilledRect(screen, float32(curX), float32(y), float32(len(word)*6), 14, color.RGBA{100, 100, 0, 255}, true)
		}

		ebitenutil.DebugPrintAt(screen, word, curX, y)
		curX += len(word) * 6 // Rough character width for debug font
	}
}

func (g *Game) drawIntro(screen *ebiten.Image) {
	// Semi-transparent overlay
	vector.DrawFilledRect(screen, 0, 0, float32(g.w), float32(g.h), color.RGBA{0, 0, 0, 180}, true)

	// Popup box
	w, h := 500.0, 350.0
	x, y := (float64(g.w)-w)/2, (float64(g.h)-h)/2
	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h), color.RGBA{20, 20, 40, 255}, true)
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), 2, color.RGBA{100, 255, 100, 255}, true)

	ebitenutil.DebugPrintAt(screen, g.currentLevel.Name, int(x)+50, int(y)+30)
	ebitenutil.DebugPrintAt(screen, "OBJECTIVE:", int(x)+50, int(y)+70)

	// Draw multideline intro text
	ebitenutil.DebugPrintAt(screen, g.currentLevel.IntroText, int(x)+50, int(y)+100)

	ebitenutil.DebugPrintAt(screen, "Architecture:", int(x)+50, int(y)+240)
	archStr := fmt.Sprintf("Layers: %d (Hidden Sizes: %v)", len(g.currentLevel.HiddenSizes), g.currentLevel.HiddenSizes)
	ebitenutil.DebugPrintAt(screen, archStr, int(x)+50, int(y)+260)

	ebitenutil.DebugPrintAt(screen, "PRESS ENTER OR CLICK TO START", int(x)+130, int(y)+310)
}

func (g *Game) drawTension(screen *ebiten.Image) {
	targets := game.Get3DTargets(0.8, g.currentLevel.NumClasses)
	splitX := float64(g.w) * 0.65
	fH := float64(g.h)
	centerX, centerY := splitX/2.0, fH/2.0

	for _, s := range g.starfield.Stars {
		// Get current 3D pos (Partial based on visibleLayers)
		tx, ty, tz := g.grid.NN.TransformPartial(s.RawData, g.visibleLayers)

		// Interpolate with morphing if active
		mx := (s.BaseX/splitX-0.5)*2.0*(1-g.grid.MorphFactor) + tx*g.grid.MorphFactor
		my := (s.BaseY/fH-0.5)*2.0*(1-g.grid.MorphFactor) + ty*g.grid.MorphFactor
		mz := 0*(1-g.grid.MorphFactor) + tz*g.grid.MorphFactor

		// Project current pos
		sx, sy, _ := g.grid.Project3D(mx, my, mz, centerX, centerY)

		// Get target 3D pos
		target := targets[s.Class]
		// Project target pos
		tsx, tsy, _ := g.grid.Project3D(target[0], target[1], target[2], centerX, centerY)

		// Calculate distance to target (Tension)
		dist := math.Sqrt(math.Pow(mx-target[0], 2) + math.Pow(my-target[1], 2) + math.Pow(mz-target[2], 2))

		// Color based on tension: Red for high tension (far), White/Green for low (close)
		tensionFactor := dist / 1.5
		if tensionFactor > 1.0 {
			tensionFactor = 1.0
		}

		r := uint8(100 + 155*tensionFactor)
		gCol := uint8(100 * (1 - tensionFactor))
		b := uint8(255 * (1 - tensionFactor))
		alpha := uint8(40 + 60*tensionFactor) // Brighter if far

		// Pulse effect using time (mocking time with tickCounter)
		pulse := 1.0 + 0.3*math.Sin(float64(g.tickCounter)*0.1)
		alphaVal := float64(alpha) * pulse
		if alphaVal > 255 {
			alphaVal = 255
		}
		if alphaVal < 20 {
			alphaVal = 20
		}

		thickness := float32(0.5+1.5*tensionFactor) * float32(pulse)

		// Draw tension line (Force Ray) - Colorized by tension
		vector.StrokeLine(screen, float32(sx), float32(sy), float32(tsx), float32(tsy), thickness, color.RGBA{r, gCol, b, uint8(alphaVal)}, true)
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	g.w, g.h = outsideWidth, outsideHeight
	if g.grid != nil {
		g.grid.UpdateSize(float64(outsideWidth)*0.65, float64(outsideHeight))
	}
	return outsideWidth, outsideHeight
}

func (g *Game) saveSnapshot() {
	fname := fmt.Sprintf("net_level%d_acc%.0f.json", g.currentLevel.ID, g.accuracy*100)
	err := g.nn.Save(fname)
	if err != nil {
		g.addLog("Error saving snapshot: " + err.Error())
		return
	}
	g.addLog("SNAP: Network saved to " + fname)
}

func (g *Game) drawBoundaries(screen *ebiten.Image) {
	splitX := float64(g.w) * 0.65
	fH := float64(g.h)
	centerX, centerY := splitX/2.0, fH/2.0
	targets := game.Get3DTargets(0.8, g.currentLevel.NumClasses)

	// We sample the grid to find boundaries where the closest target changes
	step := 8.0
	for x := 0.0; x < splitX; x += step {
		for y := 0.0; y < fH; y += step {
			// Current point class
			bx := (x/splitX - 0.5) * 2.0
			by := (y/fH - 0.5) * 2.0

			tx, ty, tz := g.nn.Transform2D(bx, by)

			bestD := 1000.0
			class := -1
			for i, t := range targets {
				d := math.Sqrt(math.Pow(tx-t[0], 2) + math.Pow(ty-t[1], 2) + math.Pow(tz-t[2], 2))
				if d < bestD {
					bestD = d
					class = i
				}
			}

			// Check neighbor (to the right)
			bx2 := ((x+step)/splitX - 0.5) * 2.0
			tx2, ty2, _ := g.nn.Transform2D(bx2, by)
			class2 := -1
			bestD2 := 1000.0
			for i, t := range targets {
				d := math.Sqrt(math.Pow(tx2-t[0], 2) + math.Pow(ty2-t[1], 2) + math.Pow(0-t[2], 2))
				if d < bestD2 {
					bestD2 = d
					class2 = i
				}
			}

			if class != class2 && class != -1 {
				sx1, sy1, _ := g.grid.Project3D(tx, ty, tz, centerX, centerY)
				sx2, sy2, _ := g.grid.Project3D(tx2, ty2, 0, centerX, centerY)
				vector.StrokeLine(screen, float32(sx1), float32(sy1), float32(sx2), float32(sy2), 1, color.RGBA{255, 255, 255, 30}, true)
			}
		}
	}
}
func (g *Game) resetAuditSelection() {
	g.auditIdx = 0
	g.selLayer = -1
	g.selNode = -1
	g.selPrev = -1
}

func (g *Game) handleInput() {
	mx, my := ebiten.CursorPosition()
	click := inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft)

	// COLLAPSIBLE PANELS FOR CNN
	if g.currentLevel.EngineType == "cnn" && click {
		// PAD COLLAPSE
		if g.padCollapsed {
			btnX, btnY := 10, g.h-40
			w, h := 190, 30
			if mx >= btnX && mx <= btnX+w && my >= btnY && my <= btnY+h {
				g.padCollapsed = false
			}
		} else {
			centerX, centerY := float32(g.w)*0.32, float32(g.h/2)
			cellSize := float32(38.0)
			gridSize := cellSize * 8
			startX, startY := centerX - gridSize/2, centerY - gridSize/2
			tx := int(startX) - 10
			ty := int(startY) - 70
			btnX, btnY := tx, ty-35
			w, h := 150, 25
			if mx >= btnX && mx <= btnX+w && my >= btnY && my <= btnY+h {
				g.padCollapsed = true
			}
		}

		// PIZARRA COLLAPSE
		pizW := 190
		pizH := 30
		pizX := g.w - 200
		pizY := g.h - 40
		if g.pizarraCollapsed {
			if mx >= pizX && mx <= pizX+pizW && my >= pizY && my <= pizY+pizH {
				g.pizarraCollapsed = false
			}
		} else {
			splitX := float64(g.w) * 0.45
			w := float64(g.w) - splitX
			btnX, btnY := int(splitX)+int(w)-140, g.h-240+10
			if mx >= btnX && mx <= btnX+130 && my >= btnY && my <= btnY+25 {
				g.pizarraCollapsed = true
			}
		}
	}

	// NEW: Horizontal Layout (X=320, Y=10)
	depthBox := image.Rect(370, 28, 410, 46)
	widthBox := image.Rect(470, 28, 510, 46)

	if click {
		if mx >= depthBox.Min.X && mx <= depthBox.Max.X && my >= depthBox.Min.Y && my <= depthBox.Max.Y {
			g.focusBox = focusDepth
			if g.inputDepth == "" {
				g.inputDepth = fmt.Sprintf("%d", len(g.nn.Layers)-1)
			}
		} else if mx >= widthBox.Min.X && mx <= widthBox.Max.X && my >= widthBox.Min.Y && my <= widthBox.Max.Y {
			g.focusBox = focusWidth
			if g.inputWidth == "" {
				g.inputWidth = fmt.Sprintf("%d", len(g.nn.Layers[0].Weights[0]))
			}
		} else {
			g.focusBox = focusNone
		}
	}

	if g.focusBox != focusNone {
		var input *string
		if g.focusBox == focusDepth {
			input = &g.inputDepth
		} else {
			input = &g.inputWidth
		}

		// Backspace
		if inpututil.IsKeyJustPressed(ebiten.KeyBackspace) && len(*input) > 0 {
			*input = (*input)[:len(*input)-1]
		}

		// Apply (Enter)
		if inpututil.IsKeyJustPressed(ebiten.KeyEnter) || inpututil.IsKeyJustPressed(ebiten.KeyKPEnter) {
			g.rebuildArchitecture()
			g.focusBox = focusNone
		}

		// Numeric Input
		var chars []rune
		chars = ebiten.AppendInputChars(chars)
		for _, c := range chars {
			if c >= '0' && c <= '9' && len(*input) < 3 {
				*input += string(c)
			}
		}
	}
}

func (g *Game) getEngine(engineType string) nn.Engine {
	switch engineType {
	case "parallel":
		return &parallel.Engine{}
	case "avx":
		return &avx.Engine{}
	case "oneapi":
		return &oneapi.Engine{}
	case "level0":
		return &level0.Engine{}
	case "wasm":
		return &wasm.Engine{}
	default:
		return &standard.Engine{}
	}
}

func (g *Game) rebuildArchitecture() {
	depth := 0
	width := 0
	fmt.Sscanf(g.inputDepth, "%d", &depth)
	fmt.Sscanf(g.inputWidth, "%d", &width)

	if depth < 1 {
		depth = 1
	}
	if depth > 20 {
		depth = 20
	} // Safety
	if width < 1 {
		width = 1
	}
	if width > 100 {
		width = 100
	}

	hiddenSizes := make([]int, depth)
	for i := range hiddenSizes {
		hiddenSizes[i] = width
	}

	g.currentLevel.HiddenSizes = hiddenSizes
	g.ResetGame()
	g.addLog(fmt.Sprintf("ARCH: Rebuilt to %d layers x %d nodes", depth, width))
}

func main() {
	headlessTrain := flag.Bool("train-only", false, "Run purely in console to train and inspect CNN")
	flag.Parse()

	// Try loading from external file (Desktop), fallback to embedded if it fails (WASM/Portability)
	if err := game.LoadLevels("levels.json"); err != nil {
		log.Printf("Warning: Could not load external levels.json, using embedded data: %v", err)
		if err := game.LoadLevelsFromBytes(defaultLevelsJSON); err != nil {
			log.Fatalf("Fatal Error: Could not load embedded levels: %v", err)
		}
	}
	if len(game.Levels) == 0 {
		log.Fatalf("Fatal Error: No enabled levels found in configuration")
	}
	level := game.Levels[0]
	inputDims := level.InputDims
	if inputDims == 0 {
		inputDims = 2
	}
	numClasses := level.NumClasses
	if numClasses == 0 {
		numClasses = 3
	}

	// Initial network based on the first enabled level
	network := nn.NewNeuralNetwork(inputDims, level.HiddenSizes, numClasses)

	g := &Game{
		nn:            network,
		w:             screenWidth,
		h:             screenHeight,
		grid:          game.NewWarpGrid(network, float64(screenWidth)*0.65, float64(screenHeight), 20, 20),
		player:        game.NewPlayer(float64(screenWidth)*0.65/2, float64(screenHeight)/2),
		currentLevel:  level,
		starfield:     game.NewStarfield(level, 200, float64(screenWidth)*0.65, float64(screenHeight)),
		mode:          "Neural Evolution",
		simSpeed:      0.0001, // Ultra Slow start
		showIntro:     true,
		visibleLayers: len(level.HiddenSizes) + 1, // hidden + output
		targetMorph:   1.0,
		netX:          float64(screenWidth) * 0.825,
		netY:          float64(screenHeight) / 2.0,
		netZoom:       1.0,
		actName:       "Tanh",
		actFormula:    "tanh(x)",
	}
	g.initJSBridge()
	g.grid.MorphFactor = 1.0
	g.grid.VisibleLayers = g.visibleLayers
	
	// Ensure the first level is correctly initialized (especially for CNN)
	g.ResetGame()

	if *headlessTrain {
		runHeadlessTraining(g)
		return
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle("Neural Levels: XOR to Manifold")
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}

func runHeadlessTraining(g *Game) {
	fmt.Println("==================================================")
	fmt.Printf("🏃‍♂️ STARTING HEADLESS TRAINING FOR: %s\n", g.currentLevel.Name)
	fmt.Printf("⚙️  Target Accuracy: %.2f%%\n", g.currentLevel.TargetAcc*100)
	fmt.Println("==================================================")

	if len(g.nn.LayersV2) == 0 {
		log.Fatal("No CNN layers initialized")
	}

	start := time.Now()
	epochs := 0
	lr := 0.05

	var conv *nn.ConvLayer
	if c, ok := g.nn.LayersV2[0].(*nn.ConvLayer); ok {
		conv = c
	}
	
	printFilterSummary := func() {
		if conv == nil { return }
		fmt.Println("\n🧠 KERNEL WEIGHTS (3x3):")
		for i := 0; i < conv.OutChannels; i++ {
			fmt.Printf("--- Filter %d ---\n", i)
			k := conv.Kernels[i][0]
			kr, kc := k.Dims()
			for r := 0; r < kr; r++ {
				for c := 0; c < kc; c++ {
					fmt.Printf("%+5.2f ", k.At(r, c))
				}
				fmt.Println()
			}
		}
	}

	fmt.Println("Initial untrained filters:")
	printFilterSummary()

	targets := game.Get3DTargets(0.8, g.currentLevel.NumClasses)

	for {
		epochs++
		g.autoTrain(lr)
		
		// Calcular precisión real (Full Dataset)
		correct := 0
		for _, s := range g.starfield.Stars {
			out := g.nn.Forward(s.RawData)
			if len(out) < 3 {
				log.Fatal("Expected output size 3 for 3D coordinates")
			}
			tx, ty, tz := out[0], out[1], out[2]
			
			// Find closest target
			minDist := 1000.0
			closestClass := -1
			for i, t := range targets {
				d := math.Sqrt(math.Pow(tx-t[0], 2) + math.Pow(ty-t[1], 2) + math.Pow(tz-t[2], 2))
				if d < minDist {
					minDist = d
					closestClass = i
				}
			}
			if closestClass == s.Class {
				correct++
			}
		}
		
		acc := float64(correct) / float64(len(g.starfield.Stars))
		g.accuracy = acc

		if epochs%100 == 0 {
			fmt.Printf("Epoch %5d | Accuracy: %5.1f%%\n", epochs, acc*100)
		}

		if acc >= g.currentLevel.TargetAcc {
			fmt.Printf("🎯 CONVERGED at epoch %d with Accuracy: %.1f%%\n", epochs, acc*100)
			break
		}
		
		if epochs >= 5000 {
			fmt.Println("⚠️ TIMEOUT: Max epochs (5000) reached.")
			break
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\n✅ TRAINING COMPLETE in %v (Epochs: %d)\n", elapsed, epochs)
	
	fmt.Println("\n🧠 TRAINED KERNELS:")
	printFilterSummary()
}

func (g *Game) updateCNNAnimation() {
	if g.currentLevel.EngineType != "cnn" {
		return
	}
	
	// Automatic cycling: if not animating, wait and start again with a random star (if not drawing)
	if !g.isAnimatingConv {
		g.convFrame++ 
		if g.convFrame > 120 { // Delay between cycles
			g.isAnimatingConv = true
			g.convFrame = 0
			g.convChannelIdx = 0
			if !g.userIsDrawing && len(g.starfield.Stars) > 0 {
				s := g.starfield.Stars[rand.IntN(len(g.starfield.Stars))]
				g.convInput = s.RawData
			}
		}
		return
	}

	g.convFrame++
	// Max frames: 6x6 positions (36) * 4 frames per pos
	if g.convFrame > 36*4 {
		g.isAnimatingConv = false
		g.convFrame = 0 // Reset for the delay counter
	}
}

func (g *Game) drawCNNAnimation(screen *ebiten.Image) {
	if !g.isAnimatingConv || len(g.convInput) < 64 {
		return
	}

	// Move to Network Panel (Right Side)
	// We'll draw it below the network diagram or as an overlay
	startX, startY := float64(g.w)*0.68, 100.0
	cellSize := 10.0
	
	// Draw the 8x8 input grid again for focus
	vector.DrawFilledRect(screen, float32(startX)-5, float32(startY)-5, float32(8*cellSize)+10, float32(8*cellSize)+10, color.RGBA{0, 0, 0, 220}, true)
	ebitenutil.DebugPrintAt(screen, "CNN LIVE INFERENCE", int(startX), int(startY)-25)
	
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			val := g.convInput[i*8+j]
			v := uint8(255 * val)
			col := color.RGBA{v, v, v, 255}
			vector.DrawFilledRect(screen, float32(startX+float64(j)*cellSize), float32(startY+float64(i)*cellSize), float32(cellSize)-1, float32(cellSize)-1, col, true)
		}
	}

	// Sliding Scan Window (3x3)
	pos := g.convFrame / 4
	row := pos / 6
	col := pos % 6
	if row > 5 { row = 5 }

	wx, wy := startX+float64(col)*cellSize, startY+float64(row)*cellSize
	vector.StrokeRect(screen, float32(wx), float32(wy), float32(cellSize*3), float32(cellSize*3), 2, color.RGBA{255, 255, 0, 255}, true)
	
	// Pulse
	pulse := float32(math.Sin(float64(g.convFrame)*0.3)*5 + 5)
	vector.StrokeRect(screen, float32(wx)-pulse/2, float32(wy)-pulse/2, float32(cellSize*3)+pulse, float32(cellSize*3)+pulse, 1, color.RGBA{255, 255, 0, 100}, true)

	// Draw Feature Map target (Place it next to the input grid)
	fx, fy := startX + 120, startY + 20
	fcellSize := 10.0
	ebitenutil.DebugPrintAt(screen, "FEATURE MAP", int(fx), int(fy)-25)
	vector.DrawFilledRect(screen, float32(fx)-2, float32(fy)-2, float32(6*fcellSize)+4, float32(6*fcellSize)+4, color.RGBA{0, 0, 0, 200}, true)
	
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			currPos := i*6+j
			alpha := uint8(40)
			if currPos < pos {
				alpha = 255
			}
			vector.DrawFilledRect(screen, float32(fx+float64(j)*fcellSize), float32(fy+float64(i)*fcellSize), float32(fcellSize)-1, float32(fcellSize)-1, color.RGBA{50, 150, 255, alpha}, true)
		}
	}
	
	// Trailing line from scan to feature map
	vector.StrokeLine(screen, float32(wx+cellSize*1.5), float32(wy+cellSize*1.5), float32(fx+float64(col)*fcellSize), float32(fy+float64(row)*fcellSize), 1, color.RGBA{255, 255, 0, 120}, true)
}

func (g *Game) updateCNNProbing() {
	if g.currentLevel.EngineType != "cnn" {
		g.isProbing = false
		return
	}

	mx, my := ebiten.CursorPosition()
	fmx, _ := float64(mx), float64(my)
	splitX := float64(g.w) * 0.65

	// Si estamos en el visualizador 3D de la derecha, estamos en modo Frustum Hover
	if fmx > splitX {
		g.isProbing = true
	} else {
		g.isProbing = false
	}
}

func (g *Game) drawCNNDrawingPad(screen *ebiten.Image) {
	if g.padCollapsed {
		// Draw miniature button
		btnX, btnY := 10, g.h-40
		w, h := 190, 30
		vector.DrawFilledRect(screen, float32(btnX), float32(btnY), float32(w), float32(h), color.RGBA{40, 40, 80, 200}, true)
		vector.StrokeRect(screen, float32(btnX), float32(btnY), float32(w), float32(h), 1, color.RGBA{100, 150, 255, 255}, true)
		ebitenutil.DebugPrintAt(screen, ">> ABRIR LIENZO (Click)", btnX+15, btnY+8)
		return
	}

	centerX, centerY := float32(g.w)*0.32, float32(g.h/2)
	
	if len(g.convInput) < 64 {
		g.convInput = make([]float64, 64)
	}

	cellSize := float32(38.0)
	gridSize := cellSize * 8
	startX, startY := centerX - gridSize/2, centerY - gridSize/2
	
	tx := int(startX) - 10
	ty := int(startY) - 70
	
	ebitenutil.DebugPrintAt(screen, "INTERACTIVE DRAWING PAD (8x8)", tx, ty)
	ebitenutil.DebugPrintAt(screen, "Dibuja libremente un Patron (Click Izq: Pintar, Click Der: Borrar)", tx, ty+15)
	
	if g.userIsDrawing {
		ebitenutil.DebugPrintAt(screen, "MODO MANUAL: Escaneo Visual en Tiempo Real", tx, ty+30)
		ebitenutil.DebugPrintAt(screen, "(Pulsa ENTER para resetear al Dataset Automático)", tx, ty+45)
	} else {
		ebitenutil.DebugPrintAt(screen, "MODO AUTO: Reproduciendo galeria de estrellas automáticas...", tx, ty+30)
	}
	
	// Close button
	btnX, btnY := tx, ty-35
	w, h := 150, 25
	vector.DrawFilledRect(screen, float32(btnX), float32(btnY), float32(w), float32(h), color.RGBA{80, 30, 30, 200}, true)
	vector.StrokeRect(screen, float32(btnX), float32(btnY), float32(w), float32(h), 1, color.RGBA{255, 100, 100, 255}, true)
	ebitenutil.DebugPrintAt(screen, "<< CERRAR LIENZO", btnX+15, btnY+5)
	
	vector.DrawFilledRect(screen, startX-10, startY-10, gridSize+20, gridSize+20, color.RGBA{15, 20, 30, 220}, true)
	vector.StrokeRect(screen, startX-10, startY-10, gridSize+20, gridSize+20, 2, color.RGBA{50, 100, 255, 120}, true)
	
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			val := g.convInput[i*8+j]
			v := uint8(math.Min(255, math.Abs(val)*255))
			col := color.RGBA{v, v, v, 255}
			
			// Highlight edge if scanning
			vector.DrawFilledRect(screen, startX+float32(j)*cellSize, startY+float32(i)*cellSize, cellSize-1, cellSize-1, col, true)
		}
	}
	// Interactive drawing logic
	mx, my := ebiten.CursorPosition()
	fmx, fmy := float32(mx), float32(my)
	
	if fmx >= startX && fmx < startX+gridSize && fmy >= startY && fmy < startY+gridSize {
		// Calculate hit cell
		col := int((fmx - startX) / cellSize)
		row := int((fmy - startY) / cellSize)
		
		// Auto-train pause shortcut
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) || ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight) {
			if !g.userIsDrawing && len(g.starfield.Stars) > 0 {
				g.userIsDrawing = true
				// Clear grid when first starting to draw
				for i := range g.convInput {
					g.convInput[i] = 0.0
				}
			}
			
			if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
				g.convInput[row*8+col] = 1.0
			} else {
				g.convInput[row*8+col] = 0.0
			}
		}
		
		// Highlight cell hover
		vector.StrokeRect(screen, startX+float32(col)*cellSize, startY+float32(row)*cellSize, cellSize, cellSize, 2, color.RGBA{255, 255, 0, 150}, true)
	}
	
	// Reset to auto with ENTER
	if g.userIsDrawing && inpututil.IsKeyJustPressed(ebiten.KeyEnter) {
		g.userIsDrawing = false
	}
}
