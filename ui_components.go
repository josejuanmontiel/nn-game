package main

import (
	"fmt"
	"image/color"

	"11-juego-final/game"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func (g *Game) drawStats(screen *ebiten.Image) {
	// HUD
	status := "MANUAL"
	if g.isTraining {
		status = "EVOLVING..."
	}
	ebitenutil.DebugPrint(screen, fmt.Sprintf(
		"LEVEL %d: %s\nMode: %s [%s]\nAccuracy: %.1f%% / Target: %.1f%%\nIterations: %d\n\nControls:\n- MOUSE: Hover network / WHEEL: Adjust Val\n- WASD: Move Probe / [ ]: Sim Speed (v: %.4f)\n- S: SIMULATE / T: Auto-train / ENTER: Step\n- L: Log / J: JUMP / N: NEXT\n- X: Cyce Viz Backend / V: Morph Toggle / C: Center",
		g.currentLevel.ID, g.currentLevel.Name, g.mode, status, g.accuracy*100, g.currentLevel.TargetAcc*100, g.iterCount, g.simSpeed))

	// NEW: Mouse Highlight and Probe Trail (Skip projection for CNN)
	splitX := float64(g.w) * 0.65
	fH := float64(g.h)
	mx, my := ebiten.CursorPosition()
	if g.currentLevel.EngineType != "cnn" && float64(mx) < splitX && float64(my) < fH {
		vector.StrokeCircle(screen, float32(mx), float32(my), 10, 1, color.RGBA{150, 150, 150, 100}, true)
		ebitenutil.DebugPrintAt(screen, "SHIFT+CLICK: LANCER", mx+12, my-5)

		// Projection of current cursor
		bx := (float64(mx)/splitX - 0.5) * 2.0
		by := (float64(my)/fH - 0.5) * 2.0
		tx, ty, tz := g.nn.Transform2D(bx, by)
		stx, sty, _ := g.grid.Project3D(tx, ty, tz, splitX/2, fH/2)

		// 3D Ghost Target
		vector.StrokeCircle(screen, float32(stx), float32(sty), 4, 1, color.RGBA{255, 255, 255, 150}, true)
	}

	// Draw active shots (Skip projection for CNN)
	if g.currentLevel.EngineType != "cnn" {
		for _, s := range g.shots {
			vector.DrawFilledCircle(screen, float32(s.X), float32(s.Y), 3, color.White, true)

			bx := (s.X/splitX - 0.5) * 2.0
			by := (s.Y/fH - 0.5) * 2.0
			tx, ty, tz := g.nn.Transform2D(bx, by)
			stx, sty, _ := g.grid.Project3D(tx, ty, tz, splitX/2, fH/2)

			c := game.ClassColors[s.Class]
			vector.DrawFilledCircle(screen, float32(stx), float32(sty), 5, c, true)
			vector.StrokeCircle(screen, float32(stx), float32(sty), 7, 1, color.White, true)
			vector.StrokeLine(screen, float32(s.X), float32(s.Y), float32(stx), float32(sty), 1, color.RGBA{255, 255, 255, 50}, true)
		}
	}

	// Architecture Inputs - Positioned at X=320 to avoid stats
	if len(g.nn.Layers) > 0 {
		currentDepth := len(g.nn.Layers) - 1
		currentWidth := 0
		if len(g.nn.Layers[0].Weights) > 0 && len(g.nn.Layers[0].Weights[0]) > 0 {
			currentWidth = len(g.nn.Layers[0].Weights[0])
		} else {
			currentWidth = g.nn.InSize
		}

		ebitenutil.DebugPrintAt(screen, "NN ARCHITECTURE CONFIG:", 320, 10)
		depthX, depthY := 320, 30
		ebitenutil.DebugPrintAt(screen, "Depth:", depthX, depthY)
		g.drawInputBox(screen, 370, depthY-2, g.inputDepth, currentDepth, g.focusBox == focusDepth)

		widthX, widthY := 420, 30
		ebitenutil.DebugPrintAt(screen, "Width:", widthX, widthY)
		g.drawInputBox(screen, 470, widthY-2, g.inputWidth, currentWidth, g.focusBox == focusWidth)
		ebitenutil.DebugPrintAt(screen, "(Enter to Apply)", 320, 50)
	}
}

func (g *Game) drawInputBox(screen *ebiten.Image, x, y int, input string, current int, focused bool) {
	w, h := 40, 18
	bg := color.RGBA{40, 40, 60, 255}
	border := color.RGBA{100, 100, 150, 255}
	if focused {
		bg = color.RGBA{60, 60, 90, 255}
		border = color.RGBA{150, 255, 150, 255}
		// Pulse if focused
		if (g.tickCounter/20)%2 == 0 {
			border = color.RGBA{200, 255, 200, 255}
		}
	}

	vector.DrawFilledRect(screen, float32(x), float32(y), float32(w), float32(h), bg, true)
	vector.StrokeRect(screen, float32(x), float32(y), float32(w), float32(h), 1, border, true)

	displayText := input
	if !focused {
		displayText = fmt.Sprintf("%d", current)
	}

	ebitenutil.DebugPrintAt(screen, displayText, x+5, y+2)
}
