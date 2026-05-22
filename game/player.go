package game

import (
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

type Player struct {
	X, Y   float64
	Radius float32
}

func NewPlayer(x, y float64) *Player {
	return &Player{
		X:      x,
		Y:      y,
		Radius: 6,
	}
}

func (p *Player) Update(grid *WarpGrid) {
	const speed = 2.0

	// 2D Movement (WASD only to free Arrows for Adjustment)
	if ebiten.IsKeyPressed(ebiten.KeyA) {
		p.X -= speed
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) {
		p.X += speed
	}
	if ebiten.IsKeyPressed(ebiten.KeyW) {
		p.Y -= speed
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) {
		p.Y += speed
	}

	// Clamp to world
	if p.X < 0 {
		p.X = 0
	}
	if p.X > 800 {
		p.X = 800
	}
	if p.Y < 0 {
		p.Y = 0
	}
	if p.Y > 600 {
		p.Y = 600
	}
}

func (p *Player) Draw(screen *ebiten.Image, grid *WarpGrid) {
	// Draw in transformed space
	tx, ty, tz := grid.GetTransformedScreen(p.X, p.Y)

	// Depth scaling for player
	depthFactor := 1.0 + tz*0.5
	if depthFactor < 0.2 {
		depthFactor = 0.2
	}

	vector.DrawFilledCircle(screen, float32(tx), float32(ty), p.Radius*float32(depthFactor), color.White, true)

	// Draw small dot in "base" space for reference
	vector.DrawFilledCircle(screen, float32(p.X), float32(p.Y), 2, color.RGBA{100, 100, 100, 100}, true)
}
