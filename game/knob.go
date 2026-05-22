package game

import (
	"fmt"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

type Knob struct {
	X, Y      float64
	Label     string
	Value     *float64
	IsFocused bool
}

func NewKnob(x, y float64, label string, value *float64) *Knob {
	return &Knob{
		X:     x,
		Y:     y,
		Label: label,
		Value: value,
	}
}

func (k *Knob) Update(playerX, playerY float64) {
	// Check if player is near the knob
	dist := math.Sqrt(math.Pow(k.X-playerX, 2) + math.Pow(k.Y-playerY, 2))
	k.IsFocused = dist < 30

	if k.IsFocused {
		if ebiten.IsKeyPressed(ebiten.KeyUp) {
			*k.Value += 0.1
		}
		if ebiten.IsKeyPressed(ebiten.KeyDown) {
			*k.Value -= 0.1
		}
	}
}

func (k *Knob) Draw(screen *ebiten.Image) {
	c := color.RGBA{150, 150, 150, 255}
	if k.IsFocused {
		c = color.RGBA{255, 200, 0, 255}
	}

	// Draw the station
	vector.DrawFilledRect(screen, float32(k.X-15), float32(k.Y-5), 30, 10, c, true)
	vector.StrokeCircle(screen, float32(k.X), float32(k.Y-20), 10, 2, c, true)

	// Draw the needle
	angle := (*k.Value) * math.Pi / 2 // Just a visual representation
	nx := float32(k.X) + float32(math.Sin(angle)*8)
	ny := float32(k.Y-20) - float32(math.Cos(angle)*8)
	vector.StrokeLine(screen, float32(k.X), float32(k.Y-20), nx, ny, 2, color.White, true)

	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%s: %.2f", k.Label, *k.Value), int(k.X)-25, int(k.Y)+10)
}
