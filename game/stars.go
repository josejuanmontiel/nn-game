package game

import (
	"encoding/json"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

type Star struct {
	BaseX, BaseY float64   // Projected 2D screen coords (legacy/HUD use)
	WorldX, WorldY, WorldZ float64 // Projected 3D input coords (Level Zero)
	RawData      []float64 // Original N-dimensional input
	Color        color.RGBA
	Class        int // 0 or 1
	IsCorrect    bool
}

type Starfield struct {
	Stars []*Star
}

var ClassColors = []color.RGBA{
	{255, 50, 50, 255},  // Red
	{50, 255, 50, 255},  // Green
	{50, 100, 255, 255}, // Blue
	{255, 255, 50, 255}, // Yellow
	{255, 50, 255, 255}, // Magenta
}

type GeneratorConfig struct {
	Type     string             `json:"Type"` // "gaussian", "ring", "spiral", "box"
	Class    int                `json:"Class"`
	Count    int                `json:"Count"`
	Settings map[string]float64 `json:"Settings"`
}

type DemoStep struct {
	Name        string             `json:"Name"`
	Description string             `json:"Description"`
	Weights     [][]float64        `json:"Weights,omitempty"`
	Biases      []float64          `json:"Biases,omitempty"`
}

type Level struct {
	ID              int               `json:"ID"`
	Enabled         bool              `json:"Enabled"`
	StartPaused     bool              `json:"StartPaused"`
	Name            string            `json:"Name"`
	NumClasses      int               `json:"NumClasses"`
	TargetAcc       float64           `json:"TargetAcc"`
	HiddenSizes     []int             `json:"HiddenSizes"`
	IntroText       string            `json:"IntroText"`
	Generators      []GeneratorConfig `json:"Generators"`
	DemoSteps       []DemoStep        `json:"DemoSteps,omitempty"`
	WeightInitScale float64           `json:"WeightInitScale"`
	BiasInitScale   float64           `json:"BiasInitScale"`
	EngineType      string            `json:"EngineType"`
	VizMode         string            `json:"VizMode,omitempty"`
	InputDims       int               `json:"InputDims,omitempty"` // D >= 2
}

var Levels []Level

func LoadLevels(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return LoadLevelsFromBytes(data)
}

func LoadLevelsFromBytes(data []byte) error {
	var allLevels []Level
	if err := json.Unmarshal(data, &allLevels); err != nil {
		return err
	}
	// Filter enabled levels
	var enabledLevels []Level
	for _, l := range allLevels {
		if l.Enabled {
			enabledLevels = append(enabledLevels, l)
		}
	}
	Levels = enabledLevels
	log.Printf("Loaded %d levels (filtered from %d) from JSON data", len(Levels), len(allLevels))
	return nil
}

func (level Level) GetClassAt(x, y float64) int {
	bestClass := 0
	minDist := 1e9

	for _, gen := range level.Generators {
		dist := 1e9
		switch gen.Type {
		case "gaussian":
			cx := gen.Settings["cx"]
			cy := gen.Settings["cy"]
			dist = math.Sqrt(math.Pow(x-cx, 2) + math.Pow(y-cy, 2))
		case "ring":
			cx := gen.Settings["cx"]
			cy := gen.Settings["cy"]
			r1 := gen.Settings["r1"]
			r2 := gen.Settings["r2"]
			d := math.Sqrt(math.Pow(x-cx, 2) + math.Pow(y-cy, 2))
			dist = math.Abs(d - (r1+r2)/2)
		case "spiral":
			cx := gen.Settings["cx"]
			cy := gen.Settings["cy"]
			radScale := gen.Settings["radiusScale"]
			dx, dy := x-cx, y-cy
			d := math.Sqrt(dx*dx + dy*dy)
			angle := math.Atan2(dy, dx)
			if angle < 0 {
				angle += 2 * math.Pi
			}
			thetaS := (d - 20) / radScale
			phase := math.Mod(angle-thetaS, 2*math.Pi)
			if phase < 0 {
				phase += 2 * math.Pi
			}
			targetPhase := gen.Settings["phase"]
			dist = math.Abs(phase - targetPhase)
			if dist > math.Pi {
				dist = 2*math.Pi - dist
			}
		case "box":
			x1, y1 := gen.Settings["x1"], gen.Settings["y1"]
			x2, y2 := gen.Settings["x2"], gen.Settings["y2"]
			if x >= x1 && x <= x2 && y >= y1 && y <= y2 {
				dist = 0
			}
		}

		if dist < minDist {
			minDist = dist
			bestClass = gen.Class
		}
	}
	return bestClass
}

func (sf *Starfield) AddStar(x, y float64, class int) {
	sf.Stars = append(sf.Stars, &Star{
		BaseX: x,
		BaseY: y,
		Class: class,
		Color: ClassColors[class],
	})
}

func NewStarfield(level Level, count int, panelW, panelH float64) *Starfield {
	sf := &Starfield{}
	dims := level.InputDims
	if dims < 2 { dims = 2 }

	var allPoints []DataPoint
	for _, gen := range level.Generators {
		if gen.Count <= 0 {
			gen.Count = count / len(level.Generators)
		}
		points := GenerateInput(gen, dims, panelW, panelH)
		allPoints = append(allPoints, points...)
	}

	// Project high-D to 3D for visual "Level Zero"
	ProjectPoints(allPoints, dims)

	for _, p := range allPoints {
		sf.Stars = append(sf.Stars, &Star{
			WorldX:  p.Projected[0],
			WorldY:  p.Projected[1],
			WorldZ:  p.Projected[2],
			RawData: p.RawData,
			Class:   p.Class,
			Color:   ClassColors[p.Class%len(ClassColors)],
		})
	}

	return sf
}

func (sf *Starfield) Update(grid *WarpGrid, numClasses int) {
	// 3D Target vertices
	targets := Get3DTargets(0.8, numClasses)

	for _, s := range sf.Stars {
		tx, ty, tz := grid.NN.Transform(s.RawData)

		// Check if point is closest to its 3D class target
		minDist := 1000.0
		closestClass := -1
		for i, t := range targets {
			d := math.Sqrt(math.Pow(tx-t[0], 2) + math.Pow(ty-t[1], 2) + math.Pow(tz-t[2], 2))
			if d < minDist {
				minDist = d
				closestClass = i
			}
		}
		s.IsCorrect = closestClass == s.Class
	}
}

func Get3DTargets(r float64, numClasses int) [][]float64 {
	targets := make([][]float64, numClasses)
	if numClasses <= 1 {
		return [][]float64{{0, 0, r}}
	}

	for i := 0; i < numClasses; i++ {
		// Better distribution: avoid poles and use golden ratio for phi
		// We want targets to be as distinct as possible in 3D
		offset := 1.0 / float64(numClasses)
		phi := math.Acos(-1.0 + offset + (2.0-2.0*offset)*float64(i)/float64(numClasses-1))
		theta := (1.0 + math.Sqrt(5.0)) * math.Pi * float64(i)

		targets[i] = []float64{
			r * math.Sin(phi) * math.Cos(theta),
			r * math.Sin(phi) * math.Sin(theta),
			r * math.Cos(phi),
		}
	}
	return targets
}

func (sf *Starfield) Draw(screen *ebiten.Image, grid *WarpGrid) {
	isCNN := grid.NN != nil && grid.NN.EngineType == "cnn"

	for _, s := range sf.Stars {
		tx, ty, tz := grid.GetTransformedScreenFromWorld(s.WorldX, s.WorldY, s.WorldZ)

		// Depth factor for stars: fainter and smaller if far
		depthFactor := 1.0 + tz*0.5 // tz is ~[-1, 1] relative depth
		if depthFactor < 0.2 {
			depthFactor = 0.2
		}
		if depthFactor > 1.8 {
			depthFactor = 1.8
		}

		c := s.Color
		size := float32(3.0 * depthFactor)

		alpha := float64(c.A) * depthFactor
		if alpha > 255 {
			alpha = 255
		}
		if alpha < 20 {
			alpha = 20
		}

		if s.IsCorrect {
			// Success Glow
			vector.DrawFilledCircle(screen, float32(tx), float32(ty), size+2, color.RGBA{255, 255, 255, uint8(40 * depthFactor)}, true)
		} else {
			// Incorrect stars are dimmer
			alpha *= 0.4
		}
		c.A = uint8(alpha)

		if isCNN {
			// Draw X or O symbol
			sSize := size * 2.5
			if s.Class == 0 {
				// Draw X
				vector.StrokeLine(screen, float32(tx)-sSize, float32(ty)-sSize, float32(tx)+sSize, float32(ty)+sSize, 1.5, c, true)
				vector.StrokeLine(screen, float32(tx)-sSize, float32(ty)+sSize, float32(tx)+sSize, float32(ty)-sSize, 1.5, c, true)
			} else {
				// Draw O
				vector.StrokeCircle(screen, float32(tx), float32(ty), sSize, 1.5, c, true)
			}
		} else {
			vector.DrawFilledCircle(screen, float32(tx), float32(ty), size, c, true)
		}
	}
}

func (sf *Starfield) GetAccuracy() float64 {
	if len(sf.Stars) == 0 {
		return 0
	}
	correct := 0
	for _, s := range sf.Stars {
		if s.IsCorrect {
			correct++
		}
	}
	return float64(correct) / float64(len(sf.Stars))
}
