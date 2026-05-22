package game

import (
	"math"
	"math/rand/v2"
)

// DataPoint represents a single generated point in high-dimensional space
type DataPoint struct {
	ID        int
	RawData   []float64
	Projected [3]float64
	Class     int
}

// GenerateInput is the main entry point for creating datasets
func GenerateInput(config GeneratorConfig, dims int, panelW, panelH float64) []DataPoint {
	switch config.Type {
	case "hyper_sphere":
		return generateHyperSphere(config, dims)
	case "swiss_roll":
		return generateSwissRoll(config, dims)
	case "clusters":
		return generateClusters(config, dims)
	case "xor_extended":
		return generateXORExtended(config, dims)
	case "shapes":
		return generateShapes(config, dims)
	case "gaussian", "ring", "spiral", "box":
		return generateLegacy2D(config, panelW, panelH)
	default:
		return generateDefault(config, dims)
	}
}

func generateHyperSphere(config GeneratorConfig, dims int) []DataPoint {
	count := config.Count
	r := config.Settings["radius"]
	if r == 0 { r = 1.0 }
	noise := config.Settings["noise"]
	
	points := make([]DataPoint, count)
	for i := 0; i < count; i++ {
		raw := make([]float64, dims)
		norm := 0.0
		for j := 0; j < dims; j++ {
			raw[j] = rand.NormFloat64()
			norm += raw[j] * raw[j]
		}
		norm = math.Sqrt(norm)
		
		// Map to surface
		actualR := r + (rand.Float64()-0.5)*noise
		for j := 0; j < dims; j++ {
			raw[j] = (raw[j] / norm) * actualR
		}
		
		points[i] = DataPoint{
			ID:      i,
			RawData: raw,
			Class:   config.Class,
		}
	}
	return points
}

func generateSwissRoll(config GeneratorConfig, dims int) []DataPoint {
	count := config.Count
	points := make([]DataPoint, count)
	
	for i := 0; i < count; i++ {
		// Classic swiss roll is (t*cos(t), t*sin(t), h, ...)
		t := 1.5 * math.Pi * (1 + 2*rand.Float64())
		raw := make([]float64, dims)
		raw[0] = t * math.Cos(t)
		raw[1] = t * math.Sin(t)
		
		for j := 2; j < dims; j++ {
			raw[j] = rand.Float64() * 20.0 // Breadth in other dimensions
		}
		
		points[i] = DataPoint{
			ID:      i,
			RawData: raw,
			Class:   config.Class,
		}
	}
	return points
}

func generateClusters(config GeneratorConfig, dims int) []DataPoint {
	count := config.Count
	dist := config.Settings["distance"]
	if dist == 0 { dist = 2.0 }
	
	points := make([]DataPoint, count)
	center := make([]float64, dims)
	// Shift center based on class
	if config.Class%2 == 0 {
		center[0] = dist
	} else {
		center[0] = -dist
	}

	for i := 0; i < count; i++ {
		raw := make([]float64, dims)
		for j := 0; j < dims; j++ {
			raw[j] = center[j] + rand.NormFloat64()*0.5
		}
		points[i] = DataPoint{
			ID:      i,
			RawData: raw,
			Class:   config.Class,
		}
	}
	return points
}

func generateXORExtended(config GeneratorConfig, dims int) []DataPoint {
	count := config.Count
	points := make([]DataPoint, count)
	
	for i := 0; i < count; i++ {
		raw := make([]float64, dims)
		for j := 0; j < dims; j++ {
			raw[j] = (rand.Float64()*2 - 1) * 2.0
		}
		
		// XOR logic: Class depends on quadrants of first two dims
		class := 0
		if (raw[0] > 0 && raw[1] > 0) || (raw[0] < 0 && raw[1] < 0) {
			class = 1
		}
		
		points[i] = DataPoint{
			ID:      i,
			RawData: raw,
			Class:   class,
		}
	}
	return points
}

func generateLegacy2D(gen GeneratorConfig, panelW, panelH float64) []DataPoint {
	count := gen.Count
	points := make([]DataPoint, count)

	for i := 0; i < count; i++ {
		var x, y float64
		switch gen.Type {
		case "gaussian":
			cx := gen.Settings["cx"]
			cy := gen.Settings["cy"]
			std := gen.Settings["std"]
			x = cx + rand.NormFloat64()*std
			y = cy + rand.NormFloat64()*std
		case "ring":
			cx := gen.Settings["cx"]
			cy := gen.Settings["cy"]
			r1 := gen.Settings["r1"]
			r2 := gen.Settings["r2"]
			angle := rand.Float64() * 2 * math.Pi
			dist := r1 + rand.Float64()*(r2-r1)
			x = cx + dist*math.Cos(angle)
			y = cy + dist*math.Sin(angle)
		case "spiral":
			cx := gen.Settings["cx"]
			cy := gen.Settings["cy"]
			radScale := gen.Settings["radiusScale"]
			rots := gen.Settings["rotations"]
			width := gen.Settings["width"]
			phase := gen.Settings["phase"]
			angle := rand.Float64() * rots * math.Pi
			dist := 20.0 + angle*radScale + (rand.Float64()-0.5)*width
			x = cx + dist*math.Cos(angle+phase)
			y = cy + dist*math.Sin(angle+phase)
		case "box":
			x1, y1 := gen.Settings["x1"], gen.Settings["y1"]
			x2, y2 := gen.Settings["x2"], gen.Settings["y2"]
			x = x1 + rand.Float64()*(x2-x1)
			y = y1 + rand.Float64()*(y2-y1)
		}

		// Normalize coordinates for RawData (NN input)
		// Original logic mapped [0, panelW] to [-1, 1]
		bx := (x/panelW - 0.5) * 2.0
		by := (y/panelH - 0.5) * 2.0

		points[i] = DataPoint{
			ID:      i,
			RawData: []float64{bx, by},
			Class:   gen.Class,
		}
		// Projected is already normalized base for legacy
		points[i].Projected[0] = bx
		points[i].Projected[1] = by
		points[i].Projected[2] = 0
	}
	return points
}

func generateDefault(config GeneratorConfig, dims int) []DataPoint {
	// Fallback to gaussian around center
	count := config.Count
	points := make([]DataPoint, count)
	for i := 0; i < count; i++ {
		raw := make([]float64, dims)
		for j := 0; j < dims; j++ {
			raw[j] = rand.NormFloat64()
		}
		points[i] = DataPoint{
			ID:      i,
			RawData: raw,
			Class:   config.Class,
		}
	}
	return points
}

// ProjectPoints maps high-D points to 3D for visualization
func ProjectPoints(points []DataPoint, dims int) {
	if dims <= 3 {
		for i := range points {
			for j := 0; j < 3; j++ {
				if j < dims {
					points[i].Projected[j] = points[i].RawData[j]
				} else {
					points[i].Projected[j] = 0
				}
			}
		}
		return
	}

	// For D > 3, use the existing PCA helper
	rawMatrix := make([][]float64, len(points))
	for i := range points {
		rawMatrix[i] = points[i].RawData
	}
	
	proj := ComputePCA(rawMatrix)
	for i := range points {
		tx, ty, tz := 0.0, 0.0, 0.0
		for j := 0; j < dims && j < 5; j++ {
			tx += points[i].RawData[j] * proj[0][j]
			ty += points[i].RawData[j] * proj[1][j]
			tz += points[i].RawData[j] * proj[2][j]
		}
		points[i].Projected[0] = tx
		points[i].Projected[1] = ty
		points[i].Projected[2] = tz
	}
}

func generateShapes(config GeneratorConfig, dims int) []DataPoint {
	count := config.Count
	noise := config.Settings["noise"]
	points := make([]DataPoint, count)

	for i := 0; i < count; i++ {
		raw := make([]float64, 64) // 8x8
		if config.Class == 0 {
			// "X" Pattern
			for j := 0; j < 8; j++ {
				raw[j*8+j] = 1.0
				raw[j*8+(7-j)] = 1.0
			}
		} else {
			// "O" Pattern
			for j := 2; j < 6; j++ {
				raw[0*8+j] = 1.0
				raw[7*8+j] = 1.0
				raw[j*8+0] = 1.0
				raw[j*8+7] = 1.0
			}
			raw[1*8+1] = 1.0
			raw[1*8+6] = 1.0
			raw[6*8+1] = 1.0
			raw[6*8+6] = 1.0
		}

		// Add noise
		for j := range raw {
			raw[j] += (rand.Float64() - 0.5) * noise
			if raw[j] > 1.0 {
				raw[j] = 1.0
			}
			if raw[j] < 0.0 {
				raw[j] = 0.0
			}
		}

		points[i] = DataPoint{
			ID:      i,
			RawData: raw,
			Class:   config.Class,
		}
	}
	return points
}
