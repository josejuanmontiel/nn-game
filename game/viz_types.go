package game

type VizMode int

const (
	VizModeFixed VizMode = iota // Standard [0,1,2] projection
	VizModePCA                  // Principal Component Analysis
	VizModeParallel             // Parallel Coordinates
	VizModeSPLOM                // Scatter Plot Matrix
	VizModeSchlegel             // 5D Hypercube projection
	VizModeGlyph                // 3D + Structural Glyphs (Dim 4/5)
)

func (m VizMode) String() string {
	switch m {
	case VizModeFixed:
		return "Fixed Projection [0,1,2]"
	case VizModePCA:
		return "PCA (Principal Component Analysis)"
	case VizModeParallel:
		return "Parallel Coordinates"
	case VizModeSPLOM:
		return "SPLOM (Scatter Plot Matrix)"
	case VizModeSchlegel:
		return "Schlegel Diagram (5D Hypercube)"
	case VizModeGlyph:
		return "Structural Glyphs (Dim 4 & 5)"
	default:
		return "Unknown"
	}
}

// ProjectionMatrix maps 5D hidden nodes to 3D screen space
type ProjectionMatrix [3][5]float64

func IdentityProjection() ProjectionMatrix {
	var m ProjectionMatrix
	m[0][0] = 1.0
	m[1][1] = 1.0
	m[2][2] = 1.0
	return m
}
