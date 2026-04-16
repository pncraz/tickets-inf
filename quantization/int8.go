package quantization

import (
	"fmt"
	"math"
)

type Int8Matrix struct {
	Rows   int
	Cols   int
	Values []int8
	Scales []float32
}

func NewInt8MatrixFromNested(values [][]int8, scales []float32) (*Int8Matrix, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("quantized matrix must have at least one row")
	}
	if len(scales) != len(values) {
		return nil, fmt.Errorf("quantized matrix row/scale mismatch: rows=%d scales=%d", len(values), len(scales))
	}

	cols := len(values[0])
	if cols == 0 {
		return nil, fmt.Errorf("quantized matrix must have at least one column")
	}

	flat := make([]int8, 0, len(values)*cols)
	for row, current := range values {
		if len(current) != cols {
			return nil, fmt.Errorf("quantized matrix row %d has inconsistent width", row)
		}
		flat = append(flat, current...)
	}

	clonedScales := make([]float32, len(scales))
	copy(clonedScales, scales)

	return &Int8Matrix{
		Rows:   len(values),
		Cols:   cols,
		Values: flat,
		Scales: clonedScales,
	}, nil
}

func QuantizeNested(values [][]float32) (*Int8Matrix, error) {
	if len(values) == 0 {
		return nil, fmt.Errorf("matrix must have at least one row")
	}
	cols := len(values[0])
	if cols == 0 {
		return nil, fmt.Errorf("matrix must have at least one column")
	}
	for row, current := range values {
		if len(current) != cols {
			return nil, fmt.Errorf("row %d has inconsistent width", row)
		}
	}

	flat := make([]float32, 0, len(values)*cols)
	for _, current := range values {
		flat = append(flat, current...)
	}

	return QuantizeFlat(flat, len(values), cols)
}

func QuantizeFlat(values []float32, rows, cols int) (*Int8Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid quantized shape rows=%d cols=%d", rows, cols)
	}
	if len(values) != rows*cols {
		return nil, fmt.Errorf("flat matrix size mismatch: got=%d want=%d", len(values), rows*cols)
	}

	result := &Int8Matrix{
		Rows:   rows,
		Cols:   cols,
		Values: make([]int8, len(values)),
		Scales: make([]float32, rows),
	}

	for row := 0; row < rows; row++ {
		start := row * cols
		end := start + cols
		scale := rowScale(values[start:end])
		result.Scales[row] = scale
		for col, value := range values[start:end] {
			quantized := int(math.Round(float64(value / scale)))
			switch {
			case quantized > 127:
				quantized = 127
			case quantized < -127:
				quantized = -127
			}
			result.Values[start+col] = int8(quantized)
		}
	}

	return result, nil
}

func (m *Int8Matrix) DotRow(row int, input []float32) float32 {
	start := row * m.Cols
	scale := m.Scales[row]

	var sum float32
	for col := 0; col < m.Cols; col++ {
		sum += float32(m.Values[start+col]) * scale * input[col]
	}
	return sum
}

func (m *Int8Matrix) DequantizeRow(row int, dst []float32) []float32 {
	if dst == nil || cap(dst) < m.Cols {
		dst = make([]float32, m.Cols)
	} else {
		dst = dst[:m.Cols]
	}

	start := row * m.Cols
	scale := m.Scales[row]
	for col := 0; col < m.Cols; col++ {
		dst[col] = float32(m.Values[start+col]) * scale
	}
	return dst
}

func (m *Int8Matrix) SizeBytes() int {
	return len(m.Values) + len(m.Scales)*4
}

func rowScale(values []float32) float32 {
	var maxAbs float32
	for _, value := range values {
		if abs := float32(math.Abs(float64(value))); abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return 1
	}
	return maxAbs / 127
}
