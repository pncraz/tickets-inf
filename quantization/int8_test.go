package quantization

import "testing"

func TestQuantizeNestedPreservesRowShape(t *testing.T) {
	matrix, err := QuantizeNested([][]float32{
		{0.5, -1.0, 1.5},
		{0, 0, 0},
	})
	if err != nil {
		t.Fatalf("quantize nested: %v", err)
	}

	if matrix.Rows != 2 || matrix.Cols != 3 {
		t.Fatalf("unexpected matrix shape: %dx%d", matrix.Rows, matrix.Cols)
	}

	row := matrix.DequantizeRow(0, nil)
	if len(row) != 3 {
		t.Fatalf("unexpected row length: %d", len(row))
	}

	if row[0] < 0.45 || row[0] > 0.55 {
		t.Fatalf("unexpected dequantized value: %f", row[0])
	}
}
