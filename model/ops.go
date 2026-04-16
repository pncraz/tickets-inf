package model

import "math"

func (layer DenseLayer) Forward(input []float32, dst []float32, useQuantized bool) []float32 {
	if dst == nil || cap(dst) < layer.Out {
		dst = make([]float32, layer.Out)
	} else {
		dst = dst[:layer.Out]
	}

	if useQuantized && layer.Quantized != nil {
		for row := 0; row < layer.Out; row++ {
			dst[row] = layer.Bias[row] + layer.Quantized.DotRow(row, input)
		}
		return dst
	}

	for row := 0; row < layer.Out; row++ {
		sum := layer.Bias[row]
		offset := row * layer.In
		for col := 0; col < layer.In; col++ {
			sum += layer.Weights[offset+col] * input[col]
		}
		dst[row] = sum
	}

	return dst
}

func (table EmbeddingTable) Average(tokenIDs []int, dst []float32, useQuantized bool) []float32 {
	if dst == nil || cap(dst) < table.Dim {
		dst = make([]float32, table.Dim)
	} else {
		dst = dst[:table.Dim]
		for i := range dst {
			dst[i] = 0
		}
	}

	if len(tokenIDs) == 0 {
		return dst
	}

	validCount := 0
	if useQuantized && table.Quantized != nil {
		for _, tokenID := range tokenIDs {
			if tokenID < 0 || tokenID >= table.Rows {
				continue
			}
			start := tokenID * table.Dim
			scale := table.Quantized.Scales[tokenID]
			for col := 0; col < table.Dim; col++ {
				dst[col] += float32(table.Quantized.Values[start+col]) * scale
			}
			validCount++
		}
	} else {
		for _, tokenID := range tokenIDs {
			if tokenID < 0 || tokenID >= table.Rows {
				continue
			}
			start := tokenID * table.Dim
			for col := 0; col < table.Dim; col++ {
				dst[col] += table.Values[start+col]
			}
			validCount++
		}
	}

	if validCount == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return dst
	}

	scale := 1 / float32(validCount)
	for i := range dst {
		dst[i] *= scale
	}
	return dst
}

func ReLUInPlace(values []float32) {
	for i, value := range values {
		if value < 0 {
			values[i] = 0
		}
	}
}

func Softmax(logits []float32) []float32 {
	output := make([]float32, len(logits))
	return SoftmaxInto(logits, output)
}

func SoftmaxInto(logits []float32, dst []float32) []float32 {
	if len(logits) == 0 {
		return dst[:0]
	}
	if dst == nil || cap(dst) < len(logits) {
		dst = make([]float32, len(logits))
	} else {
		dst = dst[:len(logits)]
	}

	maxLogit := logits[0]
	for _, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
		}
	}

	var sum float64
	for i, value := range logits {
		expValue := math.Exp(float64(value - maxLogit))
		dst[i] = float32(expValue)
		sum += expValue
	}

	invSum := float32(1 / sum)
	for i := range dst {
		dst[i] *= invSum
	}

	return dst
}

func Sigmoid(value float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-value))))
}
