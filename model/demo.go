package model

import "github.com/octate/tickets-inf/features"

func NewDemoModel() (*Model, error) {
	return BuildFromJSONModel(NewDemoJSONModel())
}

func NewDemoJSONModel() JSONModel {
	bowVocab := map[string]int{
		"refund":    0,
		"charge":    1,
		"billing":   2,
		"invoice":   3,
		"broken":    4,
		"error":     5,
		"cancel":    6,
		"pricing":   7,
		"buy":       8,
		"demo":      9,
		"delivery":  10,
		"late":      11,
		"delay":     12,
		"complaint": 13,
		"thanks":    14,
		"great":     15,
		"resolved":  16,
		"money":     17,
		"working":   18,
		"bug":       19,
	}

	embeddingVocab := cloneVocab(bowVocab)
	embeddings := makeMatrix(len(embeddingVocab), 4)

	for _, token := range []string{"refund", "charge", "billing", "invoice", "money"} {
		embeddings[embeddingVocab[token]][0] = 1
	}
	for _, token := range []string{"broken", "error", "working", "bug"} {
		embeddings[embeddingVocab[token]][1] = 1
	}
	for _, token := range []string{"pricing", "buy", "demo"} {
		embeddings[embeddingVocab[token]][2] = 1
	}
	for _, token := range []string{"delivery", "late", "delay"} {
		embeddings[embeddingVocab[token]][3] = 1
	}
	for _, token := range []string{"thanks", "great", "resolved"} {
		embeddings[embeddingVocab[token]][2] = 0.25
	}

	featureSize := len(bowVocab) + len(features.KeywordFeatureNames) + 4
	base1Weights := makeMatrix(64, featureSize)
	for i := 0; i < featureSize; i++ {
		base1Weights[i][i] = 1
	}

	base2Weights := makeMatrix(32, 64)
	for i := 0; i < featureSize && i < 32; i++ {
		base2Weights[i][i] = 1
	}

	kwOffset := len(bowVocab)
	embOffset := kwOffset + len(features.KeywordFeatureNames)

	bow := func(token string) int { return bowVocab[token] }
	kw := func(name string) int {
		for i, featureName := range features.KeywordFeatureNames {
			if featureName == name {
				return kwOffset + i
			}
		}
		panic("unknown keyword feature: " + name)
	}
	emb := func(index int) int { return embOffset + index }

	departmentWeights := makeMatrix(4, 32)
	departmentBias := []float32{0.1, 0.1, 0.1, 0.35}
	departmentWeights[0][kw("has_refund")] = 2.3
	departmentWeights[0][kw("has_money")] = 1.8
	departmentWeights[0][bow("charge")] = 1.6
	departmentWeights[0][bow("billing")] = 1.6
	departmentWeights[0][bow("invoice")] = 1.2
	departmentWeights[0][emb(0)] = 1.0
	departmentWeights[1][kw("has_not_working")] = 2.4
	departmentWeights[1][bow("broken")] = 1.7
	departmentWeights[1][bow("error")] = 1.7
	departmentWeights[1][bow("bug")] = 1.2
	departmentWeights[1][emb(1)] = 1.0
	departmentWeights[2][bow("pricing")] = 1.9
	departmentWeights[2][bow("buy")] = 1.9
	departmentWeights[2][bow("demo")] = 1.9
	departmentWeights[2][emb(2)] = 1.0
	departmentWeights[3][kw("has_delay")] = 1.2
	departmentWeights[3][bow("delivery")] = 1.8
	departmentWeights[3][bow("late")] = 1.2
	departmentWeights[3][bow("delay")] = 1.2
	departmentWeights[3][emb(3)] = 1.0

	sentimentWeights := makeMatrix(3, 32)
	sentimentBias := []float32{0.1, 0.45, 0.1}
	sentimentWeights[0][bow("thanks")] = 1.6
	sentimentWeights[0][bow("great")] = 1.4
	sentimentWeights[0][bow("resolved")] = 1.3
	sentimentWeights[0][emb(2)] = 0.3
	sentimentWeights[1][bow("delivery")] = 0.5
	sentimentWeights[1][bow("pricing")] = 0.6
	sentimentWeights[1][bow("demo")] = 0.6
	sentimentWeights[2][kw("has_refund")] = 0.9
	sentimentWeights[2][kw("has_cancel")] = 1.2
	sentimentWeights[2][kw("has_delay")] = 1.2
	sentimentWeights[2][kw("has_not_working")] = 1.8
	sentimentWeights[2][bow("complaint")] = 1.9
	sentimentWeights[2][bow("broken")] = 1.1
	sentimentWeights[2][bow("error")] = 1.1

	leadIntentWeights := makeMatrix(3, 32)
	leadIntentBias := []float32{0.15, 0.45, 0.45}
	leadIntentWeights[0][bow("pricing")] = 1.9
	leadIntentWeights[0][bow("buy")] = 2.0
	leadIntentWeights[0][bow("demo")] = 2.0
	leadIntentWeights[0][emb(2)] = 1.0
	leadIntentWeights[1][kw("has_delay")] = 0.8
	leadIntentWeights[1][bow("delivery")] = 0.8
	leadIntentWeights[1][bow("billing")] = 0.7
	leadIntentWeights[1][kw("has_refund")] = 0.4
	leadIntentWeights[2][kw("has_cancel")] = 0.9
	leadIntentWeights[2][kw("has_not_working")] = 0.7
	leadIntentWeights[2][bow("complaint")] = 1.0

	churnWeights := makeMatrix(1, 32)
	churnBias := []float32{-1.2}
	churnWeights[0][kw("has_refund")] = 1.7
	churnWeights[0][kw("has_cancel")] = 2.1
	churnWeights[0][kw("has_delay")] = 0.7
	churnWeights[0][kw("has_not_working")] = 1.6
	churnWeights[0][bow("complaint")] = 1.7
	churnWeights[0][bow("thanks")] = -1.0
	churnWeights[0][bow("resolved")] = -0.9

	intentWeights := makeMatrix(4, 32)
	intentBias := []float32{0.1, 0.1, 0.1, 0.45}
	intentWeights[0][kw("has_refund")] = 2.5
	intentWeights[0][kw("has_money")] = 1.2
	intentWeights[0][bow("charge")] = 1.1
	intentWeights[0][bow("billing")] = 0.8
	intentWeights[0][bow("invoice")] = 0.8
	intentWeights[1][kw("has_delay")] = 2.1
	intentWeights[1][bow("delivery")] = 1.8
	intentWeights[1][bow("late")] = 1.4
	intentWeights[1][bow("delay")] = 1.4
	intentWeights[1][emb(3)] = 1.0
	intentWeights[2][kw("has_not_working")] = 2.2
	intentWeights[2][bow("complaint")] = 1.9
	intentWeights[2][bow("broken")] = 1.2
	intentWeights[2][bow("error")] = 1.2
	intentWeights[2][kw("has_cancel")] = 0.4
	intentWeights[3][bow("pricing")] = 1.0
	intentWeights[3][bow("demo")] = 1.0
	intentWeights[3][bow("thanks")] = 0.6
	intentWeights[3][bow("great")] = 0.6

	return JSONModel{
		BowVocab:       bowVocab,
		EmbeddingVocab: embeddingVocab,
		Embeddings: JSONEmbedding{
			Matrix: embeddings,
		},
		Base: JSONBase{
			Dense1: JSONDenseLayer{
				Weights: base1Weights,
				Bias:    make([]float32, 64),
			},
			Dense2: JSONDenseLayer{
				Weights: base2Weights,
				Bias:    make([]float32, 32),
			},
		},
		Heads: JSONHeads{
			Department: JSONDenseLayer{
				Weights: departmentWeights,
				Bias:    departmentBias,
			},
			Sentiment: JSONDenseLayer{
				Weights: sentimentWeights,
				Bias:    sentimentBias,
			},
			LeadIntent: JSONDenseLayer{
				Weights: leadIntentWeights,
				Bias:    leadIntentBias,
			},
			ChurnRisk: JSONDenseLayer{
				Weights: churnWeights,
				Bias:    churnBias,
			},
			Intent: JSONDenseLayer{
				Weights: intentWeights,
				Bias:    intentBias,
			},
		},
	}
}

func makeMatrix(rows, cols int) [][]float32 {
	matrix := make([][]float32, rows)
	for row := range matrix {
		matrix[row] = make([]float32, cols)
	}
	return matrix
}
