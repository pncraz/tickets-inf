package model

import (
	"github.com/pncraz/tickets-inf/features"
	"github.com/pncraz/tickets-inf/quantization"
)

type JSONModel struct {
	BowVocab       map[string]int `json:"bow_vocab"`
	EmbeddingVocab map[string]int `json:"embedding_vocab"`
	Embeddings     JSONEmbedding  `json:"embeddings"`
	Base           JSONBase       `json:"base"`
	Heads          JSONHeads      `json:"heads"`
}

type JSONEmbedding struct {
	Matrix     [][]float32 `json:"matrix,omitempty"`
	MatrixInt8 [][]int8    `json:"matrix_int8,omitempty"`
	Scales     []float32   `json:"scales,omitempty"`
}

type JSONBase struct {
	Dense1 JSONDenseLayer `json:"dense1"`
	Dense2 JSONDenseLayer `json:"dense2"`
}

type JSONHeads struct {
	Department JSONDenseLayer `json:"department"`
	Sentiment  JSONDenseLayer `json:"sentiment"`
	LeadIntent JSONDenseLayer `json:"lead_intent"`
	ChurnRisk  JSONDenseLayer `json:"churn_risk"`
	Intent     JSONDenseLayer `json:"intent"`
}

type JSONDenseLayer struct {
	Weights     [][]float32 `json:"weights,omitempty"`
	WeightsInt8 [][]int8    `json:"weights_int8,omitempty"`
	Scales      []float32   `json:"scales,omitempty"`
	Bias        []float32   `json:"bias"`
}

type LabelSet struct {
	Department []string `json:"department"`
	Sentiment  []string `json:"sentiment"`
	LeadIntent []string `json:"lead_intent"`
	ChurnRisk  []string `json:"churn_risk"`
	Intent     []string `json:"intent"`
}

type Model struct {
	BowVocab              map[string]int
	EmbeddingVocab        map[string]int
	Keywords              []string
	MaxTokens             int
	PaddingIndex          int
	UnknownTokenID        int
	HasUnknownToken       bool
	UseLog1pBow           bool
	UseLegacyKeywordFlags bool
	Preprocess            features.PreprocessConfig
	Labels                LabelSet
	Version               int
	ModelType             string
	DenseSize             int
	HiddenSize            int
	Embeddings            EmbeddingTable
	Base1                 DenseLayer
	Base2                 DenseLayer
	DepartmentHead        DenseLayer
	SentimentHead         DenseLayer
	LeadIntentHead        DenseLayer
	ChurnRiskHead         DenseLayer
	IntentHead            DenseLayer
}

type DenseLayer struct {
	In        int
	Out       int
	Weights   []float32
	Bias      []float32
	Quantized *quantization.Int8Matrix
}

type EmbeddingTable struct {
	Rows      int
	Dim       int
	Values    []float32
	Quantized *quantization.Int8Matrix
}

func (m *Model) SupportsFloatInference() bool {
	if len(m.Embeddings.Values) == 0 {
		return false
	}

	layers := []DenseLayer{
		m.Base1,
		m.Base2,
		m.DepartmentHead,
		m.SentimentHead,
		m.LeadIntentHead,
		m.ChurnRiskHead,
		m.IntentHead,
	}

	for _, layer := range layers {
		if len(layer.Weights) == 0 {
			return false
		}
	}

	return true
}

func (m *Model) SupportsQuantizedInference() bool {
	if m.Embeddings.Quantized == nil && len(m.Embeddings.Values) == 0 {
		return false
	}

	layers := []DenseLayer{
		m.Base1,
		m.Base2,
		m.DepartmentHead,
		m.SentimentHead,
		m.LeadIntentHead,
		m.ChurnRiskHead,
		m.IntentHead,
	}

	for _, layer := range layers {
		if layer.Quantized == nil && len(layer.Weights) == 0 {
			return false
		}
	}

	return true
}

func (m *Model) FeatureSize() int {
	return maxIndex(m.BowVocab) + 1 + m.KeywordSize() + m.Embeddings.Dim
}

func (m *Model) KeywordSize() int {
	if m.UseLegacyKeywordFlags {
		return len(features.KeywordFeatureNames)
	}
	return len(m.Keywords)
}

func (m *Model) PrepareQuantized() error {
	layers := []*DenseLayer{
		&m.Base1,
		&m.Base2,
		&m.DepartmentHead,
		&m.SentimentHead,
		&m.LeadIntentHead,
		&m.ChurnRiskHead,
		&m.IntentHead,
	}

	for _, layer := range layers {
		if layer.Quantized == nil && len(layer.Weights) > 0 {
			quantized, err := quantization.QuantizeFlat(layer.Weights, layer.Out, layer.In)
			if err != nil {
				return err
			}
			layer.Quantized = quantized
		}
	}

	if m.Embeddings.Quantized == nil && len(m.Embeddings.Values) > 0 {
		quantized, err := quantization.QuantizeFlat(m.Embeddings.Values, m.Embeddings.Rows, m.Embeddings.Dim)
		if err != nil {
			return err
		}
		m.Embeddings.Quantized = quantized
	}

	return nil
}

func (m *Model) ParameterBytes(useQuantized bool) int {
	total := 0

	total += layerBytes(m.Base1, useQuantized)
	total += layerBytes(m.Base2, useQuantized)
	total += layerBytes(m.DepartmentHead, useQuantized)
	total += layerBytes(m.SentimentHead, useQuantized)
	total += layerBytes(m.LeadIntentHead, useQuantized)
	total += layerBytes(m.ChurnRiskHead, useQuantized)
	total += layerBytes(m.IntentHead, useQuantized)

	if useQuantized && m.Embeddings.Quantized != nil {
		total += m.Embeddings.Quantized.SizeBytes()
	} else {
		total += len(m.Embeddings.Values) * 4
	}

	return total
}

func layerBytes(layer DenseLayer, useQuantized bool) int {
	total := len(layer.Bias) * 4
	if useQuantized && layer.Quantized != nil {
		return total + layer.Quantized.SizeBytes()
	}
	return total + len(layer.Weights)*4
}

func maxIndex(vocab map[string]int) int {
	max := -1
	for _, index := range vocab {
		if index > max {
			max = index
		}
	}
	return max
}
