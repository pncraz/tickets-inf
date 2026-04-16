package model

import "github.com/octate/tickets-inf/features"

type ExportedModelJSON struct {
	Version    int                       `json:"version"`
	Metadata   ExportedModelMetadata     `json:"metadata"`
	Preprocess features.PreprocessConfig `json:"preprocess"`
	Vocab      ExportedModelVocab        `json:"vocab"`
	Labels     LabelSet                  `json:"labels"`
	Embedding  ExportedEmbedding         `json:"embedding"`
	Layers     ExportedLayers            `json:"layers"`
	Heads      ExportedHeads             `json:"heads"`
}

type ExportedModelMetadata struct {
	ModelType    string `json:"model_type"`
	DenseSize    int    `json:"dense_size"`
	EmbeddingDim int    `json:"embedding_dim"`
	HiddenSize   int    `json:"hidden_size"`
}

type ExportedModelVocab struct {
	BowVocab       map[string]int `json:"bow_vocab"`
	EmbeddingVocab map[string]int `json:"embedding_vocab"`
	Keywords       []string       `json:"keywords"`
	MaxTokens      int            `json:"max_tokens"`
}

type ExportedEmbedding struct {
	PaddingIdx int         `json:"padding_idx"`
	Weights    [][]float32 `json:"weights"`
}

type ExportedOp struct {
	Type string `json:"type"`
}

type ExportedLinearLayer struct {
	Type        string    `json:"type"`
	InFeatures  int       `json:"in_features"`
	OutFeatures int       `json:"out_features"`
	Weight      [][]int8  `json:"weight"`
	Scale       []float32 `json:"scale"`
	Bias        []float32 `json:"bias"`
}

type ExportedLayers struct {
	Base0 ExportedLinearLayer `json:"base_0"`
	Base1 ExportedOp          `json:"base_1"`
	Base2 ExportedLinearLayer `json:"base_2"`
	Base3 ExportedOp          `json:"base_3"`
}

type ExportedHeads struct {
	Department ExportedLinearLayer `json:"department"`
	Sentiment  ExportedLinearLayer `json:"sentiment"`
	LeadIntent ExportedLinearLayer `json:"lead_intent"`
	ChurnRisk  ExportedLinearLayer `json:"churn_risk"`
	Intent     ExportedLinearLayer `json:"intent"`
}
