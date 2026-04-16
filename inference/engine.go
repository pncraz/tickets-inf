package inference

import (
	"fmt"
	"io"
	"sync/atomic"

	"github.com/octate/tickets-inf/features"
	"github.com/octate/tickets-inf/model"
	"github.com/octate/tickets-inf/utils"
)

type Config struct {
	UseQuantized bool
	Debug        bool
	IntentTopK   int
	DebugWriter  io.Writer
}

type Engine struct {
	model     *model.Model
	extractor *features.Extractor
	config    Config
}

var defaultEngine atomic.Pointer[Engine]

func NewEngine(loadedModel *model.Model, config Config) (*Engine, error) {
	if loadedModel == nil {
		return nil, fmt.Errorf("model must not be nil")
	}

	if config.IntentTopK <= 0 {
		config.IntentTopK = 3
	}
	if config.DebugWriter == nil {
		config.DebugWriter = io.Discard
	}

	if !config.UseQuantized && !loadedModel.SupportsFloatInference() {
		if !loadedModel.SupportsQuantizedInference() {
			return nil, fmt.Errorf("model does not contain a usable float32 or quantized inference representation")
		}
		config.UseQuantized = true
	}
	if config.UseQuantized {
		if !loadedModel.SupportsQuantizedInference() {
			return nil, fmt.Errorf("model does not contain a usable quantized inference representation")
		}
		if err := loadedModel.PrepareQuantized(); err != nil {
			return nil, fmt.Errorf("prepare quantized model: %w", err)
		}
	}

	return &Engine{
		model: loadedModel,
		extractor: features.NewConfiguredExtractor(features.Config{
			BowVocab:              loadedModel.BowVocab,
			EmbeddingVocab:        loadedModel.EmbeddingVocab,
			KeywordPhrases:        loadedModel.Keywords,
			UseLegacyKeywordFlags: loadedModel.UseLegacyKeywordFlags,
			MaxTokens:             loadedModel.MaxTokens,
			UseLog1pBow:           loadedModel.UseLog1pBow,
			UnknownTokenID:        loadedModel.UnknownTokenID,
			HasUnknownToken:       loadedModel.HasUnknownToken,
			PreprocessConfig:      loadedModel.Preprocess,
		}),
		config: config,
	}, nil
}

func LoadEngineFromFile(path string, config Config) (*Engine, error) {
	loadedModel, err := model.LoadFile(path)
	if err != nil {
		return nil, err
	}
	return NewEngine(loadedModel, config)
}

func SetDefaultEngine(engine *Engine) {
	defaultEngine.Store(engine)
}

func Predict(text string) PredictionResult {
	engine := defaultEngine.Load()
	if engine == nil {
		return PredictionResult{Error: "default engine not configured"}
	}
	return engine.Predict(text)
}

func (engine *Engine) Model() *model.Model {
	return engine.model
}

func (engine *Engine) Labels() model.LabelSet {
	return engine.model.Labels
}

func (engine *Engine) Predict(text string) PredictionResult {
	parts := engine.extractor.Extract(text)
	embeddingVector := engine.model.Embeddings.Average(parts.EmbeddingTokenIDs, nil, engine.config.UseQuantized)
	featureVector := parts.FinalVector(embeddingVector)

	result := PredictionResult{}
	if engine.config.Debug {
		result.Debug = &DebugInfo{
			NormalizedText: parts.NormalizedText,
			Tokens:         append([]string(nil), parts.Tokens...),
			FeatureVector:  append([]float32(nil), featureVector...),
		}
		fmt.Fprintf(engine.config.DebugWriter, "normalized=%q tokens=%v feature_vector=%v\n", parts.NormalizedText, parts.Tokens, featureVector)
	}

	base1 := engine.model.Base1.Forward(featureVector, nil, engine.config.UseQuantized)
	model.ReLUInPlace(base1)
	base2 := engine.model.Base2.Forward(base1, nil, engine.config.UseQuantized)
	model.ReLUInPlace(base2)

	result.Department = engine.softmaxPrediction(engine.model.DepartmentHead, base2, engine.model.Labels.Department)
	result.Sentiment = engine.softmaxPrediction(engine.model.SentimentHead, base2, engine.model.Labels.Sentiment)
	result.LeadIntent = engine.softmaxPrediction(engine.model.LeadIntentHead, base2, engine.model.Labels.LeadIntent)
	result.ChurnRisk = engine.binaryPrediction(engine.model.ChurnRiskHead, base2, engine.model.Labels.ChurnRisk)
	result.Intent, result.IntentTopK = engine.softmaxPredictionWithTopK(engine.model.IntentHead, base2, engine.model.Labels.Intent, engine.config.IntentTopK)
	result.HumanReadable = GenerateHumanReadable(result)

	return result
}

func (engine *Engine) softmaxPrediction(layer model.DenseLayer, input []float32, labels []string) HeadPrediction {
	prediction, _ := engine.softmaxPredictionWithTopK(layer, input, labels, 0)
	return prediction
}

func (engine *Engine) softmaxPredictionWithTopK(layer model.DenseLayer, input []float32, labels []string, topK int) (HeadPrediction, []RankedPrediction) {
	logits := layer.Forward(input, nil, engine.config.UseQuantized)
	scores := model.Softmax(logits)
	bestIndex, bestScore := utils.ArgMax(scores)

	prediction := HeadPrediction{
		Label:      labels[bestIndex],
		Confidence: bestScore,
		Scores:     utils.ScoreMap(labels, scores),
	}

	if topK <= 0 {
		return prediction, nil
	}

	indices := utils.TopKIndices(scores, topK)
	ranked := make([]RankedPrediction, 0, len(indices))
	for _, index := range indices {
		ranked = append(ranked, RankedPrediction{
			Label:      labels[index],
			Confidence: scores[index],
		})
	}

	return prediction, ranked
}

func (engine *Engine) binaryPrediction(layer model.DenseLayer, input []float32, labels []string) HeadPrediction {
	logit := layer.Forward(input, nil, engine.config.UseQuantized)[0]
	positiveProbability := model.Sigmoid(logit)
	negativeProbability := 1 - positiveProbability

	label := labels[0]
	confidence := negativeProbability
	if positiveProbability >= 0.5 {
		label = labels[1]
		confidence = positiveProbability
	}

	return HeadPrediction{
		Label:      label,
		Confidence: confidence,
		Scores: map[string]float32{
			labels[0]: negativeProbability,
			labels[1]: positiveProbability,
		},
	}
}
