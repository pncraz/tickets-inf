package inference

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/octate/tickets-inf/features"
	"github.com/octate/tickets-inf/model"
	"github.com/octate/tickets-inf/quantization"
)

func TestPredictFromJSONModel(t *testing.T) {
	engine := mustJSONBackedEngine(t, Config{IntentTopK: 2})

	testCases := []struct {
		name       string
		text       string
		department string
		intent     string
		sentiment  string
		leadIntent string
		churnRisk  string
	}{
		{
			name:       "billing refund flow",
			text:       "Please refund the money and cancel this invoice",
			department: "billing",
			intent:     "refund",
			sentiment:  "negative",
			leadIntent: "low",
			churnRisk:  "high",
		},
		{
			name:       "technical complaint",
			text:       "The app is not working, broken, and showing an error",
			department: "tech",
			intent:     "complaint",
			sentiment:  "negative",
			leadIntent: "low",
			churnRisk:  "high",
		},
		{
			name:       "sales lead",
			text:       "Can I get pricing and a demo before I buy",
			department: "sales",
			intent:     "other",
			sentiment:  "neutral",
			leadIntent: "high",
			churnRisk:  "low",
		},
		{
			name:       "delivery query",
			text:       "My delivery is late and delayed",
			department: "general",
			intent:     "delivery_query",
			sentiment:  "negative",
			leadIntent: "medium",
			churnRisk:  "low",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			result := engine.Predict(testCase.text)
			assertLabel(t, "department", result.Department.Label, testCase.department)
			assertLabel(t, "intent", result.Intent.Label, testCase.intent)
			assertLabel(t, "sentiment", result.Sentiment.Label, testCase.sentiment)
			assertLabel(t, "lead_intent", result.LeadIntent.Label, testCase.leadIntent)
			assertLabel(t, "churn_risk", result.ChurnRisk.Label, testCase.churnRisk)
			if len(result.IntentTopK) != 2 {
				t.Fatalf("unexpected top-k result length: %d", len(result.IntentTopK))
			}
		})
	}
}

func TestPredictQuantizedAndDebug(t *testing.T) {
	var debugOutput bytes.Buffer
	engine := mustJSONBackedEngine(t, Config{
		UseQuantized: true,
		Debug:        true,
		DebugWriter:  &debugOutput,
		IntentTopK:   3,
	})

	result := engine.Predict("Please refund the money and cancel this invoice")
	if result.Debug == nil || len(result.Debug.FeatureVector) == 0 {
		t.Fatalf("expected debug feature vector to be returned")
	}
	if debugOutput.Len() == 0 {
		t.Fatalf("expected debug output to be written")
	}
	if result.Intent.Label != "refund" {
		t.Fatalf("unexpected quantized prediction: %s", result.Intent.Label)
	}
	if result.HumanReadable == nil || result.HumanReadable.Summary == "" || result.HumanReadable.ReplyDraft == "" {
		t.Fatalf("expected human-readable text to be populated")
	}
}

func TestPackageLevelPredict(t *testing.T) {
	engine := mustJSONBackedEngine(t, Config{})
	SetDefaultEngine(engine)

	result := Predict("Can I get pricing and a demo before I buy")
	if result.Error != "" {
		t.Fatalf("unexpected package-level error: %s", result.Error)
	}
	if result.Department.Label != "sales" {
		t.Fatalf("unexpected default-engine prediction: %s", result.Department.Label)
	}
}

func TestQuantizedOnlyModelAutoEnablesQuantizedMode(t *testing.T) {
	path := writeModelJSON(t, quantizedOnlyDemoJSON(t))

	engine, err := LoadEngineFromFile(path, Config{})
	if err != nil {
		t.Fatalf("expected quantized-only model to auto-enable quantized inference: %v", err)
	}
	if !engine.config.UseQuantized {
		t.Fatalf("expected quantized inference to be enabled automatically")
	}
}

func TestQuantizedOnlyModelPredictsWhenEnabled(t *testing.T) {
	path := writeModelJSON(t, quantizedOnlyDemoJSON(t))

	engine, err := LoadEngineFromFile(path, Config{UseQuantized: true})
	if err != nil {
		t.Fatalf("load quantized-only engine: %v", err)
	}

	result := engine.Predict("My delivery is late and delayed")
	if result.Intent.Label != "delivery_query" {
		t.Fatalf("unexpected quantized-only prediction: %s", result.Intent.Label)
	}
}

func TestExportedTrainingArtifactPredicts(t *testing.T) {
	path := writeExportedModelJSON(t, trainingStyleDemoJSON(t))

	engine, err := LoadEngineFromFile(path, Config{})
	if err != nil {
		t.Fatalf("load exported training artifact: %v", err)
	}
	if !engine.config.UseQuantized {
		t.Fatalf("expected exported artifact to use quantized inference automatically")
	}

	result := engine.Predict("Please refund the money and cancel this invoice")
	assertLabel(t, "department", result.Department.Label, "billing")
	assertLabel(t, "intent", result.Intent.Label, "refund")
	assertLabel(t, "sentiment", result.Sentiment.Label, "negative")
	assertLabel(t, "lead_intent", result.LeadIntent.Label, "low")
	assertLabel(t, "churn_risk", result.ChurnRisk.Label, "high")
	if result.HumanReadable == nil {
		t.Fatalf("expected human-readable narrative for exported artifact")
	}
	if result.HumanReadable.ManualReviewNote != "" {
		t.Fatalf("did not expect manual review note for a confident prediction: %q", result.HumanReadable.ManualReviewNote)
	}
}

func TestGenerateHumanReadableNarrative(t *testing.T) {
	result := PredictionResult{
		Department: HeadPrediction{Label: "sales", Confidence: 0.81},
		Sentiment:  HeadPrediction{Label: "positive", Confidence: 0.55},
		LeadIntent: HeadPrediction{Label: "high", Confidence: 0.97},
		ChurnRisk:  HeadPrediction{Label: "low", Confidence: 0.92},
		Intent:     HeadPrediction{Label: "pricing_inquiry", Confidence: 0.86},
		IntentTopK: []RankedPrediction{
			{Label: "pricing_inquiry", Confidence: 0.86},
			{Label: "cancellation", Confidence: 0.08},
			{Label: "order_tracking", Confidence: 0.02},
		},
	}

	narrative := GenerateHumanReadable(result)
	if narrative == nil {
		t.Fatalf("expected narrative to be generated")
	}
	if !containsText(narrative.Summary, "sales") || !containsText(narrative.Summary, "pricing inquiry") {
		t.Fatalf("unexpected summary: %q", narrative.Summary)
	}
	if !containsText(narrative.TriageNote, "Secondary intent candidates are cancellation") {
		t.Fatalf("unexpected triage note: %q", narrative.TriageNote)
	}
	if !containsText(narrative.ReplyDraft, "sales team") {
		t.Fatalf("unexpected reply draft: %q", narrative.ReplyDraft)
	}
}

func TestGenerateHumanReadableRequestsManualReviewForAmbiguousCases(t *testing.T) {
	result := PredictionResult{
		Department: HeadPrediction{Label: "general", Confidence: 0.52},
		Sentiment:  HeadPrediction{Label: "neutral", Confidence: 0.44},
		LeadIntent: HeadPrediction{Label: "medium", Confidence: 0.51},
		ChurnRisk:  HeadPrediction{Label: "low", Confidence: 0.53},
		Intent:     HeadPrediction{Label: "general_query", Confidence: 0.41},
		IntentTopK: []RankedPrediction{
			{Label: "general_query", Confidence: 0.41},
			{Label: "complaint", Confidence: 0.33},
			{Label: "billing_issue", Confidence: 0.18},
		},
	}

	narrative := GenerateHumanReadable(result)
	if narrative == nil || narrative.ManualReviewNote == "" {
		t.Fatalf("expected manual review note for ambiguous prediction")
	}
}

func BenchmarkPredict(b *testing.B) {
	engine := mustJSONBackedEngine(b, Config{UseQuantized: true})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.Predict("Please refund the money and cancel this invoice")
	}
}

func mustJSONBackedEngine(tb testing.TB, config Config) *Engine {
	tb.Helper()

	path := writeModelJSON(tb, model.NewDemoJSONModel())

	engine, err := LoadEngineFromFile(path, config)
	if err != nil {
		tb.Fatalf("load engine: %v", err)
	}
	return engine
}

func assertLabel(t *testing.T, field string, got string, want string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s mismatch: got=%s want=%s", field, got, want)
	}
}

func containsText(value string, fragment string) bool {
	return bytes.Contains([]byte(value), []byte(fragment))
}

func writeModelJSON(tb testing.TB, spec model.JSONModel) string {
	tb.Helper()

	payload, err := json.Marshal(spec)
	if err != nil {
		tb.Fatalf("marshal model json: %v", err)
	}

	path := filepath.Join(tb.TempDir(), "demo_model.json")
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		tb.Fatalf("write model json: %v", err)
	}

	return path
}

func writeExportedModelJSON(tb testing.TB, spec model.ExportedModelJSON) string {
	tb.Helper()

	payload, err := json.Marshal(spec)
	if err != nil {
		tb.Fatalf("marshal exported model json: %v", err)
	}

	path := filepath.Join(tb.TempDir(), "exported_model.json")
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		tb.Fatalf("write exported model json: %v", err)
	}

	return path
}

func quantizedOnlyDemoJSON(tb testing.TB) model.JSONModel {
	tb.Helper()

	spec := model.NewDemoJSONModel()
	var err error

	spec.Embeddings.MatrixInt8, spec.Embeddings.Scales, err = quantizeMatrix(spec.Embeddings.Matrix)
	if err != nil {
		tb.Fatalf("quantize embeddings: %v", err)
	}
	spec.Embeddings.Matrix = nil

	spec.Base.Dense1, err = quantizedDense(spec.Base.Dense1)
	if err != nil {
		tb.Fatalf("quantize dense1: %v", err)
	}
	spec.Base.Dense2, err = quantizedDense(spec.Base.Dense2)
	if err != nil {
		tb.Fatalf("quantize dense2: %v", err)
	}
	spec.Heads.Department, err = quantizedDense(spec.Heads.Department)
	if err != nil {
		tb.Fatalf("quantize department head: %v", err)
	}
	spec.Heads.Sentiment, err = quantizedDense(spec.Heads.Sentiment)
	if err != nil {
		tb.Fatalf("quantize sentiment head: %v", err)
	}
	spec.Heads.LeadIntent, err = quantizedDense(spec.Heads.LeadIntent)
	if err != nil {
		tb.Fatalf("quantize lead_intent head: %v", err)
	}
	spec.Heads.ChurnRisk, err = quantizedDense(spec.Heads.ChurnRisk)
	if err != nil {
		tb.Fatalf("quantize churn_risk head: %v", err)
	}
	spec.Heads.Intent, err = quantizedDense(spec.Heads.Intent)
	if err != nil {
		tb.Fatalf("quantize intent head: %v", err)
	}

	return spec
}

func quantizedDense(layer model.JSONDenseLayer) (model.JSONDenseLayer, error) {
	weightsInt8, scales, err := quantizeMatrix(layer.Weights)
	if err != nil {
		return model.JSONDenseLayer{}, err
	}

	layer.WeightsInt8 = weightsInt8
	layer.Scales = scales
	layer.Weights = nil
	return layer, nil
}

func quantizeMatrix(values [][]float32) ([][]int8, []float32, error) {
	matrix, err := quantization.QuantizeNested(values)
	if err != nil {
		return nil, nil, err
	}

	nested := make([][]int8, matrix.Rows)
	for row := 0; row < matrix.Rows; row++ {
		start := row * matrix.Cols
		nested[row] = append([]int8(nil), matrix.Values[start:start+matrix.Cols]...)
	}

	return nested, append([]float32(nil), matrix.Scales...), nil
}

func trainingStyleDemoJSON(tb testing.TB) model.ExportedModelJSON {
	tb.Helper()

	legacy := model.NewDemoJSONModel()
	return model.ExportedModelJSON{
		Version: 1,
		Metadata: model.ExportedModelMetadata{
			ModelType:    "hybrid_tiny_multitask",
			DenseSize:    len(legacy.BowVocab) + 5,
			EmbeddingDim: len(legacy.Embeddings.Matrix[0]),
			HiddenSize:   64,
		},
		Preprocess: features.PreprocessConfig{
			Lowercase:      true,
			ReplaceURLs:    "<url>",
			ReplaceEmails:  "<email>",
			ReplaceNumbers: "<num>",
			HinglishMap: map[string]string{
				"paisa":  "money",
				"kharab": "broken",
				"jaldi":  "urgent",
			},
		},
		Vocab: model.ExportedModelVocab{
			BowVocab:       legacy.BowVocab,
			EmbeddingVocab: legacy.EmbeddingVocab,
			Keywords:       []string{"refund", "cancel", "delay", "not working", "money"},
			MaxTokens:      24,
		},
		Labels: model.LabelSet{
			Department: []string{"billing", "tech", "sales", "general"},
			Sentiment:  []string{"positive", "neutral", "negative"},
			LeadIntent: []string{"high", "medium", "low"},
			ChurnRisk:  []string{"low", "high"},
			Intent:     []string{"refund", "delivery_query", "complaint", "other"},
		},
		Embedding: model.ExportedEmbedding{
			PaddingIdx: 0,
			Weights:    legacy.Embeddings.Matrix,
		},
		Layers: model.ExportedLayers{
			Base0: mustQuantizedExportLayer(tb, legacy.Base.Dense1),
			Base1: model.ExportedOp{Type: "relu"},
			Base2: mustQuantizedExportLayer(tb, legacy.Base.Dense2),
			Base3: model.ExportedOp{Type: "relu"},
		},
		Heads: model.ExportedHeads{
			Department: mustQuantizedExportLayer(tb, legacy.Heads.Department),
			Sentiment:  mustQuantizedExportLayer(tb, legacy.Heads.Sentiment),
			LeadIntent: mustQuantizedExportLayer(tb, legacy.Heads.LeadIntent),
			ChurnRisk:  mustQuantizedExportLayer(tb, legacy.Heads.ChurnRisk),
			Intent:     mustQuantizedExportLayer(tb, legacy.Heads.Intent),
		},
	}
}

func mustQuantizedExportLayer(tb testing.TB, layer model.JSONDenseLayer) model.ExportedLinearLayer {
	tb.Helper()

	weightsInt8, scales, err := quantizeMatrix(layer.Weights)
	if err != nil {
		tb.Fatalf("quantize export layer: %v", err)
	}

	return model.ExportedLinearLayer{
		Type:        "linear_int8_per_row",
		InFeatures:  len(layer.Weights[0]),
		OutFeatures: len(layer.Weights),
		Weight:      weightsInt8,
		Scale:       scales,
		Bias:        append([]float32(nil), layer.Bias...),
	}
}
