package benchmark

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/pncraz/tickets-inf/inference"
	"github.com/pncraz/tickets-inf/model"
)

func TestLoadJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cases.jsonl")
	content := `{"id":"case-1","text":"Need a refund","department":"billing","sentiment":"negative","lead_intent":"low","churn_risk":"high","intent":"refund"}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}

	examples, err := LoadJSONL(path)
	if err != nil {
		t.Fatalf("load jsonl: %v", err)
	}
	if len(examples) != 1 || examples[0].ID != "case-1" {
		t.Fatalf("unexpected examples: %+v", examples)
	}
}

func TestRunLocalBenchmark(t *testing.T) {
	m, err := model.NewDemoModel()
	if err != nil {
		t.Fatalf("build demo model: %v", err)
	}
	engine, err := inference.NewEngine(m, inference.Config{UseQuantized: true})
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}

	predictor := &localPredictor{engine: engine}
	report := Run("memory", m.Labels, []Example{
		{
			ID:         "case-1",
			Text:       "Please refund the money and cancel this invoice",
			Department: "billing",
			Sentiment:  "negative",
			LeadIntent: "low",
			ChurnRisk:  "high",
			Intent:     "refund",
		},
	}, []Target{{Name: "local", Provider: "local", Model: "local", Predictor: predictor}}, true)

	if len(report.Results) != 1 {
		t.Fatalf("unexpected report results: %d", len(report.Results))
	}
	if report.Results[0].ExactMatchAccuracy != 1 {
		t.Fatalf("unexpected exact match accuracy: %f", report.Results[0].ExactMatchAccuracy)
	}
}

func TestParseAndValidatePrediction(t *testing.T) {
	labels := model.LabelSet{
		Department: []string{"billing", "sales"},
		Sentiment:  []string{"negative", "neutral"},
		LeadIntent: []string{"high", "low"},
		ChurnRisk:  []string{"low", "high"},
		Intent:     []string{"refund", "pricing_inquiry"},
	}

	prediction, err := ParseAndValidatePrediction("```json\n{\"department\":\"billing\",\"sentiment\":\"negative\",\"lead_intent\":\"low\",\"churn_risk\":\"high\",\"intent\":\"refund\"}\n```", labels)
	if err != nil {
		t.Fatalf("parse prediction: %v", err)
	}

	payload, _ := json.Marshal(prediction)
	if string(payload) == "" {
		t.Fatalf("expected marshaled prediction to be non-empty")
	}
}

func TestResolveTargetsRequiresLocalModelForRemoteTargets(t *testing.T) {
	targets, labels, err := ResolveTargets([]string{"openai:gpt-5-mini"}, "", inference.Config{})
	if err != nil {
		t.Fatalf("resolve targets: %v", err)
	}
	if len(targets) != 1 {
		t.Fatalf("unexpected target count: %d", len(targets))
	}
	if targets[0].Predictor != nil {
		t.Fatalf("expected remote target without local labels to be skipped")
	}
	if len(labels.Department) != 0 || len(labels.Sentiment) != 0 || len(labels.LeadIntent) != 0 || len(labels.ChurnRisk) != 0 || len(labels.Intent) != 0 {
		t.Fatalf("expected empty labels when no local model is supplied: %+v", labels)
	}
}

func TestBuildOpenAIRequestBodyForGPT5OmitsTemperature(t *testing.T) {
	labels := model.LabelSet{
		Department: []string{"billing", "sales"},
		Sentiment:  []string{"negative", "neutral"},
		LeadIntent: []string{"high", "low"},
		ChurnRisk:  []string{"low", "high"},
		Intent:     []string{"refund", "pricing_inquiry"},
	}

	body := buildOpenAIRequestBody("gpt-5-mini", labels, "Need pricing")
	if _, ok := body["temperature"]; ok {
		t.Fatalf("expected gpt-5 request body to omit temperature")
	}
	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "minimal" {
		t.Fatalf("expected gpt-5 request body to set minimal reasoning: %+v", body["reasoning"])
	}
}

func TestBuildOpenAIRequestBodyForGPT41KeepsTemperature(t *testing.T) {
	labels := model.LabelSet{
		Department: []string{"billing", "sales"},
		Sentiment:  []string{"negative", "neutral"},
		LeadIntent: []string{"high", "low"},
		ChurnRisk:  []string{"low", "high"},
		Intent:     []string{"refund", "pricing_inquiry"},
	}

	body := buildOpenAIRequestBody("gpt-4.1-mini", labels, "Need pricing")
	if value, ok := body["temperature"]; !ok || value != 0 {
		t.Fatalf("expected gpt-4.1 request body to keep temperature=0: %+v", body["temperature"])
	}
	if _, ok := body["reasoning"]; ok {
		t.Fatalf("did not expect reasoning config for gpt-4.1")
	}
}
