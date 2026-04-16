package benchmark

import (
	"time"

	"github.com/octate/tickets-inf/model"
)

type Example struct {
	ID         string `json:"id"`
	Text       string `json:"text"`
	Department string `json:"department"`
	Sentiment  string `json:"sentiment"`
	LeadIntent string `json:"lead_intent"`
	ChurnRisk  string `json:"churn_risk"`
	Intent     string `json:"intent"`
}

type Prediction struct {
	Department string `json:"department"`
	Sentiment  string `json:"sentiment"`
	LeadIntent string `json:"lead_intent"`
	ChurnRisk  string `json:"churn_risk"`
	Intent     string `json:"intent"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

type InvocationResult struct {
	Prediction Prediction `json:"prediction"`
	Usage      Usage      `json:"usage,omitempty"`
	Raw        string     `json:"raw,omitempty"`
}

type CaseResult struct {
	ID             string          `json:"id"`
	Text           string          `json:"text"`
	Expected       Prediction      `json:"expected"`
	Predicted      Prediction      `json:"predicted,omitempty"`
	Correct        map[string]bool `json:"correct,omitempty"`
	DurationMillis float64         `json:"duration_ms"`
	Usage          Usage           `json:"usage,omitempty"`
	Error          string          `json:"error,omitempty"`
}

type ModelReport struct {
	Name                 string             `json:"name"`
	Provider             string             `json:"provider"`
	Model                string             `json:"model"`
	Skipped              bool               `json:"skipped,omitempty"`
	SkipReason           string             `json:"skip_reason,omitempty"`
	FirstError           string             `json:"first_error,omitempty"`
	TotalCases           int                `json:"total_cases"`
	CompletedCases       int                `json:"completed_cases"`
	ErrorCount           int                `json:"error_count"`
	ExactMatchAccuracy   float64            `json:"exact_match_accuracy"`
	TaskAccuracies       map[string]float64 `json:"task_accuracies"`
	AverageLatencyMillis float64            `json:"avg_latency_ms"`
	P50LatencyMillis     float64            `json:"p50_latency_ms"`
	P95LatencyMillis     float64            `json:"p95_latency_ms"`
	Usage                Usage              `json:"usage,omitempty"`
	CaseResults          []CaseResult       `json:"case_results,omitempty"`
}

type Report struct {
	GeneratedAt string         `json:"generated_at"`
	DatasetPath string         `json:"dataset_path"`
	CaseCount   int            `json:"case_count"`
	Labels      model.LabelSet `json:"labels"`
	Results     []ModelReport  `json:"results"`
}

type Predictor interface {
	Name() string
	Provider() string
	Model() string
	Predict(text string) (InvocationResult, time.Duration, error)
}

type Target struct {
	Name       string
	Provider   string
	Model      string
	Predictor  Predictor
	SkipReason string
}
