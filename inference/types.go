package inference

type RankedPrediction struct {
	Label      string  `json:"label"`
	Confidence float32 `json:"confidence"`
}

type HeadPrediction struct {
	Label      string             `json:"label"`
	Confidence float32            `json:"confidence"`
	Scores     map[string]float32 `json:"scores,omitempty"`
}

type DebugInfo struct {
	NormalizedText string    `json:"normalized_text"`
	Tokens         []string  `json:"tokens"`
	FeatureVector  []float32 `json:"feature_vector"`
}

type HumanReadable struct {
	Summary          string `json:"summary"`
	TriageNote       string `json:"triage_note"`
	ReplyDraft       string `json:"reply_draft"`
	ManualReviewNote string `json:"manual_review_note,omitempty"`
}

type PredictionResult struct {
	Department    HeadPrediction     `json:"department"`
	Sentiment     HeadPrediction     `json:"sentiment"`
	LeadIntent    HeadPrediction     `json:"lead_intent"`
	ChurnRisk     HeadPrediction     `json:"churn_risk"`
	Intent        HeadPrediction     `json:"intent"`
	IntentTopK    []RankedPrediction `json:"intent_top_k,omitempty"`
	HumanReadable *HumanReadable     `json:"human_readable,omitempty"`
	Debug         *DebugInfo         `json:"debug,omitempty"`
	Error         string             `json:"error,omitempty"`
}
