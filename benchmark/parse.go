package benchmark

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pncraz/tickets-inf/model"
)

type rawPrediction struct {
	Department string `json:"department"`
	Sentiment  string `json:"sentiment"`
	LeadIntent string `json:"lead_intent"`
	ChurnRisk  string `json:"churn_risk"`
	Intent     string `json:"intent"`
}

func ParseAndValidatePrediction(payload string, labels model.LabelSet) (Prediction, error) {
	extracted, err := extractJSONObject(strings.TrimSpace(payload))
	if err != nil {
		return Prediction{}, err
	}

	var raw rawPrediction
	if err := json.Unmarshal([]byte(extracted), &raw); err != nil {
		return Prediction{}, fmt.Errorf("decode prediction json: %w", err)
	}

	prediction := Prediction{
		Department: normalizeLabel(raw.Department),
		Sentiment:  normalizeLabel(raw.Sentiment),
		LeadIntent: normalizeLabel(raw.LeadIntent),
		ChurnRisk:  normalizeLabel(raw.ChurnRisk),
		Intent:     normalizeLabel(raw.Intent),
	}

	if !containsLabel(labels.Department, prediction.Department) {
		return Prediction{}, fmt.Errorf("invalid department label: %q", raw.Department)
	}
	if !containsLabel(labels.Sentiment, prediction.Sentiment) {
		return Prediction{}, fmt.Errorf("invalid sentiment label: %q", raw.Sentiment)
	}
	if !containsLabel(labels.LeadIntent, prediction.LeadIntent) {
		return Prediction{}, fmt.Errorf("invalid lead_intent label: %q", raw.LeadIntent)
	}
	if !containsLabel(labels.ChurnRisk, prediction.ChurnRisk) {
		return Prediction{}, fmt.Errorf("invalid churn_risk label: %q", raw.ChurnRisk)
	}
	if !containsLabel(labels.Intent, prediction.Intent) {
		return Prediction{}, fmt.Errorf("invalid intent label: %q", raw.Intent)
	}

	return prediction, nil
}

func extractJSONObject(payload string) (string, error) {
	start := strings.IndexByte(payload, '{')
	if start == -1 {
		return "", fmt.Errorf("no json object found in response")
	}

	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(payload); i++ {
		current := payload[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			switch current {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}

		switch current {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return payload[start : i+1], nil
			}
		}
	}

	return "", fmt.Errorf("unterminated json object in response")
}

func normalizeLabel(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	replacer := strings.NewReplacer(" ", "_", "-", "_", "/", "_")
	return replacer.Replace(value)
}

func containsLabel(labels []string, value string) bool {
	for _, label := range labels {
		if normalizeLabel(label) == value {
			return true
		}
	}
	return false
}
