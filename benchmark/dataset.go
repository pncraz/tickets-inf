package benchmark

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func LoadJSONL(path string) ([]Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open benchmark dataset: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buffer := make([]byte, 0, 1024*1024)
	scanner.Buffer(buffer, 1024*1024)

	examples := make([]Example, 0)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var example Example
		if err := json.Unmarshal([]byte(line), &example); err != nil {
			return nil, fmt.Errorf("decode benchmark line %d: %w", lineNumber, err)
		}
		if err := validateExample(example); err != nil {
			return nil, fmt.Errorf("invalid benchmark line %d: %w", lineNumber, err)
		}
		examples = append(examples, example)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan benchmark dataset: %w", err)
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("benchmark dataset is empty")
	}

	return examples, nil
}

func validateExample(example Example) error {
	switch {
	case strings.TrimSpace(example.ID) == "":
		return fmt.Errorf("id is required")
	case strings.TrimSpace(example.Text) == "":
		return fmt.Errorf("text is required")
	case strings.TrimSpace(example.Department) == "":
		return fmt.Errorf("department is required")
	case strings.TrimSpace(example.Sentiment) == "":
		return fmt.Errorf("sentiment is required")
	case strings.TrimSpace(example.LeadIntent) == "":
		return fmt.Errorf("lead_intent is required")
	case strings.TrimSpace(example.ChurnRisk) == "":
		return fmt.Errorf("churn_risk is required")
	case strings.TrimSpace(example.Intent) == "":
		return fmt.Errorf("intent is required")
	default:
		return nil
	}
}

func (e Example) ExpectedPrediction() Prediction {
	return Prediction{
		Department: e.Department,
		Sentiment:  e.Sentiment,
		LeadIntent: e.LeadIntent,
		ChurnRisk:  e.ChurnRisk,
		Intent:     e.Intent,
	}
}
