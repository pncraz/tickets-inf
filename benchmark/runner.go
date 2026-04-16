package benchmark

import (
	"fmt"
	"sort"
	"time"

	"github.com/octate/tickets-inf/model"
)

func Run(datasetPath string, labels model.LabelSet, examples []Example, targets []Target, includeCases bool) Report {
	report := Report{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		DatasetPath: datasetPath,
		CaseCount:   len(examples),
		Labels:      labels,
		Results:     make([]ModelReport, 0, len(targets)),
	}

	for _, target := range targets {
		if target.Predictor == nil {
			report.Results = append(report.Results, ModelReport{
				Name:       target.Name,
				Provider:   target.Provider,
				Model:      target.Model,
				Skipped:    true,
				SkipReason: target.SkipReason,
				TotalCases: len(examples),
				TaskAccuracies: map[string]float64{
					"department":  0,
					"sentiment":   0,
					"lead_intent": 0,
					"churn_risk":  0,
					"intent":      0,
				},
			})
			continue
		}

		report.Results = append(report.Results, runTarget(examples, target.Predictor, includeCases))
	}

	return report
}

func runTarget(examples []Example, predictor Predictor, includeCases bool) ModelReport {
	report := ModelReport{
		Name:       predictor.Name(),
		Provider:   predictor.Provider(),
		Model:      predictor.Model(),
		TotalCases: len(examples),
		TaskAccuracies: map[string]float64{
			"department":  0,
			"sentiment":   0,
			"lead_intent": 0,
			"churn_risk":  0,
			"intent":      0,
		},
	}

	if includeCases {
		report.CaseResults = make([]CaseResult, 0, len(examples))
	}

	correct := map[string]int{
		"department":  0,
		"sentiment":   0,
		"lead_intent": 0,
		"churn_risk":  0,
		"intent":      0,
	}
	exactMatches := 0
	latencies := make([]float64, 0, len(examples))

	for _, example := range examples {
		expected := example.ExpectedPrediction()
		invocation, duration, err := predictor.Predict(example.Text)
		caseResult := CaseResult{
			ID:             example.ID,
			Text:           example.Text,
			Expected:       expected,
			DurationMillis: duration.Seconds() * 1000,
		}
		latencies = append(latencies, caseResult.DurationMillis)

		if err != nil {
			report.ErrorCount++
			if report.FirstError == "" {
				report.FirstError = err.Error()
			}
			caseResult.Error = err.Error()
			if includeCases {
				report.CaseResults = append(report.CaseResults, caseResult)
			}
			continue
		}

		report.CompletedCases++
		report.Usage.InputTokens += invocation.Usage.InputTokens
		report.Usage.OutputTokens += invocation.Usage.OutputTokens
		report.Usage.TotalTokens += invocation.Usage.TotalTokens
		caseResult.Predicted = invocation.Prediction
		caseResult.Usage = invocation.Usage
		caseResult.Correct = map[string]bool{
			"department":  invocation.Prediction.Department == expected.Department,
			"sentiment":   invocation.Prediction.Sentiment == expected.Sentiment,
			"lead_intent": invocation.Prediction.LeadIntent == expected.LeadIntent,
			"churn_risk":  invocation.Prediction.ChurnRisk == expected.ChurnRisk,
			"intent":      invocation.Prediction.Intent == expected.Intent,
		}

		allCorrect := true
		for task, isCorrect := range caseResult.Correct {
			if isCorrect {
				correct[task]++
			} else {
				allCorrect = false
			}
		}
		if allCorrect {
			exactMatches++
		}
		if includeCases {
			report.CaseResults = append(report.CaseResults, caseResult)
		}
	}

	if len(latencies) > 0 {
		sort.Float64s(latencies)
		report.AverageLatencyMillis = average(latencies)
		report.P50LatencyMillis = percentile(latencies, 0.50)
		report.P95LatencyMillis = percentile(latencies, 0.95)
	}

	if report.CompletedCases > 0 {
		report.ExactMatchAccuracy = float64(exactMatches) / float64(report.CompletedCases)
		for task, count := range correct {
			report.TaskAccuracies[task] = float64(count) / float64(report.CompletedCases)
		}
	}

	return report
}

func PrintSummary(report Report) string {
	var lines []string
	lines = append(lines, fmt.Sprintf("Benchmark dataset: %s (%d cases)", report.DatasetPath, report.CaseCount))
	for _, result := range report.Results {
		if result.Skipped {
			lines = append(lines, fmt.Sprintf("- %s: skipped (%s)", result.Name, result.SkipReason))
			continue
		}
		line := fmt.Sprintf("- %s: exact_match=%.1f%% intent=%.1f%% department=%.1f%% sentiment=%.1f%% lead_intent=%.1f%% churn_risk=%.1f%% avg=%.2fms p95=%.2fms errors=%d",
			result.Name,
			result.ExactMatchAccuracy*100,
			result.TaskAccuracies["intent"]*100,
			result.TaskAccuracies["department"]*100,
			result.TaskAccuracies["sentiment"]*100,
			result.TaskAccuracies["lead_intent"]*100,
			result.TaskAccuracies["churn_risk"]*100,
			result.AverageLatencyMillis,
			result.P95LatencyMillis,
			result.ErrorCount,
		)
		if result.FirstError != "" {
			line += fmt.Sprintf(" first_error=%q", truncateSummary(result.FirstError, 120))
		}
		lines = append(lines, line)
	}
	return stringsJoin(lines, "\n")
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func percentile(sortedValues []float64, fraction float64) float64 {
	if len(sortedValues) == 0 {
		return 0
	}
	index := int(float64(len(sortedValues)-1) * fraction)
	return sortedValues[index]
}

func stringsJoin(values []string, separator string) string {
	if len(values) == 0 {
		return ""
	}
	result := values[0]
	for _, value := range values[1:] {
		result += separator + value
	}
	return result
}

func truncateSummary(value string, maxLen int) string {
	if len(value) <= maxLen {
		return value
	}
	if maxLen <= 3 {
		return value[:maxLen]
	}
	return value[:maxLen-3] + "..."
}
