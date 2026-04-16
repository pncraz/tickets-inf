package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/octate/tickets-inf/inference"
	"github.com/octate/tickets-inf/model"
)

func ResolveTargets(specs []string, localModelPath string, localConfig inference.Config) ([]Target, model.LabelSet, error) {
	labels := model.LabelSet{}
	var localEngine *inference.Engine

	if localModelPath != "" {
		engine, err := inference.LoadEngineFromFile(localModelPath, localConfig)
		if err != nil {
			return nil, model.LabelSet{}, fmt.Errorf("load local benchmark model: %w", err)
		}
		localEngine = engine
		labels = engine.Labels()
	}

	targets := make([]Target, 0, len(specs))
	for _, spec := range specs {
		spec = strings.TrimSpace(spec)
		if spec == "" {
			continue
		}

		switch {
		case spec == "local":
			if localEngine == nil {
				targets = append(targets, Target{
					Name:       "local",
					Provider:   "local",
					Model:      "local",
					SkipReason: "local target requires -local-model",
				})
				continue
			}
			targets = append(targets, Target{
				Name:      "local",
				Provider:  "local",
				Model:     "local",
				Predictor: &localPredictor{engine: localEngine},
			})
		case strings.HasPrefix(spec, "openai:"):
			if !labelsConfigured(labels) {
				targets = append(targets, skippedRemoteTarget(spec, "external targets require -local-model so label choices match your trained artifact"))
				continue
			}
			targets = append(targets, buildOpenAITarget(strings.TrimPrefix(spec, "openai:"), labels))
		case strings.HasPrefix(spec, "anthropic:"):
			if !labelsConfigured(labels) {
				targets = append(targets, skippedRemoteTarget(spec, "external targets require -local-model so label choices match your trained artifact"))
				continue
			}
			targets = append(targets, buildAnthropicTarget(strings.TrimPrefix(spec, "anthropic:"), labels))
		case strings.HasPrefix(spec, "gemini:"):
			if !labelsConfigured(labels) {
				targets = append(targets, skippedRemoteTarget(spec, "external targets require -local-model so label choices match your trained artifact"))
				continue
			}
			targets = append(targets, buildGeminiTarget(strings.TrimPrefix(spec, "gemini:"), labels))
		default:
			targets = append(targets, Target{
				Name:       spec,
				Provider:   "unknown",
				Model:      spec,
				SkipReason: "unsupported target spec",
			})
		}
	}

	return targets, labels, nil
}

func labelsConfigured(labels model.LabelSet) bool {
	return len(labels.Department) > 0 &&
		len(labels.Sentiment) > 0 &&
		len(labels.LeadIntent) > 0 &&
		len(labels.ChurnRisk) > 0 &&
		len(labels.Intent) > 0
}

func skippedRemoteTarget(spec string, reason string) Target {
	provider := "unknown"
	modelName := spec
	if prefix, rest, ok := strings.Cut(spec, ":"); ok {
		provider = prefix
		modelName = rest
	}
	return Target{
		Name:       spec,
		Provider:   provider,
		Model:      modelName,
		SkipReason: reason,
	}
}

type localPredictor struct {
	engine *inference.Engine
}

func (p *localPredictor) Name() string     { return "local" }
func (p *localPredictor) Provider() string { return "local" }
func (p *localPredictor) Model() string    { return "local" }

func (p *localPredictor) Predict(text string) (InvocationResult, time.Duration, error) {
	start := time.Now()
	result := p.engine.Predict(text)
	duration := time.Since(start)
	if result.Error != "" {
		return InvocationResult{}, duration, errors.New(result.Error)
	}
	return InvocationResult{
		Prediction: Prediction{
			Department: result.Department.Label,
			Sentiment:  result.Sentiment.Label,
			LeadIntent: result.LeadIntent.Label,
			ChurnRisk:  result.ChurnRisk.Label,
			Intent:     result.Intent.Label,
		},
	}, duration, nil
}

type openAIPredictor struct {
	modelName string
	labels    model.LabelSet
	client    *http.Client
	apiKey    string
}

func buildOpenAITarget(modelName string, labels model.LabelSet) Target {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return Target{Name: "openai:" + modelName, Provider: "openai", Model: modelName, SkipReason: "OPENAI_API_KEY is not set"}
	}
	return Target{
		Name:     "openai:" + modelName,
		Provider: "openai",
		Model:    modelName,
		Predictor: &openAIPredictor{
			modelName: modelName,
			labels:    labels,
			client:    &http.Client{Timeout: 60 * time.Second},
			apiKey:    apiKey,
		},
	}
}

func (p *openAIPredictor) Name() string     { return "openai:" + p.modelName }
func (p *openAIPredictor) Provider() string { return "openai" }
func (p *openAIPredictor) Model() string    { return p.modelName }

func (p *openAIPredictor) Predict(text string) (InvocationResult, time.Duration, error) {
	requestBody := buildOpenAIRequestBody(p.modelName, p.labels, text)

	var response struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
		Status            string `json:"status"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details,omitempty"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	duration, err := doJSONRequest(context.Background(), p.client, "https://api.openai.com/v1/responses", map[string]string{
		"Authorization": "Bearer " + p.apiKey,
	}, requestBody, &response)
	if err != nil {
		return InvocationResult{}, duration, err
	}
	if response.Error != nil {
		return InvocationResult{}, duration, fmt.Errorf("openai api error: %s", response.Error.Message)
	}
	if response.Status != "" && response.Status != "completed" && response.IncompleteDetails != nil {
		return InvocationResult{}, duration, fmt.Errorf("openai response status=%s reason=%s", response.Status, response.IncompleteDetails.Reason)
	}

	content := extractOpenAIOutputText(response.Output)
	if strings.TrimSpace(content) == "" {
		return InvocationResult{}, duration, fmt.Errorf("openai response contained no output_text content")
	}

	prediction, err := ParseAndValidatePrediction(content, p.labels)
	if err != nil {
		return InvocationResult{}, duration, err
	}

	return InvocationResult{
		Prediction: prediction,
		Usage: Usage{
			InputTokens:  response.Usage.InputTokens,
			OutputTokens: response.Usage.OutputTokens,
			TotalTokens:  response.Usage.TotalTokens,
		},
		Raw: content,
	}, duration, nil
}

func buildOpenAIRequestBody(modelName string, labels model.LabelSet, text string) map[string]any {
	requestBody := map[string]any{
		"model":             modelName,
		"max_output_tokens": 128,
		"input": []map[string]any{
			{
				"role":    "developer",
				"content": systemInstruction,
			},
			{
				"role":    "user",
				"content": PromptFor(labels, text),
			},
		},
		"text": map[string]any{
			"format": map[string]any{
				"type":   "json_schema",
				"name":   "ticket_classification",
				"strict": true,
				"schema": predictionSchema(labels),
			},
		},
	}
	if openAISupportsTemperature(modelName) {
		requestBody["temperature"] = 0
	}
	if openAIIsReasoningModel(modelName) {
		requestBody["reasoning"] = map[string]any{
			"effort": "minimal",
		}
	}
	return requestBody
}

func openAISupportsTemperature(modelName string) bool {
	return !openAIIsReasoningModel(modelName)
}

func openAIIsReasoningModel(modelName string) bool {
	return strings.HasPrefix(modelName, "gpt-5") || strings.HasPrefix(modelName, "o")
}

type anthropicPredictor struct {
	modelName string
	labels    model.LabelSet
	client    *http.Client
	apiKey    string
}

func buildAnthropicTarget(modelName string, labels model.LabelSet) Target {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return Target{Name: "anthropic:" + modelName, Provider: "anthropic", Model: modelName, SkipReason: "ANTHROPIC_API_KEY is not set"}
	}
	return Target{
		Name:     "anthropic:" + modelName,
		Provider: "anthropic",
		Model:    modelName,
		Predictor: &anthropicPredictor{
			modelName: modelName,
			labels:    labels,
			client:    &http.Client{Timeout: 60 * time.Second},
			apiKey:    apiKey,
		},
	}
}

func (p *anthropicPredictor) Name() string     { return "anthropic:" + p.modelName }
func (p *anthropicPredictor) Provider() string { return "anthropic" }
func (p *anthropicPredictor) Model() string    { return p.modelName }

func (p *anthropicPredictor) Predict(text string) (InvocationResult, time.Duration, error) {
	requestBody := map[string]any{
		"model":       p.modelName,
		"max_tokens":  256,
		"temperature": 0,
		"system":      systemInstruction,
		"messages": []map[string]string{
			{"role": "user", "content": PromptFor(p.labels, text)},
		},
	}

	var response struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	duration, err := doJSONRequest(context.Background(), p.client, "https://api.anthropic.com/v1/messages", map[string]string{
		"x-api-key":         p.apiKey,
		"anthropic-version": "2023-06-01",
	}, requestBody, &response)
	if err != nil {
		return InvocationResult{}, duration, err
	}
	if response.Error != nil {
		return InvocationResult{}, duration, fmt.Errorf("anthropic api error: %s", response.Error.Message)
	}

	var content string
	for _, block := range response.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}
	if strings.TrimSpace(content) == "" {
		return InvocationResult{}, duration, fmt.Errorf("anthropic response contained no text content")
	}

	prediction, err := ParseAndValidatePrediction(content, p.labels)
	if err != nil {
		return InvocationResult{}, duration, err
	}

	return InvocationResult{
		Prediction: prediction,
		Usage: Usage{
			InputTokens:  response.Usage.InputTokens,
			OutputTokens: response.Usage.OutputTokens,
			TotalTokens:  response.Usage.InputTokens + response.Usage.OutputTokens,
		},
		Raw: content,
	}, duration, nil
}

type geminiPredictor struct {
	modelName string
	labels    model.LabelSet
	client    *http.Client
	apiKey    string
}

func buildGeminiTarget(modelName string, labels model.LabelSet) Target {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return Target{Name: "gemini:" + modelName, Provider: "gemini", Model: modelName, SkipReason: "GOOGLE_API_KEY or GEMINI_API_KEY is not set"}
	}
	return Target{
		Name:     "gemini:" + modelName,
		Provider: "gemini",
		Model:    modelName,
		Predictor: &geminiPredictor{
			modelName: modelName,
			labels:    labels,
			client:    &http.Client{Timeout: 60 * time.Second},
			apiKey:    apiKey,
		},
	}
}

func (p *geminiPredictor) Name() string     { return "gemini:" + p.modelName }
func (p *geminiPredictor) Provider() string { return "gemini" }
func (p *geminiPredictor) Model() string    { return p.modelName }

func (p *geminiPredictor) Predict(text string) (InvocationResult, time.Duration, error) {
	requestBody := map[string]any{
		"contents": []map[string]any{
			{
				"role": "user",
				"parts": []map[string]string{
					{"text": systemInstruction + "\n\n" + PromptFor(p.labels, text)},
				},
			},
		},
		"generationConfig": map[string]any{
			"temperature":      0,
			"responseMimeType": "application/json",
		},
	}

	var response struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	endpoint := "https://generativelanguage.googleapis.com/v1beta/models/" + p.modelName + ":generateContent?key=" + p.apiKey
	duration, err := doJSONRequest(context.Background(), p.client, endpoint, nil, requestBody, &response)
	if err != nil {
		return InvocationResult{}, duration, err
	}
	if response.Error != nil {
		return InvocationResult{}, duration, fmt.Errorf("gemini api error: %s", response.Error.Message)
	}
	if len(response.Candidates) == 0 || len(response.Candidates[0].Content.Parts) == 0 {
		return InvocationResult{}, duration, fmt.Errorf("gemini response contained no candidates")
	}

	content := response.Candidates[0].Content.Parts[0].Text
	prediction, err := ParseAndValidatePrediction(content, p.labels)
	if err != nil {
		return InvocationResult{}, duration, err
	}

	return InvocationResult{
		Prediction: prediction,
		Usage: Usage{
			InputTokens:  response.UsageMetadata.PromptTokenCount,
			OutputTokens: response.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  response.UsageMetadata.TotalTokenCount,
		},
		Raw: content,
	}, duration, nil
}

func doJSONRequest(ctx context.Context, client *http.Client, endpoint string, headers map[string]string, requestBody any, responseBody any) (time.Duration, error) {
	payload, err := json.Marshal(requestBody)
	if err != nil {
		return 0, fmt.Errorf("marshal request: %w", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return 0, fmt.Errorf("build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		request.Header.Set(key, value)
	}

	start := time.Now()
	response, err := client.Do(request)
	duration := time.Since(start)
	if err != nil {
		return duration, fmt.Errorf("perform request: %w", err)
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		return duration, fmt.Errorf("read response: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return duration, fmt.Errorf("unexpected http %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	if err := json.Unmarshal(body, responseBody); err != nil {
		return duration, fmt.Errorf("decode response: %w", err)
	}

	return duration, nil
}

func predictionSchema(labels model.LabelSet) map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []string{"department", "sentiment", "lead_intent", "churn_risk", "intent"},
		"properties": map[string]any{
			"department": map[string]any{
				"type": "string",
				"enum": labels.Department,
			},
			"sentiment": map[string]any{
				"type": "string",
				"enum": labels.Sentiment,
			},
			"lead_intent": map[string]any{
				"type": "string",
				"enum": labels.LeadIntent,
			},
			"churn_risk": map[string]any{
				"type": "string",
				"enum": labels.ChurnRisk,
			},
			"intent": map[string]any{
				"type": "string",
				"enum": labels.Intent,
			},
		},
	}
}

func extractOpenAIOutputText(output []struct {
	Type    string `json:"type"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}) string {
	var builder strings.Builder
	for _, item := range output {
		if item.Type != "message" {
			continue
		}
		for _, content := range item.Content {
			if content.Type == "output_text" {
				builder.WriteString(content.Text)
			}
		}
	}
	return builder.String()
}
