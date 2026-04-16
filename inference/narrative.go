package inference

import (
	"fmt"
	"strings"
)

func GenerateHumanReadable(result PredictionResult) *HumanReadable {
	if result.Error != "" {
		return nil
	}

	return &HumanReadable{
		Summary:          buildSummary(result),
		TriageNote:       buildTriageNote(result),
		ReplyDraft:       buildReplyDraft(result),
		ManualReviewNote: buildManualReviewNote(result),
	}
}

func buildSummary(result PredictionResult) string {
	var builder strings.Builder
	builder.WriteString("This looks like a ")
	builder.WriteString(humanizeLabel(result.Department.Label))
	builder.WriteString(" ticket, most likely about ")
	builder.WriteString(humanizeLabel(result.Intent.Label))
	builder.WriteString(". ")
	builder.WriteString("The customer sentiment appears ")
	builder.WriteString(humanizeLabel(result.Sentiment.Label))
	builder.WriteString(", lead intent is ")
	builder.WriteString(humanizeLabel(result.LeadIntent.Label))
	builder.WriteString(", and churn risk is ")
	builder.WriteString(humanizeLabel(result.ChurnRisk.Label))
	builder.WriteString(".")

	if len(result.IntentTopK) > 1 {
		secondary := secondaryIntentText(result.IntentTopK)
		if secondary != "" {
			builder.WriteByte(' ')
			builder.WriteString(secondary)
		}
	}

	return builder.String()
}

func buildTriageNote(result PredictionResult) string {
	parts := []string{
		fmt.Sprintf("Route to %s.", humanizeLabel(result.Department.Label)),
		fmt.Sprintf("Primary intent is %s (%s confidence).", humanizeLabel(result.Intent.Label), percent(result.Intent.Confidence)),
		fmt.Sprintf("Sentiment is %s (%s), lead intent is %s (%s), and churn risk is %s (%s).",
			humanizeLabel(result.Sentiment.Label), percent(result.Sentiment.Confidence),
			humanizeLabel(result.LeadIntent.Label), percent(result.LeadIntent.Confidence),
			humanizeLabel(result.ChurnRisk.Label), percent(result.ChurnRisk.Confidence),
		),
	}

	if len(result.IntentTopK) > 1 {
		alternatives := make([]string, 0, len(result.IntentTopK)-1)
		for _, prediction := range result.IntentTopK[1:] {
			if prediction.Confidence < 0.05 {
				continue
			}
			alternatives = append(alternatives, fmt.Sprintf("%s (%s)", humanizeLabel(prediction.Label), percent(prediction.Confidence)))
		}
		if len(alternatives) > 0 {
			parts = append(parts, "Secondary intent candidates are "+joinWithAnd(alternatives)+".")
		}
	}

	if review := buildManualReviewNote(result); review != "" {
		parts = append(parts, review)
	}

	return strings.Join(parts, " ")
}

func buildReplyDraft(result PredictionResult) string {
	opening := "Thanks for reaching out."
	switch result.Sentiment.Label {
	case "negative":
		opening = "Thanks for reaching out, and sorry for the trouble."
	case "positive":
		opening = "Thanks for reaching out, and we appreciate the positive note."
	}

	body := "We've routed your request to the right team and will follow up with next steps shortly."
	switch result.Intent.Label {
	case "pricing_inquiry", "demo_request":
		body = "We've routed this to our sales team, and they can follow up with pricing, plan options, or a demo."
	case "refund", "billing_issue", "cancellation":
		body = "We've routed this to our billing team so they can review the request and follow up with the next steps."
	case "order_tracking", "delivery_issue", "order_change":
		body = "We've routed this to the logistics team so they can check the order status and follow up with an update."
	case "account_access", "technical_issue", "feature_request":
		body = "We've routed this to technical support so they can review the issue and follow up with troubleshooting or next steps."
	case "complaint":
		body = "We've routed this for prompt review so the team can investigate the issue and respond appropriately."
	case "praise":
		body = "We're glad to hear that, and we've shared the note with the appropriate team."
	}

	if result.LeadIntent.Label == "high" && result.Department.Label == "sales" {
		body = "We've routed this to our sales team, and they will follow up with the most relevant pricing, plan, or demo details."
	}

	return opening + " " + body
}

func buildManualReviewNote(result PredictionResult) string {
	needsReview := false

	if result.Intent.Confidence < 0.55 || result.Department.Confidence < 0.55 {
		needsReview = true
	}
	if len(result.IntentTopK) > 1 && result.IntentTopK[0].Confidence-result.IntentTopK[1].Confidence < 0.15 {
		needsReview = true
	}
	if len(result.IntentTopK) > 1 && result.IntentTopK[1].Confidence > 0.20 {
		needsReview = true
	}

	if !needsReview {
		return ""
	}

	return "Model confidence is mixed here, so a quick manual review is recommended before taking irreversible action."
}

func humanizeLabel(label string) string {
	return strings.ReplaceAll(label, "_", " ")
}

func percent(value float32) string {
	return fmt.Sprintf("%.1f%%", value*100)
}

func secondaryIntentText(predictions []RankedPrediction) string {
	if len(predictions) < 2 {
		return ""
	}

	second := predictions[1]
	if second.Confidence < 0.05 {
		return ""
	}

	text := fmt.Sprintf("A secondary possibility is %s (%s).", humanizeLabel(second.Label), percent(second.Confidence))
	if len(predictions) < 3 || predictions[2].Confidence < 0.05 {
		return text
	}

	return fmt.Sprintf("Secondary possibilities are %s (%s) and %s (%s).",
		humanizeLabel(second.Label), percent(second.Confidence),
		humanizeLabel(predictions[2].Label), percent(predictions[2].Confidence),
	)
}

func joinWithAnd(values []string) string {
	switch len(values) {
	case 0:
		return ""
	case 1:
		return values[0]
	case 2:
		return values[0] + " and " + values[1]
	default:
		return strings.Join(values[:len(values)-1], ", ") + ", and " + values[len(values)-1]
	}
}
