package benchmark

import (
	"fmt"
	"strings"

	"github.com/octate/tickets-inf/model"
)

const systemInstruction = "You are a strict support-ticket classification engine. Choose exactly one label for each field and return only a single JSON object."

func PromptFor(labels model.LabelSet, text string) string {
	return fmt.Sprintf(`Classify the support ticket using exactly one label from each field.

department: [%s]
sentiment: [%s]
lead_intent: [%s]
churn_risk: [%s]
intent: [%s]

Return only JSON in this exact shape:
{"department":"...","sentiment":"...","lead_intent":"...","churn_risk":"...","intent":"..."}

Support ticket:
%s`, strings.Join(labels.Department, ", "), strings.Join(labels.Sentiment, ", "), strings.Join(labels.LeadIntent, ", "), strings.Join(labels.ChurnRisk, ", "), strings.Join(labels.Intent, ", "), text)
}
