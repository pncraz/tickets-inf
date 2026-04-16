package features

import (
	"math"
	"testing"
)

func TestExtractorBuildsExpectedHybridFeatures(t *testing.T) {
	extractor := NewExtractor(
		map[string]int{"refund": 0, "delivery": 1, "delay": 2},
		map[string]int{"refund": 0, "delivery": 1, "delay": 2},
	)

	parts := extractor.Extract("Refund!! My delivery is DELAYED and not working.")

	if parts.NormalizedText != "refund my delivery is delayed and not working" {
		t.Fatalf("unexpected normalized text: %q", parts.NormalizedText)
	}

	if len(parts.Tokens) != 8 {
		t.Fatalf("unexpected token count: %d", len(parts.Tokens))
	}

	if parts.BagOfWords[0] != 1 || parts.BagOfWords[1] != 1 {
		t.Fatalf("unexpected bag-of-words vector: %v", parts.BagOfWords)
	}

	if parts.KeywordFlags[0] != 1 || parts.KeywordFlags[2] != 1 || parts.KeywordFlags[3] != 1 {
		t.Fatalf("unexpected keyword flags: %v", parts.KeywordFlags)
	}

	if len(parts.EmbeddingTokenIDs) != 2 {
		t.Fatalf("unexpected embedding ids: %v", parts.EmbeddingTokenIDs)
	}

	finalVector := parts.FinalVector([]float32{0.25, 0.75})
	expectedLength := extractor.BowSize() + len(KeywordFeatureNames) + 2
	if len(finalVector) != expectedLength {
		t.Fatalf("unexpected final feature vector length: got=%d want=%d", len(finalVector), expectedLength)
	}
}

func TestConfiguredExtractorMatchesTrainingStyleEncoding(t *testing.T) {
	extractor := NewConfiguredExtractor(Config{
		BowVocab: map[string]int{
			"refund": 0,
			"money":  1,
			"urgent": 2,
			"<num>":  3,
		},
		EmbeddingVocab: map[string]int{
			"<pad>":  0,
			"<unk>":  1,
			"refund": 2,
			"money":  3,
			"urgent": 4,
			"<num>":  5,
		},
		KeywordPhrases:  []string{"refund", "not working", "money"},
		MaxTokens:       3,
		UseLog1pBow:     true,
		UnknownTokenID:  1,
		HasUnknownToken: true,
		PreprocessConfig: PreprocessConfig{
			Lowercase:      true,
			ReplaceNumbers: "<num>",
			HinglishMap: map[string]string{
				"paisa": "money",
				"jaldi": "urgent",
			},
		},
	})

	parts := extractor.Extract("Refund paisa mystery jaldi 123")

	if parts.NormalizedText != "refund money mystery urgent <num>" {
		t.Fatalf("unexpected normalized text: %q", parts.NormalizedText)
	}

	if len(parts.Tokens) != 5 {
		t.Fatalf("unexpected token count: %d", len(parts.Tokens))
	}

	if !almostEqual(parts.BagOfWords[0], float32(math.Log1p(1))) {
		t.Fatalf("unexpected log1p bow value for refund: %f", parts.BagOfWords[0])
	}
	if !almostEqual(parts.BagOfWords[1], float32(math.Log1p(1))) {
		t.Fatalf("unexpected log1p bow value for money: %f", parts.BagOfWords[1])
	}
	if !almostEqual(parts.BagOfWords[2], float32(math.Log1p(1))) {
		t.Fatalf("unexpected log1p bow value for urgent: %f", parts.BagOfWords[2])
	}
	if !almostEqual(parts.BagOfWords[3], float32(math.Log1p(1))) {
		t.Fatalf("unexpected log1p bow value for <num>: %f", parts.BagOfWords[3])
	}

	if parts.KeywordFlags[0] != 1 || parts.KeywordFlags[2] != 1 || parts.KeywordFlags[1] != 0 {
		t.Fatalf("unexpected keyword flags: %v", parts.KeywordFlags)
	}

	expectedIDs := []int{2, 3, 1}
	if len(parts.EmbeddingTokenIDs) != len(expectedIDs) {
		t.Fatalf("unexpected embedding ids length: got=%d want=%d", len(parts.EmbeddingTokenIDs), len(expectedIDs))
	}
	for i, want := range expectedIDs {
		if parts.EmbeddingTokenIDs[i] != want {
			t.Fatalf("unexpected embedding id at %d: got=%d want=%d", i, parts.EmbeddingTokenIDs[i], want)
		}
	}
}

func almostEqual(left, right float32) bool {
	diff := math.Abs(float64(left - right))
	return diff < 1e-5
}
