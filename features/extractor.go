package features

import (
	"math"
	"strings"
)

type FeatureParts struct {
	NormalizedText    string
	Tokens            []string
	BagOfWords        []float32
	KeywordFlags      []float32
	EmbeddingTokenIDs []int
}

func (p FeatureParts) FinalVector(embeddingVector []float32) []float32 {
	return Concat(p.BagOfWords, p.KeywordFlags, embeddingVector)
}

type Config struct {
	BowVocab              map[string]int
	EmbeddingVocab        map[string]int
	KeywordPhrases        []string
	UseLegacyKeywordFlags bool
	MaxTokens             int
	UseLog1pBow           bool
	UnknownTokenID        int
	HasUnknownToken       bool
	PreprocessConfig      PreprocessConfig
}

type Extractor struct {
	normalizer            *Normalizer
	bowVocab              map[string]int
	embeddingVocab        map[string]int
	keywordPhrases        []string
	useLegacyKeywordFlags bool
	bowSize               int
	maxTokens             int
	useLog1pBow           bool
	unknownTokenID        int
	hasUnknownToken       bool
}

func NewExtractor(bowVocab, embeddingVocab map[string]int) *Extractor {
	return NewConfiguredExtractor(Config{
		BowVocab:              bowVocab,
		EmbeddingVocab:        embeddingVocab,
		UseLegacyKeywordFlags: true,
		PreprocessConfig:      LegacyPreprocessConfig(),
	})
}

func NewConfiguredExtractor(config Config) *Extractor {
	return &Extractor{
		normalizer:            NewNormalizer(config.PreprocessConfig),
		bowVocab:              cloneVocab(config.BowVocab),
		embeddingVocab:        cloneVocab(config.EmbeddingVocab),
		keywordPhrases:        append([]string(nil), config.KeywordPhrases...),
		useLegacyKeywordFlags: config.UseLegacyKeywordFlags,
		bowSize:               maxIndex(config.BowVocab) + 1,
		maxTokens:             config.MaxTokens,
		useLog1pBow:           config.UseLog1pBow,
		unknownTokenID:        config.UnknownTokenID,
		hasUnknownToken:       config.HasUnknownToken,
	}
}

func (e *Extractor) BowSize() int {
	return e.bowSize
}

func (e *Extractor) KeywordSize() int {
	if e.useLegacyKeywordFlags {
		return len(KeywordFeatureNames)
	}
	return len(e.keywordPhrases)
}

func (e *Extractor) Extract(text string) FeatureParts {
	preprocessed := e.normalizer.Preprocess(text)
	return FeatureParts{
		NormalizedText:    preprocessed.Normalized,
		Tokens:            preprocessed.Tokens,
		BagOfWords:        e.BagOfWords(preprocessed.Tokens),
		KeywordFlags:      e.KeywordFlags(preprocessed.Tokens, preprocessed.Normalized),
		EmbeddingTokenIDs: e.EmbeddingTokenIDs(preprocessed.Tokens),
	}
}

func (e *Extractor) BagOfWords(tokens []string) []float32 {
	vector := make([]float32, e.bowSize)
	for _, token := range tokens {
		if index, ok := e.bowVocab[token]; ok && index >= 0 && index < e.bowSize {
			vector[index]++
		}
	}

	if e.useLog1pBow {
		for i, value := range vector {
			if value > 0 {
				vector[i] = float32(math.Log1p(float64(value)))
			}
		}
	}
	return vector
}

func (e *Extractor) KeywordFlags(tokens []string, normalized string) []float32 {
	if e.useLegacyKeywordFlags {
		return e.legacyKeywordFlags(tokens, normalized)
	}

	flags := make([]float32, len(e.keywordPhrases))
	for i, keyword := range e.keywordPhrases {
		if keyword != "" && strings.Contains(normalized, keyword) {
			flags[i] = 1
		}
	}
	return flags
}

func (e *Extractor) legacyKeywordFlags(tokens []string, normalized string) []float32 {
	flags := make([]float32, len(KeywordFeatureNames))
	tokenSet := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		tokenSet[token] = struct{}{}
	}

	if containsAny(tokenSet, "refund", "refunded", "refunds", "chargeback", "reimbursement", "moneyback") {
		flags[0] = 1
	}
	if containsAny(tokenSet, "cancel", "cancelled", "canceled", "cancellation", "terminate", "close") {
		flags[1] = 1
	}
	if containsAny(tokenSet, "delay", "delayed", "late", "overdue", "waiting", "shipment", "shipping") {
		flags[2] = 1
	}
	if strings.Contains(normalized, "not working") || containsAny(tokenSet, "broken", "error", "errors", "bug", "issue", "failed", "failure", "down") {
		flags[3] = 1
	}
	if containsAny(tokenSet, "money", "price", "pricing", "charged", "charge", "bill", "billing", "payment", "invoice", "cost") {
		flags[4] = 1
	}

	return flags
}

func (e *Extractor) EmbeddingTokenIDs(tokens []string) []int {
	limit := len(tokens)
	if e.maxTokens > 0 && limit > e.maxTokens {
		limit = e.maxTokens
	}

	ids := make([]int, 0, limit)
	for _, token := range tokens[:limit] {
		if index, ok := e.embeddingVocab[token]; ok && index >= 0 {
			ids = append(ids, index)
			continue
		}
		if e.hasUnknownToken {
			ids = append(ids, e.unknownTokenID)
		}
	}
	return ids
}

func Concat(vectors ...[]float32) []float32 {
	total := 0
	for _, vector := range vectors {
		total += len(vector)
	}

	result := make([]float32, total)
	offset := 0
	for _, vector := range vectors {
		copy(result[offset:], vector)
		offset += len(vector)
	}
	return result
}

func containsAny(tokenSet map[string]struct{}, values ...string) bool {
	for _, value := range values {
		if _, ok := tokenSet[value]; ok {
			return true
		}
	}
	return false
}

func cloneVocab(source map[string]int) map[string]int {
	if len(source) == 0 {
		return map[string]int{}
	}

	cloned := make(map[string]int, len(source))
	for token, index := range source {
		cloned[token] = index
	}
	return cloned
}

func maxIndex(vocab map[string]int) int {
	max := -1
	for _, index := range vocab {
		if index > max {
			max = index
		}
	}
	return max
}
