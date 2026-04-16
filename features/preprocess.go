package features

import (
	"regexp"
	"sort"
	"strings"
)

type Preprocessed struct {
	Normalized string
	Tokens     []string
}

type PreprocessConfig struct {
	Lowercase      bool              `json:"lowercase"`
	ReplaceURLs    string            `json:"replace_urls,omitempty"`
	ReplaceEmails  string            `json:"replace_emails,omitempty"`
	ReplaceNumbers string            `json:"replace_numbers,omitempty"`
	HinglishMap    map[string]string `json:"hinglish_map,omitempty"`
}

type replacementRule struct {
	pattern     *regexp.Regexp
	replacement string
}

type Normalizer struct {
	config        PreprocessConfig
	hinglishRules []replacementRule
}

var (
	urlRE      = regexp.MustCompile(`https?://\S+|www\.\S+`)
	emailRE    = regexp.MustCompile(`\b[\w.+-]+@[\w-]+\.[\w.-]+\b`)
	numberRE   = regexp.MustCompile(`\b\d+\b`)
	nonTokenRE = regexp.MustCompile(`[^A-Za-z0-9<> ]+`)
	spaceRE    = regexp.MustCompile(`\s+`)

	legacyNormalizer = NewNormalizer(LegacyPreprocessConfig())
)

func LegacyPreprocessConfig() PreprocessConfig {
	return PreprocessConfig{
		Lowercase: true,
	}
}

func NewNormalizer(config PreprocessConfig) *Normalizer {
	keys := make([]string, 0, len(config.HinglishMap))
	for source := range config.HinglishMap {
		keys = append(keys, source)
	}

	sort.Slice(keys, func(i, j int) bool {
		if len(keys[i]) == len(keys[j]) {
			return keys[i] < keys[j]
		}
		return len(keys[i]) > len(keys[j])
	})

	rules := make([]replacementRule, 0, len(keys))
	for _, source := range keys {
		rules = append(rules, replacementRule{
			pattern:     regexp.MustCompile(`\b` + regexp.QuoteMeta(source) + `\b`),
			replacement: config.HinglishMap[source],
		})
	}

	return &Normalizer{
		config:        clonePreprocessConfig(config),
		hinglishRules: rules,
	}
}

func Preprocess(text string) Preprocessed {
	return legacyNormalizer.Preprocess(text)
}

func NormalizeText(text string) string {
	return legacyNormalizer.Normalize(text)
}

func Tokenize(text string) []string {
	return legacyNormalizer.Tokenize(text)
}

func (n *Normalizer) Preprocess(text string) Preprocessed {
	normalized := n.Normalize(text)
	return Preprocessed{
		Normalized: normalized,
		Tokens:     tokenizeNormalized(normalized),
	}
}

func (n *Normalizer) Normalize(text string) string {
	normalized := strings.TrimSpace(text)
	if n.config.Lowercase {
		normalized = strings.ToLower(normalized)
	}

	if replacement := strings.TrimSpace(n.config.ReplaceURLs); replacement != "" {
		normalized = urlRE.ReplaceAllString(normalized, " "+replacement+" ")
	}
	if replacement := strings.TrimSpace(n.config.ReplaceEmails); replacement != "" {
		normalized = emailRE.ReplaceAllString(normalized, " "+replacement+" ")
	}
	if replacement := strings.TrimSpace(n.config.ReplaceNumbers); replacement != "" {
		normalized = numberRE.ReplaceAllString(normalized, " "+replacement+" ")
	}

	for _, rule := range n.hinglishRules {
		normalized = rule.pattern.ReplaceAllString(normalized, rule.replacement)
	}

	normalized = strings.ReplaceAll(normalized, "&", " and ")
	normalized = nonTokenRE.ReplaceAllString(normalized, " ")
	normalized = spaceRE.ReplaceAllString(normalized, " ")
	return strings.TrimSpace(normalized)
}

func (n *Normalizer) Tokenize(text string) []string {
	return tokenizeNormalized(n.Normalize(text))
}

func tokenizeNormalized(normalized string) []string {
	if normalized == "" {
		return nil
	}
	return strings.Fields(normalized)
}

func clonePreprocessConfig(config PreprocessConfig) PreprocessConfig {
	cloned := config
	if len(config.HinglishMap) == 0 {
		cloned.HinglishMap = nil
		return cloned
	}

	cloned.HinglishMap = make(map[string]string, len(config.HinglishMap))
	for source, target := range config.HinglishMap {
		cloned.HinglishMap[source] = target
	}
	return cloned
}
