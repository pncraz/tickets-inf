package model

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/pncraz/tickets-inf/features"
	"github.com/pncraz/tickets-inf/quantization"
	"github.com/pncraz/tickets-inf/utils"
)

func LoadFile(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model file: %w", err)
	}
	return LoadJSON(data)
}

func LoadJSON(data []byte) (*Model, error) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("decode model json: %w", err)
	}

	if _, ok := envelope["vocab"]; ok {
		var spec ExportedModelJSON
		if err := json.Unmarshal(data, &spec); err != nil {
			return nil, fmt.Errorf("decode exported model json: %w", err)
		}
		return BuildFromExportedModel(spec)
	}

	var spec JSONModel
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, fmt.Errorf("decode legacy model json: %w", err)
	}
	return BuildFromJSONModel(spec)
}

func BuildFromJSONModel(spec JSONModel) (*Model, error) {
	if len(spec.BowVocab) == 0 {
		return nil, fmt.Errorf("bow_vocab must not be empty")
	}
	if len(spec.EmbeddingVocab) == 0 {
		return nil, fmt.Errorf("embedding_vocab must not be empty")
	}

	embeddings, err := buildEmbeddingTable(spec.Embeddings)
	if err != nil {
		return nil, fmt.Errorf("build embeddings: %w", err)
	}

	model := &Model{
		BowVocab:              cloneVocab(spec.BowVocab),
		EmbeddingVocab:        cloneVocab(spec.EmbeddingVocab),
		UseLegacyKeywordFlags: true,
		Preprocess:            features.LegacyPreprocessConfig(),
		Labels: LabelSet{
			Department: cloneStrings(utils.DepartmentLabels),
			Sentiment:  cloneStrings(utils.SentimentLabels),
			LeadIntent: cloneStrings(utils.LeadIntentLabels),
			ChurnRisk:  []string{"low", "high"},
			Intent:     cloneStrings(utils.IntentLabels),
		},
		Embeddings: embeddings,
		HiddenSize: 64,
	}

	if embeddings.Rows <= maxIndex(model.EmbeddingVocab) {
		return nil, fmt.Errorf("embedding matrix rows=%d do not cover embedding vocab max index=%d", embeddings.Rows, maxIndex(model.EmbeddingVocab))
	}

	if unknownTokenID, ok := model.EmbeddingVocab["<unk>"]; ok {
		model.UnknownTokenID = unknownTokenID
		model.HasUnknownToken = true
	}
	if paddingIndex, ok := model.EmbeddingVocab["<pad>"]; ok {
		model.PaddingIndex = paddingIndex
	}

	model.Base1, err = buildDenseLayer(spec.Base.Dense1)
	if err != nil {
		return nil, fmt.Errorf("build base dense1: %w", err)
	}
	model.Base2, err = buildDenseLayer(spec.Base.Dense2)
	if err != nil {
		return nil, fmt.Errorf("build base dense2: %w", err)
	}
	model.DepartmentHead, err = buildDenseLayer(spec.Heads.Department)
	if err != nil {
		return nil, fmt.Errorf("build department head: %w", err)
	}
	model.SentimentHead, err = buildDenseLayer(spec.Heads.Sentiment)
	if err != nil {
		return nil, fmt.Errorf("build sentiment head: %w", err)
	}
	model.LeadIntentHead, err = buildDenseLayer(spec.Heads.LeadIntent)
	if err != nil {
		return nil, fmt.Errorf("build lead_intent head: %w", err)
	}
	model.ChurnRiskHead, err = buildDenseLayer(spec.Heads.ChurnRisk)
	if err != nil {
		return nil, fmt.Errorf("build churn_risk head: %w", err)
	}
	model.IntentHead, err = buildDenseLayer(spec.Heads.Intent)
	if err != nil {
		return nil, fmt.Errorf("build intent head: %w", err)
	}

	if err := validateArchitecture(model); err != nil {
		return nil, err
	}

	return model, nil
}

func BuildFromExportedModel(spec ExportedModelJSON) (*Model, error) {
	if spec.Version <= 0 {
		return nil, fmt.Errorf("exported model version must be positive")
	}
	if len(spec.Vocab.BowVocab) == 0 {
		return nil, fmt.Errorf("exported bow_vocab must not be empty")
	}
	if len(spec.Vocab.EmbeddingVocab) == 0 {
		return nil, fmt.Errorf("exported embedding_vocab must not be empty")
	}
	if spec.Layers.Base1.Type != "relu" || spec.Layers.Base3.Type != "relu" {
		return nil, fmt.Errorf("exported base layers must follow linear->relu->linear->relu")
	}

	embeddingRows, embeddingDim, embeddingValues, err := flattenFloatMatrix(spec.Embedding.Weights)
	if err != nil {
		return nil, fmt.Errorf("build exported embeddings: %w", err)
	}

	model := &Model{
		BowVocab:       cloneVocab(spec.Vocab.BowVocab),
		EmbeddingVocab: cloneVocab(spec.Vocab.EmbeddingVocab),
		Keywords:       cloneStrings(spec.Vocab.Keywords),
		MaxTokens:      spec.Vocab.MaxTokens,
		PaddingIndex:   spec.Embedding.PaddingIdx,
		UseLog1pBow:    true,
		Preprocess:     spec.Preprocess,
		Labels:         cloneLabelSet(spec.Labels),
		Version:        spec.Version,
		ModelType:      spec.Metadata.ModelType,
		DenseSize:      spec.Metadata.DenseSize,
		HiddenSize:     spec.Metadata.HiddenSize,
		Embeddings: EmbeddingTable{
			Rows:   embeddingRows,
			Dim:    embeddingDim,
			Values: embeddingValues,
		},
	}

	if unknownTokenID, ok := model.EmbeddingVocab["<unk>"]; ok {
		model.UnknownTokenID = unknownTokenID
		model.HasUnknownToken = true
	}
	if paddingIndex, ok := model.EmbeddingVocab["<pad>"]; ok && paddingIndex != model.PaddingIndex {
		return nil, fmt.Errorf("embedding padding_idx mismatch: got=%d want=%d", model.PaddingIndex, paddingIndex)
	}

	if model.Embeddings.Rows <= maxIndex(model.EmbeddingVocab) {
		return nil, fmt.Errorf("embedding matrix rows=%d do not cover embedding vocab max index=%d", model.Embeddings.Rows, maxIndex(model.EmbeddingVocab))
	}

	model.Base1, err = buildExportedLinearLayer(spec.Layers.Base0)
	if err != nil {
		return nil, fmt.Errorf("build exported base_0: %w", err)
	}
	model.Base2, err = buildExportedLinearLayer(spec.Layers.Base2)
	if err != nil {
		return nil, fmt.Errorf("build exported base_2: %w", err)
	}
	model.DepartmentHead, err = buildExportedLinearLayer(spec.Heads.Department)
	if err != nil {
		return nil, fmt.Errorf("build exported department head: %w", err)
	}
	model.SentimentHead, err = buildExportedLinearLayer(spec.Heads.Sentiment)
	if err != nil {
		return nil, fmt.Errorf("build exported sentiment head: %w", err)
	}
	model.LeadIntentHead, err = buildExportedLinearLayer(spec.Heads.LeadIntent)
	if err != nil {
		return nil, fmt.Errorf("build exported lead_intent head: %w", err)
	}
	model.ChurnRiskHead, err = buildExportedLinearLayer(spec.Heads.ChurnRisk)
	if err != nil {
		return nil, fmt.Errorf("build exported churn_risk head: %w", err)
	}
	model.IntentHead, err = buildExportedLinearLayer(spec.Heads.Intent)
	if err != nil {
		return nil, fmt.Errorf("build exported intent head: %w", err)
	}

	if err := validateArchitecture(model); err != nil {
		return nil, err
	}

	return model, nil
}

func validateArchitecture(model *Model) error {
	expectedInput := model.FeatureSize()
	if model.DenseSize > 0 && model.DenseSize != maxIndex(model.BowVocab)+1+model.KeywordSize() {
		return fmt.Errorf("metadata dense_size mismatch: got=%d want=%d", model.DenseSize, maxIndex(model.BowVocab)+1+model.KeywordSize())
	}
	if model.Base1.In != expectedInput {
		return fmt.Errorf("base dense1 input mismatch: got=%d want=%d", model.Base1.In, expectedInput)
	}
	if model.HiddenSize > 0 {
		if model.Base1.Out != model.HiddenSize {
			return fmt.Errorf("base dense1 output mismatch: got=%d want=%d", model.Base1.Out, model.HiddenSize)
		}
		if model.Base2.In != model.HiddenSize || model.Base2.Out != model.HiddenSize/2 {
			return fmt.Errorf("base dense2 shape mismatch: got=%d->%d want=%d->%d", model.Base2.In, model.Base2.Out, model.HiddenSize, model.HiddenSize/2)
		}
	} else {
		if model.Base1.Out != 64 {
			return fmt.Errorf("base dense1 output mismatch: got=%d want=64", model.Base1.Out)
		}
		if model.Base2.In != 64 || model.Base2.Out != 32 {
			return fmt.Errorf("base dense2 shape mismatch: got=%d->%d want=64->32", model.Base2.In, model.Base2.Out)
		}
	}
	if model.Embeddings.Dim <= 0 {
		return fmt.Errorf("embedding dimension must be positive")
	}
	if err := validateMultiClassHead("department", model.DepartmentHead, model.Base2.Out, model.Labels.Department); err != nil {
		return err
	}
	if err := validateMultiClassHead("sentiment", model.SentimentHead, model.Base2.Out, model.Labels.Sentiment); err != nil {
		return err
	}
	if err := validateMultiClassHead("lead_intent", model.LeadIntentHead, model.Base2.Out, model.Labels.LeadIntent); err != nil {
		return err
	}
	if err := validateBinaryHead("churn_risk", model.ChurnRiskHead, model.Base2.Out, model.Labels.ChurnRisk); err != nil {
		return err
	}
	if err := validateMultiClassHead("intent", model.IntentHead, model.Base2.Out, model.Labels.Intent); err != nil {
		return err
	}

	supportedSizes := make([]int, 0, 2)
	if model.SupportsFloatInference() {
		supportedSizes = append(supportedSizes, model.ParameterBytes(false))
	}
	if model.SupportsQuantizedInference() {
		supportedSizes = append(supportedSizes, model.ParameterBytes(true))
	}
	if len(supportedSizes) == 0 {
		return fmt.Errorf("model does not contain a usable float32 or quantized inference representation")
	}

	bestSize := supportedSizes[0]
	for _, size := range supportedSizes[1:] {
		if size < bestSize {
			bestSize = size
		}
	}
	if bestSize > 2*1024*1024 {
		return fmt.Errorf("model parameters exceed 2MB budget: %d bytes", bestSize)
	}

	return nil
}

func validateMultiClassHead(name string, layer DenseLayer, expectedInput int, labels []string) error {
	if layer.In != expectedInput {
		return fmt.Errorf("%s head input mismatch: got=%d want=%d", name, layer.In, expectedInput)
	}
	if len(labels) == 0 {
		return fmt.Errorf("%s labels must not be empty", name)
	}
	if layer.Out != len(labels) {
		return fmt.Errorf("%s head output mismatch: got=%d want=%d", name, layer.Out, len(labels))
	}
	return nil
}

func validateBinaryHead(name string, layer DenseLayer, expectedInput int, labels []string) error {
	if layer.In != expectedInput {
		return fmt.Errorf("%s head input mismatch: got=%d want=%d", name, layer.In, expectedInput)
	}
	if len(labels) != 2 {
		return fmt.Errorf("%s labels must contain exactly 2 values", name)
	}
	if layer.Out != 1 {
		return fmt.Errorf("%s head output mismatch: got=%d want=1", name, layer.Out)
	}
	return nil
}

func buildEmbeddingTable(spec JSONEmbedding) (EmbeddingTable, error) {
	if len(spec.Matrix) == 0 && len(spec.MatrixInt8) == 0 {
		return EmbeddingTable{}, fmt.Errorf("embedding matrix must contain float or int8 values")
	}

	table := EmbeddingTable{}
	if len(spec.Matrix) > 0 {
		rows, cols, flat, err := flattenFloatMatrix(spec.Matrix)
		if err != nil {
			return EmbeddingTable{}, err
		}
		table.Rows = rows
		table.Dim = cols
		table.Values = flat
	}

	if len(spec.MatrixInt8) > 0 {
		quantized, err := quantization.NewInt8MatrixFromNested(spec.MatrixInt8, spec.Scales)
		if err != nil {
			return EmbeddingTable{}, err
		}
		table.Quantized = quantized
		if table.Rows == 0 {
			table.Rows = quantized.Rows
			table.Dim = quantized.Cols
		}
	}

	if table.Rows == 0 || table.Dim == 0 {
		return EmbeddingTable{}, fmt.Errorf("embedding matrix shape resolved to zero")
	}
	if table.Quantized != nil && (table.Quantized.Rows != table.Rows || table.Quantized.Cols != table.Dim) {
		return EmbeddingTable{}, fmt.Errorf("embedding float and quantized shapes differ")
	}

	return table, nil
}

func buildDenseLayer(spec JSONDenseLayer) (DenseLayer, error) {
	if len(spec.Bias) == 0 {
		return DenseLayer{}, fmt.Errorf("layer bias must not be empty")
	}
	if len(spec.Weights) == 0 && len(spec.WeightsInt8) == 0 {
		return DenseLayer{}, fmt.Errorf("layer must contain float or int8 weights")
	}

	layer := DenseLayer{
		Bias: append([]float32(nil), spec.Bias...),
	}

	if len(spec.Weights) > 0 {
		rows, cols, flat, err := flattenFloatMatrix(spec.Weights)
		if err != nil {
			return DenseLayer{}, err
		}
		layer.Out = rows
		layer.In = cols
		layer.Weights = flat
	}

	if len(spec.WeightsInt8) > 0 {
		quantized, err := quantization.NewInt8MatrixFromNested(spec.WeightsInt8, spec.Scales)
		if err != nil {
			return DenseLayer{}, err
		}
		layer.Quantized = quantized
		if layer.Out == 0 {
			layer.Out = quantized.Rows
			layer.In = quantized.Cols
		}
	}

	if layer.Out != len(layer.Bias) {
		return DenseLayer{}, fmt.Errorf("bias size mismatch: got=%d want=%d", len(layer.Bias), layer.Out)
	}
	if layer.Quantized != nil && (layer.Quantized.Rows != layer.Out || layer.Quantized.Cols != layer.In) {
		return DenseLayer{}, fmt.Errorf("float and quantized layer shapes differ")
	}

	return layer, nil
}

func buildExportedLinearLayer(spec ExportedLinearLayer) (DenseLayer, error) {
	if spec.Type != "linear_int8_per_row" {
		return DenseLayer{}, fmt.Errorf("unsupported exported layer type: %s", spec.Type)
	}
	if len(spec.Bias) == 0 {
		return DenseLayer{}, fmt.Errorf("exported layer bias must not be empty")
	}

	quantized, err := quantization.NewInt8MatrixFromNested(spec.Weight, spec.Scale)
	if err != nil {
		return DenseLayer{}, err
	}
	if quantized.Rows != spec.OutFeatures || quantized.Cols != spec.InFeatures {
		return DenseLayer{}, fmt.Errorf("exported layer shape mismatch: matrix=%dx%d metadata=%dx%d", quantized.Rows, quantized.Cols, spec.OutFeatures, spec.InFeatures)
	}
	if len(spec.Bias) != spec.OutFeatures {
		return DenseLayer{}, fmt.Errorf("exported bias size mismatch: got=%d want=%d", len(spec.Bias), spec.OutFeatures)
	}

	return DenseLayer{
		In:        spec.InFeatures,
		Out:       spec.OutFeatures,
		Bias:      append([]float32(nil), spec.Bias...),
		Quantized: quantized,
	}, nil
}

func flattenFloatMatrix(values [][]float32) (rows int, cols int, flat []float32, err error) {
	if len(values) == 0 {
		return 0, 0, nil, fmt.Errorf("matrix must have at least one row")
	}
	cols = len(values[0])
	if cols == 0 {
		return 0, 0, nil, fmt.Errorf("matrix must have at least one column")
	}

	flat = make([]float32, 0, len(values)*cols)
	for row, current := range values {
		if len(current) != cols {
			return 0, 0, nil, fmt.Errorf("row %d has inconsistent width", row)
		}
		flat = append(flat, current...)
	}

	return len(values), cols, flat, nil
}

func cloneVocab(source map[string]int) map[string]int {
	cloned := make(map[string]int, len(source))
	for token, index := range source {
		cloned[token] = index
	}
	return cloned
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	return append([]string(nil), values...)
}

func cloneLabelSet(labels LabelSet) LabelSet {
	return LabelSet{
		Department: cloneStrings(labels.Department),
		Sentiment:  cloneStrings(labels.Sentiment),
		LeadIntent: cloneStrings(labels.LeadIntent),
		ChurnRisk:  cloneStrings(labels.ChurnRisk),
		Intent:     cloneStrings(labels.Intent),
	}
}
