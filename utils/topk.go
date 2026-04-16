package utils

import "sort"

func ArgMax(scores []float32) (int, float32) {
	if len(scores) == 0 {
		return -1, 0
	}

	bestIndex := 0
	bestScore := scores[0]
	for i := 1; i < len(scores); i++ {
		if scores[i] > bestScore {
			bestIndex = i
			bestScore = scores[i]
		}
	}

	return bestIndex, bestScore
}

func ScoreMap(labels []string, scores []float32) map[string]float32 {
	result := make(map[string]float32, len(labels))
	for i, label := range labels {
		if i < len(scores) {
			result[label] = scores[i]
		}
	}
	return result
}

func TopKIndices(scores []float32, k int) []int {
	if k <= 0 || len(scores) == 0 {
		return nil
	}

	if k > len(scores) {
		k = len(scores)
	}

	indices := make([]int, len(scores))
	for i := range scores {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		return scores[indices[i]] > scores[indices[j]]
	})

	return indices[:k]
}
