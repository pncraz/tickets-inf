package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/octate/tickets-inf/benchmark"
	"github.com/octate/tickets-inf/inference"
)

func main() {
	localModelPath := flag.String("local-model", "", "path to the local JSON model artifact")
	datasetPath := flag.String("dataset", "testdata/benchmark_cases.jsonl", "path to a benchmark JSONL dataset")
	targetsFlag := flag.String("targets", "local,openai:gpt-5-mini,anthropic:claude-sonnet-4-20250514,gemini:gemini-2.5-flash", "comma-separated benchmark targets")
	jsonOutputPath := flag.String("json-output", "", "optional path to write the full benchmark report as JSON")
	includeCases := flag.Bool("include-cases", false, "include per-case results in the JSON report")
	debug := flag.Bool("debug", false, "enable debug mode for the local model target")
	flag.Parse()

	examples, err := benchmark.LoadJSONL(*datasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load benchmark dataset: %v\n", err)
		os.Exit(1)
	}

	targetSpecs := splitTargets(*targetsFlag)
	targets, labels, err := benchmark.ResolveTargets(targetSpecs, *localModelPath, inference.Config{
		Debug:        *debug,
		UseQuantized: true,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve targets: %v\n", err)
		os.Exit(1)
	}

	report := benchmark.Run(*datasetPath, labels, examples, targets, *includeCases)
	fmt.Println(benchmark.PrintSummary(report))

	if *jsonOutputPath == "" {
		return
	}

	payload, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal benchmark report: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*jsonOutputPath, payload, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write benchmark report: %v\n", err)
		os.Exit(1)
	}
}

func splitTargets(value string) []string {
	parts := strings.Split(value, ",")
	targets := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			targets = append(targets, part)
		}
	}
	return targets
}
