package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/octate/tickets-inf/inference"
	"github.com/octate/tickets-inf/model"
)

func main() {
	modelPath := flag.String("model", "", "path to a JSON model file")
	text := flag.String("text", "Please refund the money for my delayed order", "support ticket text")
	quantized := flag.Bool("quantized", false, "run with int8 weights when available")
	debug := flag.Bool("debug", false, "print the feature vector")
	dumpDemoModel := flag.String("dump-demo-model", "", "write the built-in demo model JSON to this path and exit")
	flag.Parse()

	if *dumpDemoModel != "" {
		if err := writeDemoModel(*dumpDemoModel); err != nil {
			fmt.Fprintf(os.Stderr, "write demo model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("wrote demo model to %s\n", *dumpDemoModel)
		return
	}

	var (
		engine *inference.Engine
		err    error
	)

	if *modelPath != "" {
		engine, err = inference.LoadEngineFromFile(*modelPath, inference.Config{
			UseQuantized: *quantized,
			Debug:        *debug,
			DebugWriter:  os.Stdout,
		})
	} else {
		demoModel, buildErr := model.NewDemoModel()
		if buildErr != nil {
			fmt.Fprintf(os.Stderr, "build demo model: %v\n", buildErr)
			os.Exit(1)
		}
		engine, err = inference.NewEngine(demoModel, inference.Config{
			UseQuantized: *quantized,
			Debug:        *debug,
			DebugWriter:  os.Stdout,
		})
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "initialize engine: %v\n", err)
		os.Exit(1)
	}

	result := engine.Predict(*text)
	output, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode prediction: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(string(output))
}

func writeDemoModel(path string) error {
	payload, err := json.MarshalIndent(model.NewDemoJSONModel(), "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, payload, 0o644)
}
