package main

import "strings"

func retiredInferenceRoute(path string) bool {
	return path == "/v1/adapters" || strings.HasPrefix(path, "/v1/adapters/")
}
