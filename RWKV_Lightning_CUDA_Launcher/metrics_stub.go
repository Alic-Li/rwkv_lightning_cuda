//go:build !linux || !cgo

package main

// NVML fallback for builds without Linux cgo (macOS, Windows, or
// CGO_ENABLED=0). The metrics endpoint still answers — with available=false
// and a concrete reason, never zero-filled GPUs (§5.7).

import (
	"fmt"
	"runtime"
)

func nvmlInit() error {
	return fmt.Errorf("NVML requires a Linux launcher built with cgo (this build: %s, cgo off or unsupported)", runtime.GOOS)
}

func nvmlSample(rootPID int) ([]gpuSample, error) {
	return nil, nvmlInit()
}
