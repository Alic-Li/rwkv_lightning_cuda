package main

// §5.7 GPU metrics. NVIDIA is sampled through NVML bindings (dlopen, no
// per-request nvidia-smi forking: a dashboard polling a dozen nodes every
// 1–2s would fork hundreds of processes per minute). AMD is sampled through
// rocm-smi with a short cache for the same reason. When nothing can be
// sampled we return available=false plus a concrete reason — never zeros,
// which read as "service down" on a dashboard.

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	vendorNvidia  = "nvidia"
	vendorAMD     = "amd"
	vendorUnknown = "unknown"
)

// gpuSample mirrors §6.3. Temperature and power are optional per vendor:
// they are omitted entirely when unavailable, never zero-filled.
type gpuSample struct {
	Index              int    `json:"index"`
	Name               string `json:"name"`
	MemoryTotalBytes   uint64 `json:"memory_total_bytes"`
	MemoryUsedBytes    uint64 `json:"memory_used_bytes"`
	UtilizationPercent int    `json:"utilization_percent"`
	TemperatureC       *int   `json:"temperature_c,omitempty"`
	PowerWatts         *int   `json:"power_watts,omitempty"`
	// Of MemoryUsedBytes, what the launcher's own runtime holds. Optional:
	// only NVML attributes memory per process, and only for the PIDs of a
	// runtime this launcher started. When it is absent the console falls back
	// to the device spec, which can only split a card all-or-nothing.
	OwnMemoryBytes *uint64 `json:"own_memory_bytes,omitempty"`
}

type metricsResponse struct {
	Available bool        `json:"available"`
	Vendor    string      `json:"vendor"`
	Reason    string      `json:"reason,omitempty"`
	SampledAt int64       `json:"sampled_at,omitempty"`
	GPUs      []gpuSample `json:"gpus"`
}

// rootPID is the runtime this launcher started, whose memory the sample
// attributes where the driver can. Zero means no runtime of ours: pass it
// when the sample only needs the aggregate numbers.
func sampleMetrics(rootPID int) metricsResponse {
	vendor, why := gpuVendor()
	resp := metricsResponse{Vendor: vendor, GPUs: []gpuSample{}}
	var gpus []gpuSample
	var err error
	switch vendor {
	case vendorNvidia:
		gpus, err = nvmlSample(rootPID)
	case vendorAMD:
		gpus, err = rocmSample()
	default:
		resp.Reason = why
		return resp
	}
	if err != nil {
		resp.Reason = err.Error()
		return resp
	}
	resp.Available = true
	resp.SampledAt = time.Now().Unix()
	resp.GPUs = gpus
	return resp
}

var (
	vendorOnce sync.Once
	vendorName string
	vendorWhy  string
)

// gpuVendor decides which sampling path to use. NVML wins when it
// initializes; otherwise rocm-smi; otherwise unknown. The result is static
// for the process lifetime. The "why" text is surfaced as the unavailable
// reason on machines without a GPU management interface.
func gpuVendor() (string, string) {
	vendorOnce.Do(func() {
		nvErr := nvmlInit()
		if nvErr == nil {
			vendorName = vendorNvidia
			return
		}
		if _, lookErr := exec.LookPath("rocm-smi"); lookErr == nil {
			vendorName = vendorAMD
			return
		}
		vendorName = vendorUnknown
		vendorWhy = "nvml: " + nvErr.Error() + "; rocm-smi: not found in PATH"
	})
	return vendorName, vendorWhy
}

// rocm-smi sampling, cached for one second so 1–2s dashboard polling costs
// at most one child process per second.
var rocmCache struct {
	mu   sync.Mutex
	at   time.Time
	gpus []gpuSample
	err  error
}

func rocmSample() ([]gpuSample, error) {
	rocmCache.mu.Lock()
	defer rocmCache.mu.Unlock()
	if time.Since(rocmCache.at) < time.Second {
		return rocmCache.gpus, rocmCache.err
	}
	rocmCache.at = time.Now()
	rocmCache.gpus, rocmCache.err = rocmSampleUncached()
	return rocmCache.gpus, rocmCache.err
}

// rocmSampleUncached parses `rocm-smi --json`. Field names vary across ROCm
// releases; the parser matches by suffix so minor label changes do not
// break sampling. T1 (real-machine verification on the W7900 target) is
// still open: the accepted key set must be confirmed there.
func rocmSampleUncached() ([]gpuSample, error) {
	// A hung rocm-smi would block every later metrics request behind
	// rocmCache.mu, and through sampleFreestDevice it would take l.mu down
	// with it (l.start holds it). Bound it exactly like nvidia-smi is bounded
	// in devices.go.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "rocm-smi", "--json",
		"--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--showproductname").Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			return nil, fmt.Errorf("rocm-smi failed: %s", strings.TrimSpace(string(ee.Stderr)))
		}
		return nil, fmt.Errorf("rocm-smi failed: %v", err)
	}
	var cards map[string]map[string]any
	if err := json.Unmarshal(out, &cards); err != nil {
		return nil, fmt.Errorf("rocm-smi returned invalid JSON: %v", err)
	}
	gpus := []gpuSample{}
	for key, fields := range cards {
		if !strings.HasPrefix(key, "card") {
			continue
		}
		idx := 0
		if n, err := strconv.Atoi(strings.TrimPrefix(key, "card")); err == nil {
			idx = n
		}
		gpu := gpuSample{Index: idx, Name: rocmString(fields, "Card series")}
		if gpu.Name == "" {
			gpu.Name = "AMD GPU " + key
		}
		gpu.MemoryTotalBytes = rocmNumber(fields, "VRAM Total Memory (B)")
		gpu.MemoryUsedBytes = rocmNumber(fields, "VRAM Total Used Memory (B)")
		gpu.UtilizationPercent = int(rocmNumber(fields, "GPU use (%)"))
		// Prefer the junction (hotspot) sensor, fall back to edge.
		if t, ok := rocmOptional(fields, "Temperature (Sensor junction) (C)"); ok {
			gpu.TemperatureC = &t
		} else if t, ok := rocmOptional(fields, "Temperature (Sensor edge) (C)"); ok {
			gpu.TemperatureC = &t
		}
		if p, ok := rocmOptional(fields, "Average Graphics Package Power (W)"); ok {
			gpu.PowerWatts = &p
		}
		gpus = append(gpus, gpu)
	}
	if len(gpus) == 0 {
		return nil, fmt.Errorf("rocm-smi reported no cards")
	}
	return gpus, nil
}

func rocmString(fields map[string]any, suffix string) string {
	for k, v := range fields {
		if strings.HasSuffix(k, suffix) {
			if s, ok := v.(string); ok {
				return strings.TrimSpace(s)
			}
		}
	}
	return ""
}

func rocmNumber(fields map[string]any, suffix string) uint64 {
	for k, v := range fields {
		if strings.HasSuffix(k, suffix) {
			if s, ok := v.(string); ok {
				if n, err := strconv.ParseUint(strings.ReplaceAll(strings.TrimSpace(s), ",", ""), 10, 64); err == nil {
					return n
				}
			}
		}
	}
	return 0
}

func rocmOptional(fields map[string]any, suffix string) (int, bool) {
	for k, v := range fields {
		if strings.HasSuffix(k, suffix) {
			if s, ok := v.(string); ok {
				if f, err := strconv.ParseFloat(strings.ReplaceAll(strings.TrimSpace(s), ",", ""), 64); err == nil {
					return int(f), true
				}
			}
		}
	}
	return 0, false
}
