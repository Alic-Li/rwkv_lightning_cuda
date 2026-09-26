package main

// §5.8 visible_devices: GPU selection is done by injecting environment
// variables when the Agent spawns its managed processes, not by adding flags
// to the C++ binaries. CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES are the
// native, zero-change equivalent of a --device flag.

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// resolvedDevices captures which GPU(s) a managed process is pinned to.
//   - spec     is the raw injected spec ("" when nothing is injected);
//   - explicit is true only when the spec came from the request body, the
//     --card flag, or auto placement. An inherited CUDA_VISIBLE_DEVICES is
//     NOT explicit: its meaning for a child process is not something the
//     Agent can reason about.
//   - auto     is true only when the Agent itself chose the card by free-VRAM
//     placement, so callers can log the decision where the user can see it.
type resolvedDevices struct {
	spec     string
	explicit bool
	auto     bool
}

// autoDeviceSpec is a package var so tests can stub the sampler.
var autoDeviceSpec = sampleFreestDevice

// resolveVisibleDevices applies the §5.8 priority chain (highest wins):
//  1. the request body's visible_devices — "" is an explicit "do not inject"
//     and deliberately differs from an absent field, which falls through;
//  2. the Agent's --card default;
//  3. the Agent process's own CUDA/HIP_VISIBLE_DEVICES (children inherit it
//     through os.Environ, nothing is injected here);
//  4. nothing — instead of the device-0 status quo, the Agent places the
//     process on the card with the most free VRAM, so back-to-back starts do
//     not pile up on device 0 while card 1 sits idle. When nothing can be
//     sampled, fall back to "inject nothing" (the child sees every card and
//     the driver picks device 0, exactly the old behavior).
func (l *launcher) resolveVisibleDevices(reqVisible *string) resolvedDevices {
	if reqVisible != nil {
		if *reqVisible == "" {
			return resolvedDevices{}
		}
		return resolvedDevices{spec: strings.TrimSpace(*reqVisible), explicit: true}
	}
	if l.card != "" {
		return resolvedDevices{spec: strings.TrimSpace(l.card), explicit: true}
	}
	for _, name := range deviceEnvVars {
		if os.Getenv(name) != "" {
			return resolvedDevices{} // inherited by children as-is
		}
	}
	if spec := autoDeviceSpec(); spec != "" {
		return resolvedDevices{spec: spec, explicit: true, auto: true}
	}
	return resolvedDevices{}
}

// validateDeviceRequest enforces the --card pin (§5.8): the flag is a hard
// restriction on this launcher, not a default a request body may override.
// A request that names a different spec — including "" ("inject nothing"),
// which would let the driver fall back to device 0 — is refused before any
// process is stopped or spawned.
func (l *launcher) validateDeviceRequest(reqVisible *string) error {
	if l.card == "" || reqVisible == nil {
		return nil
	}
	card := strings.TrimSpace(l.card)
	req := strings.TrimSpace(*reqVisible)
	if req == card {
		return nil
	}
	// A different spelling of the same card set ("1,0" vs "0,1") stays
	// inside the pin; unparseable specs (UUIDs, MIG) never match.
	if cs, rs := parseDeviceSet(card), parseDeviceSet(req); cs != nil && rs != nil && sameDeviceSet(cs, rs) {
		return nil
	}
	return fmt.Errorf("this launcher is pinned to GPU %q by --card; loading on %q is refused", card, req)
}

func sameDeviceSet(a, b map[int]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for d := range a {
		if !b[d] {
			return false
		}
	}
	return true
}

// sampleFreestDevice asks NVML for per-card free VRAM and returns the index
// of the emptiest card. Windows builds have no NVML binding, so fall back to
// one nvidia-smi query (drivers ship it on every platform).
func sampleFreestDevice() string {
	// Placement only needs what is free: no runtime to attribute to.
	if resp := sampleMetrics(0); resp.Available && len(resp.GPUs) > 0 {
		return pickFreestDevice(resp.GPUs)
	}
	return nvidiaSmiFreest()
}

// pickFreestDevice returns the index of the GPU with the most free memory.
// "Most free", not "least used": a 48 GiB card 20% busy still beats an empty
// 24 GiB card for a model that will not fit there anyway.
func pickFreestDevice(gpus []gpuSample) string {
	best := -1
	var bestFree uint64
	for _, gpu := range gpus {
		// rocmNumber returns 0 for any key it cannot match, and the ROCm
		// field names drift between releases (metrics.go). A missed total
		// next to a matched used would wrap this subtraction to ~2^64 and
		// pin every auto-placed process onto that one card forever.
		free := uint64(0)
		if gpu.MemoryTotalBytes > gpu.MemoryUsedBytes {
			free = gpu.MemoryTotalBytes - gpu.MemoryUsedBytes
		}
		if best == -1 || free > bestFree {
			best, bestFree = gpu.Index, free
		}
	}
	if best < 0 {
		return ""
	}
	return strconv.Itoa(best)
}

// nvidiaSmiFreest runs `nvidia-smi --query-gpu=index,memory.free` — one
// short-lived process per start (never per poll), bounded by a 3s deadline.
// Any failure returns "", never a guess.
func nvidiaSmiFreest() string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx,
		"nvidia-smi",
		"--query-gpu=index,memory.free",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return ""
	}
	return freestFromSmiCSV(string(out))
}

// freestFromSmiCSV parses "index, free MiB" lines into the freest index.
// Malformed lines are skipped; anything unparseable as a whole returns "".
func freestFromSmiCSV(out string) string {
	best := -1
	var bestFree int64
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Split(strings.TrimSpace(line), ",")
		if len(fields) != 2 {
			continue
		}
		index, err := strconv.Atoi(strings.TrimSpace(fields[0]))
		if err != nil || index < 0 {
			continue
		}
		mib, err := strconv.ParseInt(strings.TrimSpace(fields[1]), 10, 64)
		if err != nil || mib < 0 {
			continue
		}
		if best == -1 || mib > bestFree {
			best, bestFree = index, mib
		}
	}
	if best < 0 {
		return ""
	}
	return strconv.Itoa(best)
}

var deviceEnvVars = []string{"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}

// parseDeviceSet parses a CUDA_VISIBLE_DEVICES-style spec into device
// indexes. It returns nil for anything that is not a plain comma-separated
// non-negative integer list (UUIDs, MIG selectors, ...): those cannot be
// compared, so callers must treat them as "undeterminable".
func parseDeviceSet(spec string) map[int]bool {
	if spec == "" {
		return nil
	}
	out := map[int]bool{}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil
		}
		n, err := strconv.Atoi(part)
		if err != nil || n < 0 {
			return nil
		}
		out[n] = true
	}
	return out
}

// devicesOverlap reports whether two managed GPU processes may end up on the
// same physical card. §5.8(a): when either side is not explicitly pinned the
// answer must be "yes" — a false block costs the user one extra click, a
// false pass costs an OOM mid-training.
func devicesOverlap(a, b resolvedDevices) bool {
	if !a.explicit || !b.explicit {
		return true
	}
	sa, sb := parseDeviceSet(a.spec), parseDeviceSet(b.spec)
	if sa == nil || sb == nil {
		return true
	}
	for d := range sa {
		if sb[d] {
			return true
		}
	}
	return false
}

// applyVisibleDevices rewrites the child environment to pin it to spec.
// The NVIDIA path sets CUDA_VISIBLE_DEVICES; the AMD path sets both
// HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES (they act at different layers
// and some ROCm releases only honor one). All device variables are removed
// first so a stale inherited value can never combine with the injected one.
func applyVisibleDevices(env []string, spec, vendor string) []string {
	if spec == "" {
		return env
	}
	set := map[string]string{}
	if vendor == vendorAMD {
		set["HIP_VISIBLE_DEVICES"] = spec
		set["ROCR_VISIBLE_DEVICES"] = spec
	} else {
		set["CUDA_VISIBLE_DEVICES"] = spec
	}
	out := make([]string, 0, len(env)+len(set))
	for _, kv := range env {
		key := kv
		if i := strings.IndexByte(kv, '='); i >= 0 {
			key = kv[:i]
		}
		if knownDeviceVar(key) {
			continue // dropped, re-added below with the injected value
		}
		out = append(out, kv)
	}
	// Append in deviceEnvVars order, not map order: deterministic output.
	for _, name := range deviceEnvVars {
		if v, ok := set[name]; ok {
			out = append(out, name+"="+v)
		}
	}
	return out
}

func knownDeviceVar(key string) bool {
	for _, name := range deviceEnvVars {
		if key == name {
			return true
		}
	}
	return false
}

// sanitizeDeviceTag converts a visible_devices spec into a path-safe tag.
func sanitizeDeviceTag(spec string) string {
	var b strings.Builder
	for _, r := range spec {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

// deviceTuneCache derives a card-bound default --tune-cache path (§5.8(c)).
// W8A16/W4A16 split-K and sparse thresholds are tuned per GPU: the C++ default
// path already keys on model stem + GPU model name, but two same-model cards
// share that path, so switching visible_devices between them would silently
// reuse the other card's results. The Agent therefore pins the cache next to
// the state DB, named after the model and the device spec. Explicit
// --tune-cache values and dynamic-loading mode are left untouched (§3.3: the
// per-model default naming applies there).
func deviceTuneCache(req startRequest, devices resolvedDevices) string {
	tag := sanitizeDeviceTag(devices.spec)
	if tag == "" || req.EnableDynamicLoading {
		return ""
	}
	model := req.ModelPath
	if !filepath.IsAbs(model) {
		model = filepath.Join(appDir(), model)
	}
	stem := strings.TrimSuffix(filepath.Base(model), filepath.Ext(model))
	if stem == "" {
		return ""
	}
	dir := appDir()
	if p := req.StateDBPath; p != "" {
		if !filepath.IsAbs(p) {
			p = filepath.Join(appDir(), p)
		}
		dir = filepath.Dir(p)
	}
	return filepath.Join(dir, stem+".dev-"+tag+".w8a16.tune")
}

// tuneCacheNote explains the one-off W8A16 retune that a card-bound cache
// path implies. deviceTuneCache pins the cache next to the state DB and names
// it after the device spec, so the first start on a given card never finds
// the C++ default cache (named after model stem + GPU name) and retunes
// before serving. It costs a couple of minutes, once per model per card — a
// log line is enough, but without one the runtime just looks hung.
func tuneCacheNote(req startRequest, devices resolvedDevices) string {
	if strings.TrimSpace(req.TuneCache) != "" {
		return "" // explicit path: the caller owns it, say nothing
	}
	path := deviceTuneCache(req, devices)
	if path == "" {
		return "" // dynamic loading or no device tag: C++ default naming applies
	}
	if _, err := os.Stat(path); err == nil {
		return ""
	}
	return "W8A16 tuning cache " + filepath.Base(path) + " not found for this card; " +
		"the runtime retunes once before it starts serving (typically a couple of minutes)"
}
