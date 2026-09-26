package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestParseDeviceSet(t *testing.T) {
	for spec, want := range map[string][]int{
		"0":     {0},
		"1":     {1},
		"0,1":   {0, 1},
		" 0 ,1": {0, 1},
	} {
		set := parseDeviceSet(spec)
		if set == nil {
			t.Fatalf("parse(%q) failed", spec)
		}
		for _, d := range want {
			if !set[d] {
				t.Fatalf("parse(%q) missing %d", spec, d)
			}
		}
	}
	for _, spec := range []string{"", "GPU-abc", "0,,1", "-1", "0:1"} {
		if parseDeviceSet(spec) != nil {
			t.Fatalf("parse(%q) should be undeterminable", spec)
		}
	}
}

func TestDevicesOverlap(t *testing.T) {
	explicit := func(spec string) resolvedDevices { return resolvedDevices{spec: spec, explicit: true} }
	cases := []struct {
		a, b   resolvedDevices
		over   bool
		reason string
	}{
		{explicit("0"), explicit("1"), false, "disjoint cards must run in parallel"},
		{explicit("0"), explicit("0"), true, "same card collides"},
		{explicit("0,1"), explicit("1"), true, "spec intersection collides"},
		{explicit("0,1"), explicit("2,3"), false, "disjoint multi-card specs pass"},
		{explicit("0"), resolvedDevices{}, true, "undeterminable side blocks"},
		{resolvedDevices{}, resolvedDevices{}, true, "both undeterminable blocks"},
		{explicit("GPU-uuid"), explicit("1"), true, "unparseable spec blocks"},
	}
	for _, c := range cases {
		if got := devicesOverlap(c.a, c.b); got != c.over {
			t.Errorf("devicesOverlap(%+v, %+v) = %v, want %v (%s)", c.a, c.b, got, c.over, c.reason)
		}
	}
}

func TestApplyVisibleDevices(t *testing.T) {
	base := []string{"PATH=/usr/bin", "CUDA_VISIBLE_DEVICES=9"}
	out := applyVisibleDevices(base, "0,1", vendorNvidia)
	want := []string{"PATH=/usr/bin", "CUDA_VISIBLE_DEVICES=0,1"}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("nvidia inject: %v", out)
	}
	out = applyVisibleDevices([]string{"PATH=/usr/bin", "CUDA_VISIBLE_DEVICES=9"}, "1", vendorAMD)
	want = []string{"PATH=/usr/bin", "HIP_VISIBLE_DEVICES=1", "ROCR_VISIBLE_DEVICES=1"}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("amd inject: %v", out)
	}
	if out := applyVisibleDevices(base, "", vendorNvidia); !reflect.DeepEqual(out, base) {
		t.Fatalf("empty spec must not touch env: %v", out)
	}
}

func TestResolveVisibleDevices(t *testing.T) {
	l := newLauncher()
	spec := "0,1"
	if r := l.resolveVisibleDevices(&spec); !r.explicit || r.spec != "0,1" {
		t.Fatalf("request body wins: %+v", r)
	}
	empty := ""
	if r := l.resolveVisibleDevices(&empty); r.explicit || r.spec != "" {
		t.Fatalf("explicit empty string means no injection: %+v", r)
	}
	l.card = "2"
	if r := l.resolveVisibleDevices(nil); !r.explicit || r.spec != "2" {
		t.Fatalf("--card applies when the request is silent: %+v", r)
	}
	l.card = ""
	// The device-0 status quo is gone: a silent request now auto-places on
	// the freest card. Stub the sampler — the real one depends on the host.
	restore := autoDeviceSpec
	autoDeviceSpec = func() string { return "5" }
	defer func() { autoDeviceSpec = restore }()
	if r := l.resolveVisibleDevices(nil); !r.explicit || !r.auto || r.spec != "5" {
		t.Fatalf("silent request must auto-place: %+v", r)
	}
	autoDeviceSpec = func() string { return "" }
	if r := l.resolveVisibleDevices(nil); r.explicit {
		t.Fatalf("unsampleable host must stay unpinned: %+v", r)
	}
	autoDeviceSpec = restore
	t.Setenv("CUDA_VISIBLE_DEVICES", "3")
	if r := l.resolveVisibleDevices(nil); r.explicit || r.auto {
		t.Fatalf("inherited env wins over auto placement and is not explicit: %+v", r)
	}
	if r := l.resolveVisibleDevices(&spec); !r.explicit || r.spec != "0,1" {
		t.Fatalf("request overrides inherited env: %+v", r)
	}
}

// The --card flag is a hard pin: an explicit request outside it — including
// the empty "inject nothing" spec — must be refused, while a silent request
// keeps falling through to the pinned card.
func TestValidateDeviceRequest(t *testing.T) {
	l := newLauncher()
	if err := l.validateDeviceRequest(nil); err != nil {
		t.Fatalf("unpinned launcher accepts anything: %v", err)
	}
	spec := "1"
	if err := l.validateDeviceRequest(&spec); err != nil {
		t.Fatalf("unpinned launcher accepts explicit specs: %v", err)
	}
	l.card = "1"
	same := "1"
	if err := l.validateDeviceRequest(&same); err != nil {
		t.Fatalf("the pinned spec itself is allowed: %v", err)
	}
	reordered := " 1 "
	if err := l.validateDeviceRequest(&reordered); err != nil {
		t.Fatalf("same card, different spelling is allowed: %v", err)
	}
	if err := l.validateDeviceRequest(nil); err != nil {
		t.Fatalf("silent request falls through to --card: %v", err)
	}
	for _, bad := range []string{"0", "2", "", "0,1", "GPU-uuid"} {
		spec := bad
		if err := l.validateDeviceRequest(&spec); err == nil {
			t.Fatalf("pinned launcher must refuse %q", bad)
		} else if !strings.Contains(err.Error(), `"1"`) {
			t.Fatalf("refusal must name the pinned card: %v", err)
		}
	}
	l.card = "0,1"
	multi := "1,0"
	if err := l.validateDeviceRequest(&multi); err != nil {
		t.Fatalf("same set in another order stays inside the pin: %v", err)
	}
	outside := "2"
	if err := l.validateDeviceRequest(&outside); err == nil {
		t.Fatalf("multi-card pin must still refuse outside cards")
	}
}

func TestPickFreestDevice(t *testing.T) {
	if got := pickFreestDevice(nil); got != "" {
		t.Fatalf("empty GPU list: %q", got)
	}
	gpus := []gpuSample{
		{Index: 0, MemoryTotalBytes: 48 << 30, MemoryUsedBytes: 12 << 30},
		{Index: 1, MemoryTotalBytes: 96 << 30, MemoryUsedBytes: 60 << 30},
		{Index: 2, MemoryTotalBytes: 96 << 30, MemoryUsedBytes: 0},
	}
	// Most free, not least used: card 0 (36 GiB free) beats the busier 96G
	// card, but the empty 96G card wins overall.
	if got := pickFreestDevice(gpus[:2]); got != "0" {
		t.Fatalf("picked %q, want 0", got)
	}
	if got := pickFreestDevice(gpus); got != "2" {
		t.Fatalf("picked %q, want 2", got)
	}
}

func TestFreestFromSmiCSV(t *testing.T) {
	// nounits output is bare MiB; a unit-suffixed row is skipped, which is
	// fine — real nvidia-smi never emits one with these flags.
	csv := "0, 12288\n1, 81052\n2, 12288 MiB\n"
	if got := freestFromSmiCSV(csv); got != "1" {
		t.Fatalf("picked %q, want 1", got)
	}
	for _, broken := range []string{"", "GPU-abc, 81052 MiB", "0, -5 MiB", "no gpu output"} {
		if got := freestFromSmiCSV(broken); got != "" {
			t.Fatalf("broken input %q: picked %q, want empty", broken, got)
		}
	}
}

func TestSanitizeDeviceTag(t *testing.T) {
	for in, want := range map[string]string{"0,1": "0-1", "GPU-aa::bb": "GPU-aa--bb", " 0 ": "0", "a b/c": "a-b-c"} {
		if got := sanitizeDeviceTag(in); got != want {
			t.Errorf("sanitizeDeviceTag(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDeviceTuneCache(t *testing.T) {
	req := startRequest{ModelPath: "/data/models/rwkv7-g1i.pth", StateDBPath: "/state/rwkv_sessions.db"}
	dev := resolvedDevices{spec: "0,1", explicit: true}
	got := deviceTuneCache(req, dev)
	// Mirror the implementation's path resolution so the expectation is
	// portable across OS path separators (and Windows' drive-less IsAbs).
	db := req.StateDBPath
	if !filepath.IsAbs(db) {
		db = filepath.Join(appDir(), db)
	}
	want := filepath.Join(filepath.Dir(db), "rwkv7-g1i.dev-0-1.w8a16.tune")
	if got != want {
		t.Fatalf("deviceTuneCache = %q, want %q", got, want)
	}
	req.EnableDynamicLoading = true
	if got := deviceTuneCache(req, dev); got != "" {
		t.Fatalf("dynamic loading keeps the C++ per-model default: %q", got)
	}
	if got := deviceTuneCache(req, resolvedDevices{}); got != "" {
		t.Fatalf("no devices, no override: %q", got)
	}
}

func TestRuntimeArgsInjectsDeviceTuneCache(t *testing.T) {
	l := newLauncher()
	model := testFile(t, "rwkv7-g1i.pth", "model")
	vocab := testFile(t, "vocab.txt", "vocab")
	req := startRequest{ModelPath: model, VocabPath: vocab, Port: "8000", StateDBPath: "sessions.db"}
	dev := "0,1"
	args, err := l.runtimeArgs(req, resolvedDevices{})
	if err != nil {
		t.Fatal(err)
	}
	if hasTuneCacheArg(args) {
		t.Fatalf("expected no --tune-cache without visible_devices: %v", args)
	}
	req.VisibleDevices = &dev
	args, err = l.runtimeArgs(req, resolvedDevices{spec: "0,1", explicit: true})
	if err != nil {
		t.Fatal(err)
	}
	if !hasTuneCacheArg(args) {
		t.Fatalf("expected card-bound --tune-cache with visible_devices: %v", args)
	}
	found := ""
	for i, a := range args {
		if a == "--tune-cache" && i+1 < len(args) {
			found = args[i+1]
		}
	}
	if found == "" || !strings.Contains(found, "dev-0-1.w8a16.tune") {
		t.Fatalf("tune cache not card-bound: %q", found)
	}
}

func hasTuneCacheArg(args []string) bool {
	for _, a := range args {
		if a == "--tune-cache" {
			return true
		}
	}
	return false
}

// visible_devices must not inherit across starts. Every other field does
// (start decodes onto the saved config), but an absent visible_devices means
// "resolve through the §5.8 chain" — inheriting the previous pin would make
// auto placement unreachable once a card had been named once.
func TestStartDoesNotInheritVisibleDevices(t *testing.T) {
	// Nothing inherited from the host, so a nil request falls all the way to
	// auto placement — which is exactly what we use as the probe.
	for _, name := range deviceEnvVars {
		t.Setenv(name, "")
	}
	restore := autoDeviceSpec
	defer func() { autoDeviceSpec = restore }()

	// runtimeAction refuses outright in client form, so give this launcher a
	// runtime binary to be an agent for (same trick as the capability tests).
	if err := os.WriteFile(backendExecutable(), []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(backendExecutable()) })

	pinned := "1"
	newPinnedLauncher := func() *launcher {
		l := newLauncher()
		l.config = startRequest{ModelPath: "/data/model.pth", Port: "8000", VisibleDevices: &pinned}
		return l
	}
	// Call runtimeAction directly: routing through handler() would hit
	// agentGate first, since a test binary has no runtime next to it.
	post := func(l *launcher, action, body string) {
		t.Helper()
		r := httptest.NewRequest("POST", "http://127.0.0.1:10721/api/v1/runtime/"+action,
			strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		_ = l.runtimeAction(action, w, r)
		// The start fails at launch (no runtime binary); device resolution has
		// already happened by then, which is all this test observes.
	}

	// A body with no visible_devices key — exactly what the picker's auto
	// mode sends, since applyDevice() deletes the field. Auto placement must
	// run; before the fix the saved "1" was inherited and it never did.
	sampled := false
	autoDeviceSpec = func() string { sampled = true; return "7" }
	post(newPinnedLauncher(), "start", `{"model_path":"/nope/missing.pth","port":"8000"}`)
	if !sampled {
		t.Fatal("an omitted visible_devices inherited the saved pin instead of falling through to the §5.8 chain")
	}

	// An explicit spec in the body still wins over auto placement.
	sampled = false
	post(newPinnedLauncher(), "start", `{"model_path":"/nope/missing.pth","visible_devices":"0"}`)
	if sampled {
		t.Fatal("an explicit visible_devices must not fall through to auto placement")
	}

	// restart keeps the saved config, card included: it is the same runtime.
	sampled = false
	post(newPinnedLauncher(), "restart", "")
	if sampled {
		t.Fatal("restart must reuse the saved card, not re-run auto placement")
	}
}

// rwkv_quantize is a CPU tool (tools/CMakeLists.txt links neither
// rwkv::backend nor the CUDA includes), so the Agent must not apply the
// --card pin to it and must not sample free VRAM on its behalf.
func TestQuantizationIsCPUOnly(t *testing.T) {
	for _, name := range deviceEnvVars {
		t.Setenv(name, "")
	}
	restore := autoDeviceSpec
	defer func() { autoDeviceSpec = restore }()

	// Two guards sit in front of the device handling and would otherwise hide
	// what this test checks: the client-only role check (needs a runtime
	// binary present) and "quantizer is unavailable" (needs the tool itself).
	for _, path := range []string{backendExecutable(), toolBinary("rwkv_quantize")} {
		if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.Remove(path) })
	}

	dir := t.TempDir()
	input := filepath.Join(dir, "model.pth")
	if err := os.WriteFile(input, []byte("m"), 0600); err != nil {
		t.Fatal(err)
	}
	body := func(devices *string) *http.Request {
		payload, err := json.Marshal(quantizeRequest{
			InputPath:      input,
			OutputPath:     filepath.Join(dir, "model.rwkvq"),
			Format:         "w8a16",
			VisibleDevices: devices,
		})
		if err != nil {
			t.Fatal(err)
		}
		r := httptest.NewRequest("POST", "http://127.0.0.1:10721/api/v1/jobs/quantization",
			bytes.NewReader(payload))
		r.Header.Set("Content-Type", "application/json")
		return r
	}

	// A --card pin must not refuse a CPU job that names another card.
	other := "1"
	l := newLauncher()
	l.card = "0"
	autoDeviceSpec = func() string { t.Error("quantization sampled free VRAM"); return "" }
	err := l.handleQuantizationStart(httptest.NewRecorder(), body(&other))
	t.Cleanup(func() { _ = l.quantization.stop() })
	if err != nil && strings.Contains(err.Error(), "pinned to GPU") {
		t.Fatalf("a CPU job was refused by the --card pin: %v", err)
	}

	// And a silent request must not trigger free-VRAM placement either: the
	// autoDeviceSpec stub above fails the test if it is ever called.
	l2 := newLauncher()
	if err := l2.handleQuantizationStart(httptest.NewRecorder(), body(nil)); err != nil &&
		strings.Contains(err.Error(), "pinned to GPU") {
		t.Fatalf("unexpected pin refusal: %v", err)
	}
	t.Cleanup(func() { _ = l2.quantization.stop() })

	// The CLI never carries a device selection.
	args, err := quantizationArgs(quantizeRequest{
		InputPath: input, OutputPath: filepath.Join(dir, "b.rwkvq"),
		Format: "w8a16", VisibleDevices: &other,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range args {
		if a == "1" || strings.Contains(a, "device") || strings.Contains(a, "visible") {
			t.Fatalf("quantization args carry a device selection: %v", args)
		}
	}
}

// A missed ROCm memory key must not wrap the free-VRAM subtraction and pin
// every auto-placed process onto a phantom card.
func TestPickFreestDeviceSurvivesMissingTotal(t *testing.T) {
	gpus := []gpuSample{
		{Index: 0, MemoryTotalBytes: 24 << 30, MemoryUsedBytes: 1 << 30},
		// rocmNumber returns 0 for a key it cannot match: total 0, used 8 GiB.
		{Index: 1, MemoryTotalBytes: 0, MemoryUsedBytes: 8 << 30},
	}
	if got := pickFreestDevice(gpus); got != "0" {
		t.Fatalf("picked %q; a zero total must clamp to 0 free, not wrap to 2^64", got)
	}
}

// The card-bound tune cache implies a one-off retune; say so in the log.
func TestTuneCacheNote(t *testing.T) {
	dir := t.TempDir()
	req := startRequest{
		ModelPath:   filepath.Join(dir, "rwkv7-g1i.pth"),
		StateDBPath: filepath.Join(dir, "sessions.db"),
	}
	dev := resolvedDevices{spec: "1", explicit: true}

	note := tuneCacheNote(req, dev)
	if note == "" || !strings.Contains(note, "retune") {
		t.Fatalf("missing cache should announce a retune, got %q", note)
	}

	// Once the cache exists there is nothing to announce.
	if err := os.WriteFile(deviceTuneCache(req, dev), []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	if note := tuneCacheNote(req, dev); note != "" {
		t.Fatalf("existing cache must stay quiet, got %q", note)
	}

	// An explicit --tune-cache is the caller's business.
	req.TuneCache = "/somewhere/custom.tune"
	if note := tuneCacheNote(req, resolvedDevices{spec: "2", explicit: true}); note != "" {
		t.Fatalf("explicit tune_cache must stay quiet, got %q", note)
	}
}
