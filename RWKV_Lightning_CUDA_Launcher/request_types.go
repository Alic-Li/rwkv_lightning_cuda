package main

// Request bodies for the /api/v1 control plane.
// Keep JSON field names aligned with docs/control-plane-api.md.
// VisibleDevices is a pointer: omitted and explicitly empty have different semantics.

type startRequest struct {
	ModelPath            string  `json:"model_path"`
	VocabPath            string  `json:"vocab_path"`
	Port                 string  `json:"port"`
	Password             string  `json:"password"`
	UseWKV32             bool    `json:"use_wkv32"`
	ChunkLoad            bool    `json:"chunk_load"`
	EnableDynamicLoading bool    `json:"enable_dynamic_loading"`
	ChunkSize            int     `json:"chunk_size"`
	StateDBPath          string  `json:"state_db_path"`
	TuneCache            string  `json:"tune_cache"`
	VisibleDevices       *string `json:"visible_devices,omitempty"`
}
type tuneRequest struct {
	Method         string  `json:"method"`
	Rank           int     `json:"rank"`
	Alpha          float64 `json:"alpha"`
	Targets        string  `json:"targets"`
	State          string  `json:"state"`
	Resume         string  `json:"resume"`
	Model          string  `json:"model"`
	Data           string  `json:"data"`
	Output         string  `json:"output"`
	Vocab          string  `json:"vocab"`
	Ctx            int     `json:"ctx"`
	Chunk          int     `json:"chunk"`
	Epochs         int     `json:"epochs"`
	BatchSize      int     `json:"batch_size"`
	MaxSteps       int     `json:"max_steps"`
	LR             float64 `json:"lr"`
	LRFinal        float64 `json:"lr_final"`
	WarmupSteps    int     `json:"warmup_steps"`
	SaveEvery      int     `json:"save_every"`
	Seed           int     `json:"seed"`
	Optimizer      string  `json:"optimizer"`
	WKVTape        bool    `json:"wkv_tape"`
	VisibleDevices *string `json:"visible_devices,omitempty"`
}
type quantizeRequest struct {
	InputPath      string  `json:"input_path"`
	OutputPath     string  `json:"output_path"`
	Format         string  `json:"format"`
	GroupSize      int     `json:"group_size"`
	VisibleDevices *string `json:"visible_devices,omitempty"`
}
