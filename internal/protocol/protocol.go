// Package protocol defines the wire types shared between daemon, client, and TUI.
package protocol

import (
	"encoding/json"
	"time"
)

type ProcessState string

const (
	StateActive     ProcessState = "active"
	StateFrozen     ProcessState = "frozen"
	StateHibernated ProcessState = "hibernated"
	StateDead       ProcessState = "dead"
)

type Tier string

const (
	TierGPU  Tier = "gpu"
	TierRAM  Tier = "ram"
	TierDisk Tier = "disk"
)

type Request struct {
	Method string          `json:"method"`
	Params json.RawMessage `json:"params,omitempty"`
}

type Response struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

type Event struct {
	Time     time.Time `json:"time"`
	Type     string    `json:"type"`
	Process  string    `json:"process,omitempty"`
	Detail   string    `json:"detail,omitempty"`
	Duration int64     `json:"duration_ms,omitempty"`
}

type RunParams struct {
	Name string   `json:"name"`
	Cmd  []string `json:"cmd"`
	Dir  string   `json:"dir,omitempty"`
	GPU  int      `json:"gpu"`
}

type NameParams struct {
	Name string `json:"name"`
}

type ForkParams struct {
	Name   string `json:"name"`
	Copies int    `json:"copies"`
	GPUs   []int  `json:"gpus,omitempty"`
}

type MigrateParams struct {
	Name string `json:"name"`
	GPU  int    `json:"gpu"`
}

type LogsParams struct {
	Name   string `json:"name"`
	Lines  int    `json:"lines"`
	Follow bool   `json:"follow"`
}

type StatusResult struct {
	GPUs      []GPUInfo     `json:"gpus"`
	Processes []ProcessInfo `json:"processes"`
	Memory    MemoryInfo    `json:"memory"`
	Metrics   Metrics       `json:"metrics"`
	Events    []Event       `json:"recent_events"`
	Caps      Capabilities  `json:"capabilities"`
}

type GPUInfo struct {
	Index    int    `json:"index"`
	Name     string `json:"name"`
	MemTotal int64  `json:"mem_total_mb"`
	MemUsed  int64  `json:"mem_used_mb"`
	MemFree  int64  `json:"mem_free_mb"`
}

type ProcessInfo struct {
	Name    string       `json:"name"`
	PID     int          `json:"pid"`
	State   ProcessState `json:"state"`
	GPU     int          `json:"gpu"`
	MemMB   int64        `json:"mem_mb"`
	Age     string       `json:"age"`
	Started time.Time    `json:"started"`
	Tier    Tier         `json:"tier"`
}

type MemoryInfo struct {
	HostRAMTotalMB  int64 `json:"host_ram_total_mb"`
	HostRAMFreeMB   int64 `json:"host_ram_free_mb"`
	HostRAMBudgetMB int64 `json:"host_ram_budget_mb"`
	SnapshotsMB     int64 `json:"snapshots_mb"`
	DiskUsedMB      int64 `json:"disk_used_mb"`
	DiskBudgetMB    int64 `json:"disk_budget_mb"`
}

type Metrics struct {
	Requests     int   `json:"requests"`
	CacheHits    int   `json:"cache_hits"`
	Freezes      int   `json:"freezes"`
	Thaws        int   `json:"thaws"`
	Forks        int   `json:"forks"`
	Migrations   int   `json:"migrations"`
	Hibernations int   `json:"hibernations"`
	ColdStarts   int   `json:"cold_starts"`
	AvgFreezeMs  int64 `json:"avg_freeze_ms"`
	AvgThawMs    int64 `json:"avg_thaw_ms"`
}

type Capabilities struct {
	CUDACheckpoint bool   `json:"cuda_checkpoint"`
	CRIU           bool   `json:"criu"`
	DriverVersion  string `json:"driver_version,omitempty"`
}

type RunResult struct {
	Name string `json:"name"`
	PID  int    `json:"pid"`
}

type FreezeResult struct {
	Name       string `json:"name"`
	DurationMs int64  `json:"duration_ms"`
	Tier       Tier   `json:"tier"`
	MemMB      int64  `json:"mem_mb"`
}

type ThawResult struct {
	Name       string `json:"name"`
	DurationMs int64  `json:"duration_ms"`
	FromTier   Tier   `json:"from_tier"`
	MemMB      int64  `json:"mem_mb"`
}

type ForkResult struct {
	Source string   `json:"source"`
	Copies []string `json:"copies"`
}

type MigrateResult struct {
	Name    string `json:"name"`
	FromGPU int    `json:"from_gpu"`
	ToGPU   int    `json:"to_gpu"`
}

type LogsResult struct {
	Lines []string `json:"lines"`
}

func OkResponse(result interface{}) Response {
	data, _ := json.Marshal(result)
	return Response{OK: true, Result: data}
}

func ErrResponse(msg string) Response {
	return Response{OK: false, Error: msg}
}
