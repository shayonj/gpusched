package protocol

import (
	"encoding/json"
	"testing"
)

func TestRequestMarshal(t *testing.T) {
	r := Request{Method: "freeze", Params: json.RawMessage(`{"name":"model-a"}`)}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var r2 Request
	if err := json.Unmarshal(b, &r2); err != nil {
		t.Fatal(err)
	}
	if r2.Method != "freeze" {
		t.Fatalf("expected freeze, got %s", r2.Method)
	}
}

func TestResponseOK(t *testing.T) {
	r := Response{OK: true, Result: json.RawMessage(`{"name":"model-a"}`)}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var r2 Response
	json.Unmarshal(b, &r2)
	if !r2.OK {
		t.Fatal("expected OK")
	}
}

func TestResponseError(t *testing.T) {
	r := Response{OK: false, Error: "not found"}
	b, _ := json.Marshal(r)
	var r2 Response
	json.Unmarshal(b, &r2)
	if r2.OK {
		t.Fatal("expected not OK")
	}
	if r2.Error != "not found" {
		t.Fatalf("expected 'not found', got %q", r2.Error)
	}
}

func TestProcessStates(t *testing.T) {
	states := []ProcessState{StateActive, StateFrozen, StateHibernated, StateDead}
	for _, s := range states {
		if s == "" {
			t.Fatal("empty state")
		}
	}
}

func TestTiers(t *testing.T) {
	tiers := []Tier{TierGPU, TierRAM, TierDisk}
	for _, ti := range tiers {
		if ti == "" {
			t.Fatal("empty tier")
		}
	}
}

func TestStatusResultRoundtrip(t *testing.T) {
	s := StatusResult{
		GPUs: []GPUInfo{{Index: 0, Name: "H100", MemTotal: 81559, MemUsed: 5000, MemFree: 76559}},
		Processes: []ProcessInfo{
			{Name: "model-a", PID: 1234, State: StateActive, GPU: 0, MemMB: 1500, Age: "5m", Tier: TierGPU},
		},
		Memory:  MemoryInfo{SnapshotsMB: 0, HostRAMBudgetMB: 100000},
		Metrics: Metrics{Requests: 10, Freezes: 3, Thaws: 2, AvgFreezeMs: 600, AvgThawMs: 400},
		Caps:    Capabilities{CUDACheckpoint: true, CRIU: true, DriverVersion: "580.126.09"},
	}
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var s2 StatusResult
	if err := json.Unmarshal(b, &s2); err != nil {
		t.Fatal(err)
	}
	if len(s2.GPUs) != 1 || s2.GPUs[0].Name != "H100" {
		t.Fatal("GPU roundtrip failed")
	}
	if len(s2.Processes) != 1 || s2.Processes[0].Name != "model-a" {
		t.Fatal("process roundtrip failed")
	}
	if !s2.Caps.CUDACheckpoint {
		t.Fatal("capabilities roundtrip failed")
	}
}

func TestEventMarshal(t *testing.T) {
	e := Event{Type: "freeze", Process: "train", Duration: 609, Detail: "→ ram"}
	b, _ := json.Marshal(e)
	var e2 Event
	json.Unmarshal(b, &e2)
	if e2.Type != "freeze" || e2.Duration != 609 {
		t.Fatalf("event roundtrip: %+v", e2)
	}
}

func TestRunParamsMarshal(t *testing.T) {
	p := RunParams{Name: "train", Cmd: []string{"python3", "train.py"}, GPU: 0}
	b, _ := json.Marshal(p)
	var p2 RunParams
	json.Unmarshal(b, &p2)
	if p2.Name != "train" || len(p2.Cmd) != 2 {
		t.Fatalf("runparams roundtrip: %+v", p2)
	}
}
