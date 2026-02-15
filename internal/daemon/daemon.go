// Package daemon implements the gpusched process manager core.
package daemon

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"gpusched/internal/checkpoint"
	"gpusched/internal/gpu"
	"gpusched/internal/protocol"
)

type Proc struct {
	Name    string
	PID     int
	State   protocol.ProcessState
	GPU     int
	MemMB   int64
	Started time.Time
	Cmd     *exec.Cmd
	LogPath string
	logFile *os.File
}

type Config struct {
	RAMBudgetMB int64
	LogDir      string
}

type Daemon struct {
	mu      sync.RWMutex
	procs   map[string]*Proc
	events  []protocol.Event
	metrics protocol.Metrics

	cuda *checkpoint.CUDA
	cfg  Config
	log  *log.Logger

	subs  []chan protocol.Event
	subMu sync.Mutex

	freezeTotalMs int64
	thawTotalMs   int64
}

func New(cfg Config) *Daemon {
	if cfg.LogDir == "" {
		cfg.LogDir = "/tmp/gpusched/logs"
	}
	if cfg.RAMBudgetMB == 0 {
		total, _ := gpu.HostMemInfo()
		if total > 0 {
			cfg.RAMBudgetMB = total * 80 / 100
		} else {
			cfg.RAMBudgetMB = 64 * 1024
		}
	}

	os.MkdirAll(cfg.LogDir, 0o755)

	cuda := checkpoint.NewCUDA()

	d := &Daemon{
		procs: make(map[string]*Proc),
		cuda:  cuda,
		cfg:   cfg,
		log:   log.New(os.Stderr, "[gpusched] ", log.LstdFlags|log.Lmsgprefix),
	}

	d.log.Printf("capabilities: cuda-checkpoint=%v", cuda.Available)
	d.log.Printf("config: ram_budget=%dMB", cfg.RAMBudgetMB)

	return d
}

func (d *Daemon) Run(params protocol.RunParams) (protocol.RunResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if _, exists := d.procs[params.Name]; exists {
		return protocol.RunResult{}, fmt.Errorf("process %q already exists", params.Name)
	}
	if len(params.Cmd) == 0 {
		return protocol.RunResult{}, fmt.Errorf("empty command")
	}

	logPath := filepath.Join(d.cfg.LogDir, params.Name+".log")
	logFile, err := os.Create(logPath)
	if err != nil {
		return protocol.RunResult{}, fmt.Errorf("creating log: %w", err)
	}

	cmd := exec.Command(params.Cmd[0], params.Cmd[1:]...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Dir = params.Dir

	env := os.Environ()
	env = append(env,
		"GPUSCHED_MANAGED=1",
		fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", params.GPU),
	)
	if os.Getuid() == 0 {
		env = appendPythonPath(env)
	}
	cmd.Env = env

	if err := cmd.Start(); err != nil {
		logFile.Close()
		return protocol.RunResult{}, fmt.Errorf("starting process: %w", err)
	}

	p := &Proc{
		Name:    params.Name,
		PID:     cmd.Process.Pid,
		State:   protocol.StateActive,
		GPU:     params.GPU,
		Started: time.Now(),
		Cmd:     cmd,
		LogPath: logPath,
		logFile: logFile,
	}
	d.procs[params.Name] = p
	d.metrics.ColdStarts++

	go d.monitorProcess(params.Name, cmd)

	go func() {
		for i := 0; i < 12; i++ {
			time.Sleep(5 * time.Second)
			d.mu.Lock()
			if p.State != protocol.StateActive {
				d.mu.Unlock()
				return
			}
			if mem := gpu.ProcessGPUMem(p.PID); mem > 0 {
				p.MemMB = mem
				d.mu.Unlock()
				return
			}
			d.mu.Unlock()
		}
	}()

	d.emit(protocol.Event{
		Type:    "run",
		Process: params.Name,
		Detail:  fmt.Sprintf("pid=%d gpu=%d cmd=%v", p.PID, params.GPU, params.Cmd),
	})

	d.log.Printf("RUN %s pid=%d gpu=%d cmd=%v", params.Name, p.PID, params.GPU, params.Cmd)
	return protocol.RunResult{Name: params.Name, PID: p.PID}, nil
}

func (d *Daemon) Freeze(name string) (protocol.FreezeResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	p, ok := d.procs[name]
	if !ok {
		return protocol.FreezeResult{}, fmt.Errorf("process %q not found", name)
	}
	if p.State != protocol.StateActive {
		return protocol.FreezeResult{}, fmt.Errorf("process %q is %s, not active", name, p.State)
	}
	if !d.cuda.Available {
		return protocol.FreezeResult{}, fmt.Errorf("cuda-checkpoint not available")
	}

	if mem := gpu.ProcessGPUMem(p.PID); mem > 0 {
		p.MemMB = mem
	}

	dur, err := d.cuda.Freeze(p.PID)
	if err != nil {
		return protocol.FreezeResult{}, fmt.Errorf("cuda freeze: %w", err)
	}

	syscall.Kill(p.PID, syscall.SIGSTOP)

	p.State = protocol.StateFrozen

	d.metrics.Freezes++
	d.freezeTotalMs += dur.Milliseconds()
	d.metrics.AvgFreezeMs = d.freezeTotalMs / int64(d.metrics.Freezes)

	d.emit(protocol.Event{
		Type:     "freeze",
		Process:  name,
		Duration: dur.Milliseconds(),
		Detail:   fmt.Sprintf("→ RAM (%d MB)", p.MemMB),
	})

	d.log.Printf("FREEZE %s pid=%d %dms %dMB → RAM", name, p.PID, dur.Milliseconds(), p.MemMB)
	return protocol.FreezeResult{
		Name:       name,
		DurationMs: dur.Milliseconds(),
		MemMB:      p.MemMB,
	}, nil
}

func (d *Daemon) Thaw(name string) (protocol.ThawResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	p, ok := d.procs[name]
	if !ok {
		return protocol.ThawResult{}, fmt.Errorf("process %q not found", name)
	}
	if p.State != protocol.StateFrozen {
		return protocol.ThawResult{}, fmt.Errorf("process %q is %s, not frozen", name, p.State)
	}

	syscall.Kill(p.PID, syscall.SIGCONT)

	dur, err := d.cuda.Thaw(p.PID)
	if err != nil {
		syscall.Kill(p.PID, syscall.SIGSTOP)
		return protocol.ThawResult{}, fmt.Errorf("cuda thaw: %w", err)
	}

	p.State = protocol.StateActive

	d.metrics.Thaws++
	d.thawTotalMs += dur.Milliseconds()
	d.metrics.AvgThawMs = d.thawTotalMs / int64(d.metrics.Thaws)

	d.emit(protocol.Event{
		Type:     "thaw",
		Process:  p.Name,
		Duration: dur.Milliseconds(),
		Detail:   fmt.Sprintf("← RAM (%d MB)", p.MemMB),
	})

	d.log.Printf("THAW %s pid=%d %dms ← RAM", p.Name, p.PID, dur.Milliseconds())
	return protocol.ThawResult{
		Name:       p.Name,
		DurationMs: dur.Milliseconds(),
		MemMB:      p.MemMB,
	}, nil
}

func (d *Daemon) Kill(name string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	p, ok := d.procs[name]
	if !ok {
		return fmt.Errorf("process %q not found", name)
	}

	if p.State == protocol.StateFrozen {
		syscall.Kill(p.PID, syscall.SIGCONT)
	}
	syscall.Kill(p.PID, syscall.SIGTERM)
	go func(pid int) {
		time.Sleep(3 * time.Second)
		syscall.Kill(pid, syscall.SIGKILL)
	}(p.PID)

	p.State = protocol.StateDead
	if p.logFile != nil {
		p.logFile.Close()
	}

	d.emit(protocol.Event{Type: "kill", Process: name})
	d.log.Printf("KILL %s pid=%d", name, p.PID)

	delete(d.procs, name)
	return nil
}

func (d *Daemon) Migrate(params protocol.MigrateParams) (protocol.MigrateResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	p, ok := d.procs[params.Name]
	if !ok {
		return protocol.MigrateResult{}, fmt.Errorf("process %q not found", params.Name)
	}
	if !d.cuda.Available {
		return protocol.MigrateResult{}, fmt.Errorf("cuda-checkpoint not available")
	}

	fromGPU := p.GPU

	if p.State == protocol.StateActive {
		if mem := gpu.ProcessGPUMem(p.PID); mem > 0 {
			p.MemMB = mem
		}
		if _, err := d.cuda.Freeze(p.PID); err != nil {
			return protocol.MigrateResult{}, fmt.Errorf("freeze for migrate: %w", err)
		}
		syscall.Kill(p.PID, syscall.SIGSTOP)
	}

	syscall.Kill(p.PID, syscall.SIGCONT)
	dur, err := d.cuda.RestoreOnDevice(p.PID, params.GPU)
	if err != nil {
		return protocol.MigrateResult{}, fmt.Errorf("restore on gpu %d: %w", params.GPU, err)
	}
	if _, err := d.cuda.Unlock(p.PID); err != nil {
		return protocol.MigrateResult{}, fmt.Errorf("unlock after migrate: %w", err)
	}

	p.State = protocol.StateActive
	p.GPU = params.GPU

	d.metrics.Migrations++
	d.emit(protocol.Event{
		Type:     "migrate",
		Process:  params.Name,
		Duration: dur.Milliseconds(),
		Detail:   fmt.Sprintf("GPU %d → GPU %d", fromGPU, params.GPU),
	})

	d.log.Printf("MIGRATE %s GPU %d → %d %dms", params.Name, fromGPU, params.GPU, dur.Milliseconds())
	return protocol.MigrateResult{
		Name:    params.Name,
		FromGPU: fromGPU,
		ToGPU:   params.GPU,
	}, nil
}

func (d *Daemon) Status() protocol.StatusResult {
	d.mu.RLock()
	defer d.mu.RUnlock()

	gpus, _ := gpu.QueryGPUs()
	totalRAM, freeRAM := gpu.HostMemInfo()

	var procs []protocol.ProcessInfo
	var snapshotsMB int64

	for _, p := range d.procs {
		if p.State == protocol.StateActive {
			if mem := gpu.ProcessGPUMem(p.PID); mem > 0 {
				p.MemMB = mem
			}
		}

		tier := protocol.TierGPU
		if p.State == protocol.StateFrozen {
			tier = protocol.TierRAM
			snapshotsMB += p.MemMB
		}

		procs = append(procs, protocol.ProcessInfo{
			Name:    p.Name,
			PID:     p.PID,
			State:   p.State,
			GPU:     p.GPU,
			MemMB:   p.MemMB,
			Age:     formatDuration(time.Since(p.Started)),
			Started: p.Started,
			Tier:    tier,
		})
	}

	sort.Slice(procs, func(i, j int) bool {
		order := map[protocol.ProcessState]int{
			protocol.StateActive: 0,
			protocol.StateFrozen: 1,
			protocol.StateDead:   2,
		}
		return order[procs[i].State] < order[procs[j].State]
	})

	recentEvents := d.events
	if len(recentEvents) > 20 {
		recentEvents = recentEvents[len(recentEvents)-20:]
	}

	return protocol.StatusResult{
		GPUs:      gpus,
		Processes: procs,
		Memory: protocol.MemoryInfo{
			HostRAMTotalMB:  totalRAM,
			HostRAMFreeMB:   freeRAM,
			HostRAMBudgetMB: d.cfg.RAMBudgetMB,
			SnapshotsMB:     snapshotsMB,
		},
		Metrics: d.metrics,
		Events:  recentEvents,
		Caps: protocol.Capabilities{
			CUDACheckpoint: d.cuda.Available,
			DriverVersion:  gpu.DriverVersion(),
		},
	}
}

func (d *Daemon) Logs(name string, lines int) (protocol.LogsResult, error) {
	d.mu.RLock()
	p, ok := d.procs[name]
	d.mu.RUnlock()

	if !ok {
		return protocol.LogsResult{}, fmt.Errorf("process %q not found", name)
	}

	data, err := os.ReadFile(p.LogPath)
	if err != nil {
		return protocol.LogsResult{}, fmt.Errorf("reading logs: %w", err)
	}

	allLines := splitLines(string(data))
	if lines > 0 && lines < len(allLines) {
		allLines = allLines[len(allLines)-lines:]
	}

	return protocol.LogsResult{Lines: allLines}, nil
}

func (d *Daemon) Subscribe() chan protocol.Event {
	ch := make(chan protocol.Event, 64)
	d.subMu.Lock()
	d.subs = append(d.subs, ch)
	d.subMu.Unlock()
	return ch
}

func (d *Daemon) Unsubscribe(ch chan protocol.Event) {
	d.subMu.Lock()
	defer d.subMu.Unlock()
	for i, s := range d.subs {
		if s == ch {
			d.subs = append(d.subs[:i], d.subs[i+1:]...)
			close(ch)
			return
		}
	}
}

func (d *Daemon) Handle(req protocol.Request) protocol.Response {
	d.metrics.Requests++

	switch req.Method {
	case "run":
		var p protocol.RunParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		res, err := d.Run(p)
		if err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse(res)

	case "freeze":
		var p protocol.NameParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		res, err := d.Freeze(p.Name)
		if err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse(res)

	case "thaw":
		var p protocol.NameParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		res, err := d.Thaw(p.Name)
		if err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse(res)

	case "kill":
		var p protocol.NameParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		if err := d.Kill(p.Name); err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse("ok")

	case "migrate":
		var p protocol.MigrateParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		res, err := d.Migrate(p)
		if err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse(res)

	case "status":
		return protocol.OkResponse(d.Status())

	case "logs":
		var p protocol.LogsParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		if p.Lines == 0 {
			p.Lines = 50
		}
		res, err := d.Logs(p.Name, p.Lines)
		if err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse(res)

	default:
		return protocol.ErrResponse("unknown method: " + req.Method)
	}
}

func (d *Daemon) Shutdown() {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.log.Println("shutting down — cleaning up processes")
	for name, p := range d.procs {
		switch p.State {
		case protocol.StateActive:
			d.log.Printf("  killing active process %s (pid=%d)", name, p.PID)
			syscall.Kill(p.PID, syscall.SIGTERM)
		case protocol.StateFrozen:
			d.log.Printf("  killing frozen process %s (pid=%d)", name, p.PID)
			syscall.Kill(p.PID, syscall.SIGCONT)
			syscall.Kill(p.PID, syscall.SIGTERM)
		}
		if p.logFile != nil {
			p.logFile.Close()
		}
	}

	d.subMu.Lock()
	for _, ch := range d.subs {
		close(ch)
	}
	d.subs = nil
	d.subMu.Unlock()
}

func (d *Daemon) emit(e protocol.Event) {
	e.Time = time.Now()
	d.events = append(d.events, e)

	if len(d.events) > 1000 {
		d.events = d.events[len(d.events)-500:]
	}

	d.subMu.Lock()
	for _, ch := range d.subs {
		select {
		case ch <- e:
		default:
		}
	}
	d.subMu.Unlock()
}

func (d *Daemon) monitorProcess(name string, cmd *exec.Cmd) {
	err := cmd.Wait()
	d.mu.Lock()
	defer d.mu.Unlock()

	p, ok := d.procs[name]
	if !ok {
		return
	}
	if p.State == protocol.StateDead {
		return
	}

	detail := "exited"
	if err != nil {
		detail = err.Error()
	}

	p.State = protocol.StateDead
	if p.logFile != nil {
		p.logFile.Close()
	}

	d.emit(protocol.Event{Type: "exit", Process: name, Detail: detail})
	d.log.Printf("EXIT %s pid=%d: %s", name, p.PID, detail)
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	if d < 24*time.Hour {
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
	days := int(d.Hours()) / 24
	hours := int(d.Hours()) % 24
	return fmt.Sprintf("%dd%dh", days, hours)
}

func appendPythonPath(env []string) []string {
	var extraPaths []string
	entries, _ := os.ReadDir("/home")
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		globPattern := "/home/" + e.Name() + "/.local/lib/python*/site-packages"
		matches, _ := filepath.Glob(globPattern)
		extraPaths = append(extraPaths, matches...)
	}
	if len(extraPaths) == 0 {
		return env
	}

	extra := strings.Join(extraPaths, ":")
	for i, e := range env {
		if strings.HasPrefix(e, "PYTHONPATH=") {
			env[i] = e + ":" + extra
			return env
		}
	}
	return append(env, "PYTHONPATH="+extra)
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}
