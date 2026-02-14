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
	Name     string
	PID      int
	State    protocol.ProcessState
	GPU      int
	MemMB    int64
	Started  time.Time
	FrozenAt time.Time
	Cmd      *exec.Cmd
	LogPath  string
	CRIUDir  string
	logFile  *os.File
}

type Config struct {
	RAMBudgetMB  int64
	DiskBudgetMB int64
	DiskDir      string
	LogDir       string
}

type Daemon struct {
	mu      sync.RWMutex
	procs   map[string]*Proc
	events  []protocol.Event
	metrics protocol.Metrics

	cuda *checkpoint.CUDA
	criu *checkpoint.CRIU
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
	if cfg.DiskDir == "" {
		cfg.DiskDir = "/tmp/gpusched/snapshots"
	}
	if cfg.RAMBudgetMB == 0 {
		total, _ := gpu.HostMemInfo()
		if total > 0 {
			cfg.RAMBudgetMB = total * 80 / 100
		} else {
			cfg.RAMBudgetMB = 64 * 1024
		}
	}
	if cfg.DiskBudgetMB == 0 {
		cfg.DiskBudgetMB = 500 * 1024
	}

	os.MkdirAll(cfg.LogDir, 0o755)
	os.MkdirAll(cfg.DiskDir, 0o755)

	cuda := checkpoint.NewCUDA()
	criu := checkpoint.NewCRIU()

	d := &Daemon{
		procs: make(map[string]*Proc),
		cuda:  cuda,
		criu:  criu,
		cfg:   cfg,
		log:   log.New(os.Stderr, "[gpusched] ", log.LstdFlags|log.Lmsgprefix),
	}

	d.log.Printf("capabilities: cuda-checkpoint=%v criu=%v", cuda.Available, criu.Available)
	d.log.Printf("config: ram_budget=%dMB disk_budget=%dMB disk_dir=%s", cfg.RAMBudgetMB, cfg.DiskBudgetMB, cfg.DiskDir)

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

	if err := d.ensureRAMBudget(p.MemMB); err != nil {
		d.log.Printf("WARN: RAM pressure during freeze of %s: %v", name, err)
	}

	dur, err := d.cuda.Freeze(p.PID)
	if err != nil {
		return protocol.FreezeResult{}, fmt.Errorf("cuda freeze: %w", err)
	}

	syscall.Kill(p.PID, syscall.SIGSTOP)

	p.State = protocol.StateFrozen
	p.FrozenAt = time.Now()

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
		Tier:       protocol.TierRAM,
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

	switch p.State {
	case protocol.StateFrozen:
		return d.thawFromRAM(p)
	case protocol.StateHibernated:
		return d.thawFromDisk(p)
	default:
		return protocol.ThawResult{}, fmt.Errorf("process %q is %s, not frozen/hibernated", name, p.State)
	}
}

func (d *Daemon) thawFromRAM(p *Proc) (protocol.ThawResult, error) {
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
		FromTier:   protocol.TierRAM,
		MemMB:      p.MemMB,
	}, nil
}

func (d *Daemon) thawFromDisk(p *Proc) (protocol.ThawResult, error) {
	if !d.criu.Available {
		return protocol.ThawResult{}, fmt.Errorf("criu not available — can't restore from disk")
	}

	newPID, criuDur, err := d.criu.Restore(p.CRIUDir)
	if err != nil {
		return protocol.ThawResult{}, fmt.Errorf("criu restore: %w", err)
	}

	cudaDur, err := d.cuda.Thaw(newPID)
	if err != nil {
		syscall.Kill(newPID, syscall.SIGKILL)
		return protocol.ThawResult{}, fmt.Errorf("cuda thaw after criu: %w", err)
	}

	totalDur := criuDur + cudaDur
	p.PID = newPID
	p.State = protocol.StateActive
	p.CRIUDir = ""

	go d.monitorPID(p.Name, newPID)

	d.metrics.Thaws++
	d.thawTotalMs += totalDur.Milliseconds()
	d.metrics.AvgThawMs = d.thawTotalMs / int64(d.metrics.Thaws)

	d.emit(protocol.Event{
		Type:     "thaw",
		Process:  p.Name,
		Duration: totalDur.Milliseconds(),
		Detail:   fmt.Sprintf("← disk (criu=%dms cuda=%dms)", criuDur.Milliseconds(), cudaDur.Milliseconds()),
	})

	d.log.Printf("THAW %s pid=%d %dms ← disk", p.Name, newPID, totalDur.Milliseconds())
	return protocol.ThawResult{
		Name:       p.Name,
		DurationMs: totalDur.Milliseconds(),
		FromTier:   protocol.TierDisk,
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

	switch p.State {
	case protocol.StateActive, protocol.StateFrozen:
		if p.State == protocol.StateFrozen {
			syscall.Kill(p.PID, syscall.SIGCONT)
		}
		syscall.Kill(p.PID, syscall.SIGTERM)
		go func(pid int) {
			time.Sleep(3 * time.Second)
			syscall.Kill(pid, syscall.SIGKILL)
		}(p.PID)
	case protocol.StateHibernated:
		os.RemoveAll(p.CRIUDir)
	}

	p.State = protocol.StateDead
	if p.logFile != nil {
		p.logFile.Close()
	}

	d.emit(protocol.Event{Type: "kill", Process: name})
	d.log.Printf("KILL %s pid=%d", name, p.PID)

	delete(d.procs, name)
	return nil
}

func (d *Daemon) Fork(params protocol.ForkParams) (protocol.ForkResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.criu.Available {
		return protocol.ForkResult{}, fmt.Errorf("fork requires CRIU (install: apt install criu)")
	}

	p, ok := d.procs[params.Name]
	if !ok {
		return protocol.ForkResult{}, fmt.Errorf("process %q not found", params.Name)
	}

	if p.State == protocol.StateActive {
		d.mu.Unlock()
		if _, err := d.Freeze(params.Name); err != nil {
			d.mu.Lock()
			return protocol.ForkResult{}, fmt.Errorf("freezing source: %w", err)
		}
		d.mu.Lock()
	}

	dumpDir := checkpoint.DumpDir(d.cfg.DiskDir, params.Name+"-fork-source")
	if _, err := d.criu.Dump(p.PID, dumpDir); err != nil {
		return protocol.ForkResult{}, fmt.Errorf("criu dump for fork: %w", err)
	}

	var copies []string
	for i := 1; i <= params.Copies; i++ {
		copyName := fmt.Sprintf("%s-%d", params.Name, i)
		targetGPU := p.GPU
		if i-1 < len(params.GPUs) {
			targetGPU = params.GPUs[i-1]
		}

		newPID, _, err := d.criu.Restore(dumpDir)
		if err != nil {
			d.log.Printf("WARN: fork copy %d failed criu restore: %v", i, err)
			continue
		}

		if targetGPU != p.GPU {
			_, err = d.cuda.RestoreOnDevice(newPID, targetGPU)
		} else {
			_, err = d.cuda.Thaw(newPID)
		}
		if err != nil {
			syscall.Kill(newPID, syscall.SIGKILL)
			d.log.Printf("WARN: fork copy %d failed cuda restore: %v", i, err)
			continue
		}

		logPath := filepath.Join(d.cfg.LogDir, copyName+".log")
		logFile, _ := os.Create(logPath)

		d.procs[copyName] = &Proc{
			Name:    copyName,
			PID:     newPID,
			State:   protocol.StateActive,
			GPU:     targetGPU,
			MemMB:   p.MemMB,
			Started: time.Now(),
			LogPath: logPath,
			logFile: logFile,
		}
		copies = append(copies, copyName)

		go d.monitorPID(copyName, newPID)
	}

	d.metrics.Forks++
	d.emit(protocol.Event{
		Type:    "fork",
		Process: params.Name,
		Detail:  fmt.Sprintf("%d copies: %v", len(copies), copies),
	})

	d.log.Printf("FORK %s → %d copies: %v", params.Name, len(copies), copies)
	return protocol.ForkResult{Source: params.Name, Copies: copies}, nil
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

func (d *Daemon) Hibernate(name string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.criu.Available {
		return fmt.Errorf("hibernate requires CRIU (install: apt install criu)")
	}

	p, ok := d.procs[name]
	if !ok {
		return fmt.Errorf("process %q not found", name)
	}

	if p.State == protocol.StateActive {
		if mem := gpu.ProcessGPUMem(p.PID); mem > 0 {
			p.MemMB = mem
		}
		if _, err := d.cuda.Freeze(p.PID); err != nil {
			return fmt.Errorf("freeze for hibernate: %w", err)
		}
		syscall.Kill(p.PID, syscall.SIGSTOP)
		p.State = protocol.StateFrozen
	}

	if p.State != protocol.StateFrozen {
		return fmt.Errorf("process %q must be active or frozen to hibernate", name)
	}

	syscall.Kill(p.PID, syscall.SIGCONT)

	dumpDir := checkpoint.DumpDir(d.cfg.DiskDir, name)
	dur, err := d.criu.Dump(p.PID, dumpDir)
	if err != nil {
		return fmt.Errorf("criu dump: %w", err)
	}

	p.State = protocol.StateHibernated
	p.CRIUDir = dumpDir

	d.metrics.Hibernations++
	d.emit(protocol.Event{
		Type:     "hibernate",
		Process:  name,
		Duration: dur.Milliseconds(),
		Detail:   fmt.Sprintf("→ disk %s (%d MB)", dumpDir, p.MemMB),
	})

	d.log.Printf("HIBERNATE %s → %s %dms", name, dumpDir, dur.Milliseconds())
	return nil
}

func (d *Daemon) Status() protocol.StatusResult {
	d.mu.RLock()
	defer d.mu.RUnlock()

	gpus, _ := gpu.QueryGPUs()
	totalRAM, freeRAM := gpu.HostMemInfo()

	var procs []protocol.ProcessInfo
	var snapshotsMB int64
	var diskMB int64

	for _, p := range d.procs {
		if p.State == protocol.StateActive {
			if mem := gpu.ProcessGPUMem(p.PID); mem > 0 {
				p.MemMB = mem
			}
		}

		tier := protocol.TierGPU
		switch p.State {
		case protocol.StateFrozen:
			tier = protocol.TierRAM
			snapshotsMB += p.MemMB
		case protocol.StateHibernated:
			tier = protocol.TierDisk
			diskMB += p.MemMB
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
			protocol.StateActive:     0,
			protocol.StateFrozen:     1,
			protocol.StateHibernated: 2,
			protocol.StateDead:       3,
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
			DiskUsedMB:      diskMB,
			DiskBudgetMB:    d.cfg.DiskBudgetMB,
		},
		Metrics: d.metrics,
		Events:  recentEvents,
		Caps: protocol.Capabilities{
			CUDACheckpoint: d.cuda.Available,
			CRIU:           d.criu.Available,
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

	case "fork":
		var p protocol.ForkParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		res, err := d.Fork(p)
		if err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse(res)

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

	case "hibernate":
		var p protocol.NameParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return protocol.ErrResponse("bad params: " + err.Error())
		}
		if err := d.Hibernate(p.Name); err != nil {
			return protocol.ErrResponse(err.Error())
		}
		return protocol.OkResponse("ok")

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
		case protocol.StateHibernated:
			d.log.Printf("  hibernated process %s left on disk at %s", name, p.CRIUDir)
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

func (d *Daemon) ensureRAMBudget(sizeMB int64) error {
	_, freeRAM := gpu.HostMemInfo()
	safetyMB := int64(4096)

	if freeRAM-sizeMB > safetyMB {
		return nil
	}

	var frozen []*Proc
	for _, p := range d.procs {
		if p.State == protocol.StateFrozen {
			frozen = append(frozen, p)
		}
	}
	sort.Slice(frozen, func(i, j int) bool {
		return frozen[i].FrozenAt.Before(frozen[j].FrozenAt)
	})

	for _, victim := range frozen {
		if freeRAM-sizeMB > safetyMB {
			break
		}

		if d.criu.Available {
			d.log.Printf("RAM pressure: hibernating %s (%d MB) to disk", victim.Name, victim.MemMB)
			syscall.Kill(victim.PID, syscall.SIGCONT)
			dumpDir := checkpoint.DumpDir(d.cfg.DiskDir, victim.Name)
			if _, err := d.criu.Dump(victim.PID, dumpDir); err != nil {
				d.log.Printf("  WARN: criu dump failed: %v", err)
				continue
			}
			victim.State = protocol.StateHibernated
			victim.CRIUDir = dumpDir
			freeRAM += victim.MemMB
			d.metrics.Hibernations++
			d.emit(protocol.Event{
				Type:    "evict",
				Process: victim.Name,
				Detail:  fmt.Sprintf("RAM → disk (%d MB)", victim.MemMB),
			})
		} else {
			d.log.Printf("RAM pressure: killing frozen %s (no CRIU for disk tier)", victim.Name)
			syscall.Kill(victim.PID, syscall.SIGCONT)
			syscall.Kill(victim.PID, syscall.SIGKILL)
			victim.State = protocol.StateDead
			freeRAM += victim.MemMB
			d.emit(protocol.Event{
				Type:    "evict-kill",
				Process: victim.Name,
				Detail:  fmt.Sprintf("killed — no CRIU (%d MB freed)", victim.MemMB),
			})
		}
	}

	return nil
}

func (d *Daemon) monitorProcess(name string, cmd *exec.Cmd) {
	err := cmd.Wait()
	d.mu.Lock()
	defer d.mu.Unlock()

	p, ok := d.procs[name]
	if !ok {
		return
	}
	if p.State == protocol.StateDead || p.State == protocol.StateHibernated {
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

func (d *Daemon) monitorPID(name string, pid int) {
	for {
		if err := syscall.Kill(pid, 0); err != nil {
			d.mu.Lock()
			if p, ok := d.procs[name]; ok && p.PID == pid && p.State != protocol.StateDead && p.State != protocol.StateHibernated {
				p.State = protocol.StateDead
				d.emit(protocol.Event{Type: "exit", Process: name, Detail: "process gone"})
				d.log.Printf("EXIT %s pid=%d: process gone", name, pid)
			}
			d.mu.Unlock()
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
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
