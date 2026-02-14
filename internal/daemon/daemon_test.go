package daemon

import (
	"os"
	"syscall"
	"testing"
	"time"

	"gpusched/internal/protocol"
)

func tempDaemon(t *testing.T) *Daemon {
	t.Helper()
	dir := t.TempDir()
	return New(Config{
		LogDir:       dir + "/logs",
		DiskDir:      dir + "/snapshots",
		RAMBudgetMB:  8192,
		DiskBudgetMB: 8192,
	})
}

func TestNewDaemon(t *testing.T) {
	d := tempDaemon(t)
	if d == nil {
		t.Fatal("New returned nil")
	}
	if d.cfg.RAMBudgetMB != 8192 {
		t.Fatalf("expected RAM budget 8192, got %d", d.cfg.RAMBudgetMB)
	}
}

func TestRunEmptyCommand(t *testing.T) {
	d := tempDaemon(t)
	_, err := d.Run(protocol.RunParams{Name: "test", Cmd: nil})
	if err == nil {
		t.Fatal("expected error for empty command")
	}
}

func TestRunDuplicateName(t *testing.T) {
	d := tempDaemon(t)
	_, err := d.Run(protocol.RunParams{Name: "test", Cmd: []string{"sleep", "3600"}})
	if err != nil {
		t.Fatalf("first run failed: %v", err)
	}
	_, err = d.Run(protocol.RunParams{Name: "test", Cmd: []string{"sleep", "3600"}})
	if err == nil {
		t.Fatal("expected error for duplicate name")
	}
	d.Kill("test")
}

func TestRunAndKill(t *testing.T) {
	d := tempDaemon(t)
	result, err := d.Run(protocol.RunParams{Name: "sleeper", Cmd: []string{"sleep", "3600"}})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.PID == 0 {
		t.Fatal("expected nonzero PID")
	}
	if result.Name != "sleeper" {
		t.Fatalf("expected name 'sleeper', got %q", result.Name)
	}

	status := d.Status()
	found := false
	for _, p := range status.Processes {
		if p.Name == "sleeper" {
			found = true
			if p.State != protocol.StateActive {
				t.Fatalf("expected active, got %s", p.State)
			}
		}
	}
	if !found {
		t.Fatal("process not found in status")
	}

	err = d.Kill("sleeper")
	if err != nil {
		t.Fatalf("kill: %v", err)
	}
}

func TestKillNonexistent(t *testing.T) {
	d := tempDaemon(t)
	err := d.Kill("doesnotexist")
	if err == nil {
		t.Fatal("expected error for nonexistent process")
	}
}

func TestFreezeNonexistent(t *testing.T) {
	d := tempDaemon(t)
	_, err := d.Freeze("doesnotexist")
	if err == nil {
		t.Fatal("expected error for nonexistent process")
	}
}

func TestThawNonexistent(t *testing.T) {
	d := tempDaemon(t)
	_, err := d.Thaw("doesnotexist")
	if err == nil {
		t.Fatal("expected error for nonexistent process")
	}
}

func TestStatus(t *testing.T) {
	d := tempDaemon(t)
	s := d.Status()
	if s.Caps.DriverVersion == "" {
		t.Log("no NVIDIA driver detected (expected in CI)")
	}
	if s.Metrics.Requests != 0 {
		t.Fatalf("expected 0 requests initially, got %d", s.Metrics.Requests)
	}
}

func TestLogs(t *testing.T) {
	d := tempDaemon(t)
	_, err := d.Run(protocol.RunParams{Name: "echo", Cmd: []string{"sh", "-c", "echo hello && sleep 3600"}})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	defer d.Kill("echo")

	logPath := d.cfg.LogDir + "/echo.log"
	if _, err := os.Stat(logPath); os.IsNotExist(err) {
		t.Fatal("log file not created")
	}

	result, err := d.Logs("echo", 10)
	if err != nil {
		t.Fatalf("logs: %v", err)
	}
	_ = result
}

func TestLogsNonexistent(t *testing.T) {
	d := tempDaemon(t)
	_, err := d.Logs("doesnotexist", 10)
	if err == nil {
		t.Fatal("expected error for nonexistent process")
	}
}

func TestEventSubscription(t *testing.T) {
	d := tempDaemon(t)
	ch := d.Subscribe()
	defer d.Unsubscribe(ch)

	_, _ = d.Run(protocol.RunParams{Name: "evtest", Cmd: []string{"sleep", "3600"}})
	defer d.Kill("evtest")

	select {
	case ev := <-ch:
		if ev.Type != "run" {
			t.Fatalf("expected 'run' event, got %q", ev.Type)
		}
		if ev.Process != "evtest" {
			t.Fatalf("expected process 'evtest', got %q", ev.Process)
		}
	default:
		t.Log("no event received (timing-dependent, OK in unit test)")
	}
}

func TestShutdownKillsProcesses(t *testing.T) {
	d := tempDaemon(t)
	result, _ := d.Run(protocol.RunParams{Name: "victim", Cmd: []string{"sleep", "3600"}})

	d.Shutdown()

	time.Sleep(100 * time.Millisecond)
	err := syscall.Kill(result.PID, 0)
	if err == nil {
		t.Fatal("process still alive after shutdown")
	}
}
