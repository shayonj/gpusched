// Package checkpoint wraps the cuda-checkpoint and CRIU command-line tools.
package checkpoint

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type CUDA struct {
	Binary    string
	Available bool
}

func NewCUDA() *CUDA {
	path, err := exec.LookPath("cuda-checkpoint")
	if err != nil {
		for _, p := range []string{
			"/usr/bin/cuda-checkpoint",
			"/usr/local/bin/cuda-checkpoint",
			"/usr/lib/nvidia/bin/cuda-checkpoint",
		} {
			if _, err := os.Stat(p); err == nil {
				return &CUDA{Binary: p, Available: true}
			}
		}
		return &CUDA{Available: false}
	}
	return &CUDA{Binary: path, Available: true}
}

func (c *CUDA) Lock(pid int) (time.Duration, error) {
	return c.run("lock", pid)
}

func (c *CUDA) Checkpoint(pid int) (time.Duration, error) {
	return c.run("checkpoint", pid)
}

func (c *CUDA) Restore(pid int) (time.Duration, error) {
	return c.run("restore", pid)
}

func (c *CUDA) Unlock(pid int) (time.Duration, error) {
	return c.run("unlock", pid)
}

func (c *CUDA) RestoreOnDevice(pid, device int) (time.Duration, error) {
	if !c.Available {
		return 0, fmt.Errorf("cuda-checkpoint not available")
	}
	return c.exec("restore", pid, "--device", strconv.Itoa(device))
}

// Freeze performs the full lock→checkpoint sequence.
func (c *CUDA) Freeze(pid int) (time.Duration, error) {
	lockDur, err := c.Lock(pid)
	if err != nil {
		return lockDur, fmt.Errorf("lock: %w", err)
	}
	ckptDur, err := c.Checkpoint(pid)
	total := lockDur + ckptDur
	if err != nil {
		c.Unlock(pid) //nolint:errcheck
		return total, fmt.Errorf("checkpoint: %w", err)
	}
	return total, nil
}

// Thaw performs the full restore→unlock sequence.
func (c *CUDA) Thaw(pid int) (time.Duration, error) {
	restDur, err := c.Restore(pid)
	if err != nil {
		return restDur, fmt.Errorf("restore: %w", err)
	}
	unlDur, err := c.Unlock(pid)
	total := restDur + unlDur
	if err != nil {
		return total, fmt.Errorf("unlock: %w", err)
	}
	return total, nil
}

func (c *CUDA) run(action string, pid int) (time.Duration, error) {
	if !c.Available {
		return 0, fmt.Errorf("cuda-checkpoint not available")
	}
	return c.exec(action, pid)
}

func (c *CUDA) exec(action string, pid int, extra ...string) (time.Duration, error) {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	args = append(args, extra...)

	start := time.Now()
	cmd := exec.Command(c.Binary, args...)
	out, err := cmd.CombinedOutput()
	elapsed := time.Since(start)
	if err != nil {
		return elapsed, fmt.Errorf("cuda-checkpoint --%s pid=%d: %s (%w)",
			action, pid, strings.TrimSpace(string(out)), err)
	}
	return elapsed, nil
}

type CRIU struct {
	Binary    string
	Available bool
}

func NewCRIU() *CRIU {
	path, err := exec.LookPath("criu")
	if err != nil {
		return &CRIU{Available: false}
	}
	cmd := exec.Command(path, "check")
	if err := cmd.Run(); err != nil {
		return &CRIU{Binary: path, Available: false}
	}
	return &CRIU{Binary: path, Available: true}
}

func (cr *CRIU) Dump(pid int, dir string) (time.Duration, error) {
	if !cr.Available {
		return 0, fmt.Errorf("criu not available (install: apt install criu)")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, fmt.Errorf("creating dump dir: %w", err)
	}

	start := time.Now()
	cmd := exec.Command(cr.Binary, "dump",
		"-t", strconv.Itoa(pid),
		"-D", dir,
		"--shell-job",
		"--tcp-established",
		"-v0",
	)
	out, err := cmd.CombinedOutput()
	elapsed := time.Since(start)
	if err != nil {
		return elapsed, fmt.Errorf("criu dump pid=%d: %s (%w)", pid, strings.TrimSpace(string(out)), err)
	}
	return elapsed, nil
}

func (cr *CRIU) Restore(dir string) (int, time.Duration, error) {
	if !cr.Available {
		return 0, 0, fmt.Errorf("criu not available")
	}
	pidFile := dir + "/restored.pid"

	start := time.Now()
	cmd := exec.Command(cr.Binary, "restore",
		"-D", dir,
		"--shell-job",
		"--tcp-established",
		"-d",
		"--pidfile", pidFile,
		"-v0",
	)
	out, err := cmd.CombinedOutput()
	elapsed := time.Since(start)
	if err != nil {
		return 0, elapsed, fmt.Errorf("criu restore: %s (%w)", strings.TrimSpace(string(out)), err)
	}

	pidBytes, err := os.ReadFile(pidFile)
	if err != nil {
		return 0, elapsed, fmt.Errorf("reading restored pid: %w", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(pidBytes)))
	if err != nil {
		return 0, elapsed, fmt.Errorf("parsing restored pid: %w", err)
	}
	return pid, elapsed, nil
}

func DumpDir(baseDir, name string) string {
	return baseDir + "/" + name
}
