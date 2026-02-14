// Package gpu queries GPU state via nvidia-smi.
package gpu

import (
	"bufio"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"gpusched/internal/protocol"
)

func QueryGPUs() ([]protocol.GPUInfo, error) {
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.used,memory.free",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}

	var gpus []protocol.GPUInfo
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ", ")
		if len(parts) < 5 {
			continue
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		total, _ := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
		used, _ := strconv.ParseInt(strings.TrimSpace(parts[3]), 10, 64)
		free, _ := strconv.ParseInt(strings.TrimSpace(parts[4]), 10, 64)

		gpus = append(gpus, protocol.GPUInfo{
			Index:    idx,
			Name:     strings.TrimSpace(parts[1]),
			MemTotal: total,
			MemUsed:  used,
			MemFree:  free,
		})
	}
	return gpus, nil
}

func ProcessGPUMem(pid int) int64 {
	cmd := exec.Command("nvidia-smi",
		"--query-compute-apps=pid,used_memory",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	pidStr := strconv.Itoa(pid)
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	for scanner.Scan() {
		parts := strings.Split(scanner.Text(), ", ")
		if len(parts) >= 2 && strings.TrimSpace(parts[0]) == pidStr {
			mem, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
			return mem
		}
	}
	return 0
}

func DriverVersion() string {
	cmd := exec.Command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func HostMemInfo() (totalMB, freeMB int64) {
	cmd := exec.Command("awk", "/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}", "/proc/meminfo")
	out, err := cmd.Output()
	if err != nil {
		return 0, 0
	}
	parts := strings.Fields(strings.TrimSpace(string(out)))
	if len(parts) >= 2 {
		t, _ := strconv.ParseInt(parts[0], 10, 64)
		a, _ := strconv.ParseInt(parts[1], 10, 64)
		return t / 1024, a / 1024
	}
	return 0, 0
}
