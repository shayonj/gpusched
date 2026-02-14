package gpu

import (
	"testing"
)

func TestHostMemInfo(t *testing.T) {
	total, free := HostMemInfo()
	if total < 0 || free < 0 {
		t.Fatalf("negative memory: total=%d, free=%d", total, free)
	}
	t.Logf("host memory: total=%dMB free=%dMB", total, free)
}

func TestDriverVersion(t *testing.T) {
	v := DriverVersion()
	t.Logf("driver version: %q", v)
}

func TestProcessGPUMemNonexistent(t *testing.T) {
	mem := ProcessGPUMem(999999)
	if mem != 0 {
		t.Fatalf("expected 0 for nonexistent PID, got %d", mem)
	}
}
