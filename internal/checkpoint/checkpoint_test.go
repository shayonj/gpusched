package checkpoint

import (
	"testing"
)

func TestNewCUDA(t *testing.T) {
	c := NewCUDA()
	if c == nil {
		t.Fatal("NewCUDA returned nil")
	}
	t.Logf("cuda-checkpoint available: %v, binary: %q", c.Available, c.Binary)
}

func TestNewCRIU(t *testing.T) {
	cr := NewCRIU()
	if cr == nil {
		t.Fatal("NewCRIU returned nil")
	}
	t.Logf("criu available: %v, binary: %q", cr.Available, cr.Binary)
}

func TestCUDAFreezeRequiresAvailable(t *testing.T) {
	c := &CUDA{Available: false}
	_, err := c.Freeze(99999)
	if err == nil {
		t.Fatal("expected error for unavailable cuda-checkpoint")
	}
}

func TestCUDAThawRequiresAvailable(t *testing.T) {
	c := &CUDA{Available: false}
	_, err := c.Thaw(99999)
	if err == nil {
		t.Fatal("expected error for unavailable cuda-checkpoint")
	}
}

func TestCRIUDumpRequiresAvailable(t *testing.T) {
	cr := &CRIU{Available: false}
	_, err := cr.Dump(99999, "/tmp")
	if err == nil {
		t.Fatal("expected error for unavailable criu")
	}
}

func TestCRIURestoreRequiresAvailable(t *testing.T) {
	cr := &CRIU{Available: false}
	_, _, err := cr.Restore("/tmp/nonexistent")
	if err == nil {
		t.Fatal("expected error for unavailable criu")
	}
}
