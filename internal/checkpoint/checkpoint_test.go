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
