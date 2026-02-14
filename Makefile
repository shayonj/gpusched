.PHONY: build build-linux clean test test-go test-python install fmt vet

BINARY  := gpusched
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo none)
DATE    := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS := -s -w -X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)

build:
	go build -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/gpusched

build-linux:
	GOOS=linux GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/gpusched

clean:
	rm -f $(BINARY)

test: test-go test-python

test-go:
	go test ./... -count=1

test-python:
	cd sdk/python && python3 -m pytest tests/ -v -m "not integration"

install: build
	cp $(BINARY) /usr/local/bin/

fmt:
	gofmt -w .

vet:
	go vet ./...
