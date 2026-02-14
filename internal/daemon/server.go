package daemon

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"gpusched/internal/protocol"
)

const DefaultSocket = "/tmp/gpusched.sock"

type Server struct {
	daemon   *Daemon
	sockPath string
	listener net.Listener
	wg       sync.WaitGroup
}

func NewServer(d *Daemon, sockPath string) *Server {
	if sockPath == "" {
		sockPath = DefaultSocket
	}
	return &Server{daemon: d, sockPath: sockPath}
}

func (s *Server) ListenAndServe() error {
	os.Remove(s.sockPath)

	ln, err := net.Listen("unix", s.sockPath)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.sockPath, err)
	}
	s.listener = ln
	os.Chmod(s.sockPath, 0o666)

	s.daemon.log.Printf("listening on %s", s.sockPath)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		s.daemon.log.Printf("received %s — shutting down", sig)
		s.daemon.Shutdown()
		ln.Close()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			return nil
		}
		s.wg.Add(1)
		go s.handleConn(conn)
	}
}

func (s *Server) handleConn(conn net.Conn) {
	defer s.wg.Done()
	defer conn.Close()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var req protocol.Request
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			writeJSON(conn, protocol.ErrResponse("invalid json: "+err.Error()))
			continue
		}

		if req.Method == "subscribe" {
			s.handleSubscribe(conn)
			return
		}

		resp := s.daemon.Handle(req)
		if err := writeJSON(conn, resp); err != nil {
			return
		}
	}
}

func (s *Server) handleSubscribe(conn net.Conn) {
	ch := s.daemon.Subscribe()
	defer s.daemon.Unsubscribe(ch)

	status := s.daemon.Status()
	writeJSON(conn, protocol.OkResponse(status))

	for event := range ch {
		if err := writeJSON(conn, event); err != nil {
			return
		}
	}
}

func writeJSON(conn net.Conn, v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = conn.Write(data)
	return err
}

func (s *Server) Cleanup() {
	if s.sockPath != "" {
		os.Remove(s.sockPath)
	}
}
