// Package client connects to the gpusched daemon via Unix socket.
package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"

	"gpusched/internal/daemon"
	"gpusched/internal/protocol"
)

type Client struct {
	sockPath string
}

func New(sockPath string) *Client {
	if sockPath == "" {
		sockPath = daemon.DefaultSocket
	}
	return &Client{sockPath: sockPath}
}

func (c *Client) Call(method string, params interface{}) (protocol.Response, error) {
	conn, err := net.Dial("unix", c.sockPath)
	if err != nil {
		return protocol.Response{}, fmt.Errorf(
			"cannot connect to daemon at %s — is 'gpusched daemon' running?\n  error: %w",
			c.sockPath, err,
		)
	}
	defer conn.Close()

	var rawParams json.RawMessage
	if params != nil {
		rawParams, err = json.Marshal(params)
		if err != nil {
			return protocol.Response{}, fmt.Errorf("marshaling params: %w", err)
		}
	}

	req := protocol.Request{Method: method, Params: rawParams}
	data, _ := json.Marshal(req)
	data = append(data, '\n')
	if _, err := conn.Write(data); err != nil {
		return protocol.Response{}, fmt.Errorf("writing request: %w", err)
	}

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	if !scanner.Scan() {
		return protocol.Response{}, fmt.Errorf("no response from daemon")
	}

	var resp protocol.Response
	if err := json.Unmarshal(scanner.Bytes(), &resp); err != nil {
		return protocol.Response{}, fmt.Errorf("decoding response: %w", err)
	}
	return resp, nil
}

// Subscribe opens a persistent connection for event streaming.
func (c *Client) Subscribe() (protocol.StatusResult, <-chan protocol.Event, func(), error) {
	conn, err := net.Dial("unix", c.sockPath)
	if err != nil {
		return protocol.StatusResult{}, nil, nil, fmt.Errorf(
			"cannot connect to daemon at %s: %w", c.sockPath, err,
		)
	}

	req := protocol.Request{Method: "subscribe"}
	data, _ := json.Marshal(req)
	data = append(data, '\n')
	if _, err := conn.Write(data); err != nil {
		conn.Close()
		return protocol.StatusResult{}, nil, nil, fmt.Errorf("sending subscribe: %w", err)
	}

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	if !scanner.Scan() {
		conn.Close()
		return protocol.StatusResult{}, nil, nil, fmt.Errorf("no initial status")
	}

	var initResp protocol.Response
	if err := json.Unmarshal(scanner.Bytes(), &initResp); err != nil {
		conn.Close()
		return protocol.StatusResult{}, nil, nil, fmt.Errorf("decoding initial status: %w", err)
	}

	var status protocol.StatusResult
	if initResp.OK {
		json.Unmarshal(initResp.Result, &status)
	}

	ch := make(chan protocol.Event, 64)
	go func() {
		defer close(ch)
		defer conn.Close()
		for scanner.Scan() {
			var event protocol.Event
			if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
				continue
			}
			ch <- event
		}
	}()

	cancel := func() { conn.Close() }
	return status, ch, cancel, nil
}

// Command holds a persistent connection for sending multiple requests.
type Command struct {
	conn    net.Conn
	scanner *bufio.Scanner
}

func (c *Client) OpenCommand() (*Command, error) {
	conn, err := net.Dial("unix", c.sockPath)
	if err != nil {
		return nil, fmt.Errorf("cannot connect to daemon at %s: %w", c.sockPath, err)
	}
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	return &Command{conn: conn, scanner: scanner}, nil
}

func (cmd *Command) Call(method string, params interface{}) (protocol.Response, error) {
	var rawParams json.RawMessage
	if params != nil {
		var err error
		rawParams, err = json.Marshal(params)
		if err != nil {
			return protocol.Response{}, err
		}
	}
	req := protocol.Request{Method: method, Params: rawParams}
	data, _ := json.Marshal(req)
	data = append(data, '\n')
	if _, err := cmd.conn.Write(data); err != nil {
		return protocol.Response{}, err
	}
	if !cmd.scanner.Scan() {
		return protocol.Response{}, fmt.Errorf("connection closed")
	}
	var resp protocol.Response
	json.Unmarshal(cmd.scanner.Bytes(), &resp)
	return resp, nil
}

func (cmd *Command) Close() {
	cmd.conn.Close()
}
