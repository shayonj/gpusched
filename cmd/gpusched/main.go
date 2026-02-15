// gpusched — GPU Process Manager
// Freeze, thaw, and migrate any CUDA process.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"gpusched/internal/client"
	"gpusched/internal/daemon"
	"gpusched/internal/protocol"
	"gpusched/internal/tui"

	"github.com/spf13/cobra"
)

var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

var sockPath string

func main() {
	root := &cobra.Command{
		Use:     "gpusched",
		Short:   "GPU Process Manager — systemd for GPU processes",
		Version: version,
	}

	root.PersistentFlags().StringVarP(&sockPath, "socket", "s", daemon.DefaultSocket, "daemon socket path")

	root.AddCommand(
		daemonCmd(),
		runCmd(),
		freezeCmd(),
		thawCmd(),
		killCmd(),
		statusCmd(),
		logsCmd(),
		migrateCmd(),
		dashboardCmd(),
	)

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// ── daemon ──────────────────────────────────────────────────────────────────

func daemonCmd() *cobra.Command {
	var ramBudget string
	var logDir string

	cmd := &cobra.Command{
		Use:   "daemon",
		Short: "Start the gpusched daemon (run as root for cuda-checkpoint)",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg := daemon.Config{
				RAMBudgetMB: parseMB(ramBudget),
				LogDir:      logDir,
			}

			d := daemon.New(cfg)
			srv := daemon.NewServer(d, sockPath)
			defer srv.Cleanup()

			fmt.Fprintf(os.Stderr, "gpusched v%s — GPU Process Manager\n", version)
			fmt.Fprintf(os.Stderr, "listening on %s\n", sockPath)
			if os.Getuid() != 0 {
				fmt.Fprintf(os.Stderr, "WARNING: not running as root — cuda-checkpoint may fail\n")
			}

			return srv.ListenAndServe()
		},
	}

	cmd.Flags().StringVar(&ramBudget, "ram-budget", "", "max host RAM for snapshots (e.g. 80G, 80000M)")
	cmd.Flags().StringVar(&logDir, "log-dir", "/tmp/gpusched/logs", "process log directory")

	return cmd
}

// ── run ─────────────────────────────────────────────────────────────────────

func runCmd() *cobra.Command {
	var name string
	var gpuID int
	var dir string

	cmd := &cobra.Command{
		Use:   "run [flags] -- COMMAND [ARGS...]",
		Short: "Spawn a managed GPU process",
		Example: `  gpusched run --name train -- python train.py
  gpusched run --name eval --gpu 1 -- python eval.py`,
		Args:               cobra.MinimumNArgs(1),
		DisableFlagParsing: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			if name == "" {
				name = args[0]
			}

			c := client.New(sockPath)
			resp, err := c.Call("run", protocol.RunParams{
				Name: name,
				Cmd:  args,
				Dir:  dir,
				GPU:  gpuID,
			})
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}

			var result protocol.RunResult
			json.Unmarshal(resp.Result, &result)
			fmt.Printf("Started %s (pid=%d)\n", result.Name, result.PID)
			return nil
		},
	}

	cmd.Flags().StringVarP(&name, "name", "n", "", "process name (default: command name)")
	cmd.Flags().IntVarP(&gpuID, "gpu", "g", 0, "GPU device index")
	cmd.Flags().StringVarP(&dir, "dir", "d", "", "working directory")

	return cmd
}

// ── freeze ──────────────────────────────────────────────────────────────────

func freezeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "freeze NAME",
		Short: "Checkpoint a process to host RAM (frees GPU)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			resp, err := c.Call("freeze", protocol.NameParams{Name: args[0]})
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}

			var result protocol.FreezeResult
			json.Unmarshal(resp.Result, &result)
			fmt.Printf("Frozen %s → ram (%d ms)\n", result.Name, result.DurationMs)
			return nil
		},
	}
}

// ── thaw ────────────────────────────────────────────────────────────────────

func thawCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "thaw NAME",
		Short: "Restore a frozen process (reclaims GPU)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			resp, err := c.Call("thaw", protocol.NameParams{Name: args[0]})
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}

			var result protocol.ThawResult
			json.Unmarshal(resp.Result, &result)
			fmt.Printf("Thawed %s ← ram (%d ms)\n", result.Name, result.DurationMs)
			return nil
		},
	}
}

// ── kill ────────────────────────────────────────────────────────────────────

func killCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "kill NAME",
		Short: "Terminate a managed process",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			resp, err := c.Call("kill", protocol.NameParams{Name: args[0]})
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}
			fmt.Printf("Killed %s\n", args[0])
			return nil
		},
	}
}

// ── status ──────────────────────────────────────────────────────────────────

func statusCmd() *cobra.Command {
	var jsonOut bool

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show all processes, GPU usage, and snapshots",
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			resp, err := c.Call("status", nil)
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}

			if jsonOut {
				fmt.Println(string(resp.Result))
				return nil
			}

			var s protocol.StatusResult
			json.Unmarshal(resp.Result, &s)
			printStatus(s)
			return nil
		},
	}

	cmd.Flags().BoolVar(&jsonOut, "json", false, "output as JSON")
	return cmd
}

func printStatus(s protocol.StatusResult) {
	for _, g := range s.GPUs {
		pct := float64(g.MemUsed) / float64(g.MemTotal) * 100
		fmt.Printf("GPU %d: %s (%d / %d MB, %.0f%%)\n", g.Index, g.Name, g.MemUsed, g.MemTotal, pct)
	}

	var active, frozen []protocol.ProcessInfo
	for _, p := range s.Processes {
		switch p.State {
		case protocol.StateActive:
			active = append(active, p)
		case protocol.StateFrozen:
			frozen = append(frozen, p)
		}
	}

	if len(active) > 0 {
		fmt.Println()
		for _, p := range active {
			fmt.Printf("  ● %-16s active      %6d MB  %s\n", p.Name, p.MemMB, p.Age)
		}
	}

	if len(frozen) > 0 {
		fmt.Printf("\nSnapshots (host RAM: %d / %d MB):\n", s.Memory.SnapshotsMB, s.Memory.HostRAMBudgetMB)
		for _, p := range frozen {
			fmt.Printf("  ○ %-16s frozen      %6d MB  %s\n", p.Name, p.MemMB, p.Age)
		}
	}

	if len(s.Processes) == 0 {
		fmt.Println("\n  (no managed processes)")
	}

	m := s.Metrics
	if m.Requests > 0 {
		fmt.Printf("\nMetrics: %d req | %d freezes | %d thaws | avg freeze %dms | avg thaw %dms\n",
			m.Requests, m.Freezes, m.Thaws, m.AvgFreezeMs, m.AvgThawMs)
	}

	fmt.Printf("\nCapabilities: cuda-checkpoint=%v  driver=%s\n",
		s.Caps.CUDACheckpoint, s.Caps.DriverVersion)
}

// ── logs ────────────────────────────────────────────────────────────────────

func logsCmd() *cobra.Command {
	var lines int

	cmd := &cobra.Command{
		Use:   "logs NAME",
		Short: "View process stdout/stderr",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			resp, err := c.Call("logs", protocol.LogsParams{
				Name:  args[0],
				Lines: lines,
			})
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}

			var result protocol.LogsResult
			json.Unmarshal(resp.Result, &result)
			for _, line := range result.Lines {
				fmt.Println(line)
			}
			return nil
		},
	}

	cmd.Flags().IntVarP(&lines, "lines", "n", 50, "number of lines")
	return cmd
}

// ── migrate ─────────────────────────────────────────────────────────────────

func migrateCmd() *cobra.Command {
	var gpuID int

	cmd := &cobra.Command{
		Use:     "migrate NAME",
		Short:   "Move a process to a different GPU",
		Example: "  gpusched migrate train --to 1",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			resp, err := c.Call("migrate", protocol.MigrateParams{
				Name: args[0],
				GPU:  gpuID,
			})
			if err != nil {
				return err
			}
			if !resp.OK {
				return fmt.Errorf("%s", resp.Error)
			}

			var result protocol.MigrateResult
			json.Unmarshal(resp.Result, &result)
			fmt.Printf("Migrated %s: GPU %d → GPU %d\n", result.Name, result.FromGPU, result.ToGPU)
			return nil
		},
	}

	cmd.Flags().IntVar(&gpuID, "to", 0, "target GPU device index")
	cmd.MarkFlagRequired("to")

	return cmd
}

// ── dashboard ───────────────────────────────────────────────────────────────

func dashboardCmd() *cobra.Command {
	return &cobra.Command{
		Use:     "dashboard",
		Aliases: []string{"dash", "tui"},
		Short:   "Interactive terminal dashboard",
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client.New(sockPath)
			return tui.Run(c)
		},
	}
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// parseMB converts strings like "80G", "80000M", "80000" to MB.
func parseMB(s string) int64 {
	if s == "" {
		return 0
	}
	s = strings.TrimSpace(s)
	multiplier := int64(1)
	if strings.HasSuffix(s, "G") || strings.HasSuffix(s, "g") {
		multiplier = 1024
		s = s[:len(s)-1]
	} else if strings.HasSuffix(s, "M") || strings.HasSuffix(s, "m") {
		s = s[:len(s)-1]
	} else if strings.HasSuffix(s, "T") || strings.HasSuffix(s, "t") {
		multiplier = 1024 * 1024
		s = s[:len(s)-1]
	}
	v, _ := strconv.ParseInt(s, 10, 64)
	return v * multiplier
}
