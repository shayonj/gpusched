// Package tui implements the interactive gpusched dashboard.
package tui

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"gpusched/internal/client"
	"gpusched/internal/protocol"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			Padding(0, 1)

	headerStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#7D56F4")).
			BorderStyle(lipgloss.NormalBorder()).
			BorderBottom(true).
			BorderForeground(lipgloss.Color("#444444"))

	activeStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#04B575"))
	frozenStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#3DAEE9"))
	hibStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#888888"))
	deadStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5555"))
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#666666"))
	warnStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFAA00"))
	boldStyle   = lipgloss.NewStyle().Bold(true)
	helpStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#626262"))

	barFull  = lipgloss.NewStyle().Foreground(lipgloss.Color("#04B575"))
	barEmpty = lipgloss.NewStyle().Foreground(lipgloss.Color("#333333"))
	barWarn  = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFAA00"))
	barCrit  = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5555"))

	logoLine1 = lipgloss.NewStyle().Foreground(lipgloss.Color("#61AFEF"))
	logoLine2 = lipgloss.NewStyle().Foreground(lipgloss.Color("#528BFF"))
	logoLine3 = lipgloss.NewStyle().Foreground(lipgloss.Color("#7D56F4"))
)

type Model struct {
	client   *client.Client
	status   protocol.StatusResult
	events   []protocol.Event
	eventCh  <-chan protocol.Event
	cancelFn func()
	cursor   int
	width    int
	height   int
	err      error
	cmdConn  *client.Command
}

func NewModel(c *client.Client) Model {
	return Model{client: c, width: 80, height: 24}
}

type eventMsg protocol.Event
type statusMsg protocol.StatusResult
type errMsg error
type tickMsg time.Time

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.subscribe(),
		m.tick(),
	)
}

func (m Model) subscribe() tea.Cmd {
	return func() tea.Msg {
		status, ch, cancel, err := m.client.Subscribe()
		if err != nil {
			return errMsg(err)
		}
		m.eventCh = ch
		m.cancelFn = cancel
		return initMsg{status: status, ch: ch, cancel: cancel}
	}
}

type initMsg struct {
	status protocol.StatusResult
	ch     <-chan protocol.Event
	cancel func()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func waitForEvent(ch <-chan protocol.Event) tea.Cmd {
	return func() tea.Msg {
		event, ok := <-ch
		if !ok {
			return errMsg(fmt.Errorf("event stream closed"))
		}
		return eventMsg(event)
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case initMsg:
		m.status = msg.status
		m.eventCh = msg.ch
		m.cancelFn = msg.cancel
		m.events = append(m.events, m.status.Events...)

		conn, err := m.client.OpenCommand()
		if err == nil {
			m.cmdConn = conn
		}

		return m, waitForEvent(m.eventCh)

	case eventMsg:
		event := protocol.Event(msg)
		m.events = append(m.events, event)
		if len(m.events) > 100 {
			m.events = m.events[len(m.events)-50:]
		}
		return m, waitForEvent(m.eventCh)

	case tickMsg:
		return m, tea.Batch(m.refreshStatus(), m.tick())

	case statusMsg:
		m.status = protocol.StatusResult(msg)
		return m, nil

	case errMsg:
		m.err = msg
		return m, nil

	case tea.KeyMsg:
		return m.handleKey(msg)
	}
	return m, nil
}

func (m Model) refreshStatus() tea.Cmd {
	return func() tea.Msg {
		resp, err := m.client.Call("status", nil)
		if err != nil {
			return errMsg(err)
		}
		if !resp.OK {
			return errMsg(fmt.Errorf("%s", resp.Error))
		}
		var s protocol.StatusResult
		json.Unmarshal(resp.Result, &s)
		return statusMsg(s)
	}
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q", "ctrl+c":
		if m.cancelFn != nil {
			m.cancelFn()
		}
		if m.cmdConn != nil {
			m.cmdConn.Close()
		}
		return m, tea.Quit

	case "up", "k":
		if m.cursor > 0 {
			m.cursor--
		}
	case "down", "j":
		if m.cursor < len(m.status.Processes)-1 {
			m.cursor++
		}

	case "f":
		return m, m.doAction("freeze")
	case "t":
		return m, m.doAction("thaw")
	case "x":
		return m, m.doAction("kill")
	}
	return m, nil
}

func (m Model) doAction(method string) tea.Cmd {
	if m.cmdConn == nil || len(m.status.Processes) == 0 {
		return nil
	}
	proc := m.status.Processes[m.cursor]
	return func() tea.Msg {
		resp, err := m.cmdConn.Call(method, protocol.NameParams{Name: proc.Name})
		if err != nil {
			return errMsg(err)
		}
		if !resp.OK {
			return errMsg(fmt.Errorf("%s", resp.Error))
		}
		resp2, _ := m.cmdConn.Call("status", nil)
		var s protocol.StatusResult
		if resp2.OK {
			json.Unmarshal(resp2.Result, &s)
		}
		return statusMsg(s)
	}
}

func (m Model) View() string {
	var b strings.Builder

	gpuName := "unknown GPU"
	gpuMem := ""
	if len(m.status.GPUs) > 0 {
		g := m.status.GPUs[0]
		gpuName = g.Name
		gpuMem = fmt.Sprintf("%d GB", g.MemTotal/1024)
	}

	logo := logoLine1.Render("  ╔═╗╔═╗╦ ╦╔═╗╔═╗╦ ╦╔═╗╔╦╗") + "\n" +
		logoLine2.Render("  ║ ╦╠═╝║ ║╚═╗║  ╠═╣║╣  ║║") + "\n" +
		logoLine3.Render("  ╚═╝╩  ╚═╝╚═╝╚═╝╩ ╩╚═╝═╩╝")
	info := dimStyle.Render(fmt.Sprintf("\n  %s · %s", gpuName, gpuMem))
	b.WriteString(logo + info + "\n\n")

	for _, g := range m.status.GPUs {
		label := fmt.Sprintf("GPU %d", g.Index)
		bar := renderBar(g.MemUsed, g.MemTotal, 30)
		info := fmt.Sprintf("%d / %d MB", g.MemUsed, g.MemTotal)
		b.WriteString(fmt.Sprintf("  %-6s %s  %s\n", label, bar, dimStyle.Render(info)))
	}

	mem := m.status.Memory
	if mem.HostRAMTotalMB > 0 {
		used := mem.HostRAMTotalMB - mem.HostRAMFreeMB
		bar := renderBar(used, mem.HostRAMTotalMB, 30)
		info := fmt.Sprintf("%d / %d MB", used, mem.HostRAMTotalMB)
		snapInfo := ""
		if mem.SnapshotsMB > 0 {
			snapInfo = fmt.Sprintf("  snapshots: %d MB", mem.SnapshotsMB)
		}
		b.WriteString(fmt.Sprintf("  %-6s %s  %s%s\n", "RAM", bar, dimStyle.Render(info), dimStyle.Render(snapInfo)))
	}

	if mem.DiskUsedMB > 0 || mem.DiskBudgetMB > 0 {
		bar := renderBar(mem.DiskUsedMB, mem.DiskBudgetMB, 30)
		info := fmt.Sprintf("%d / %d MB", mem.DiskUsedMB, mem.DiskBudgetMB)
		b.WriteString(fmt.Sprintf("  %-6s %s  %s\n", "Disk", bar, dimStyle.Render(info)))
	}
	b.WriteString("\n")

	b.WriteString(headerStyle.Render("  PROCESSES") + "\n\n")
	if len(m.status.Processes) == 0 {
		b.WriteString(dimStyle.Render("  (no processes — use 'gpusched run' to start one)") + "\n")
	} else {
		b.WriteString(dimStyle.Render("  NAME              STATE         MEM        AGE") + "\n")
		for i, p := range m.status.Processes {
			cursor := "  "
			if i == m.cursor {
				cursor = boldStyle.Render("▸ ")
			}

			icon, nameStyled := stateStyle(p.State, p.Name)
			state := stateLabel(p.State)
			mem := fmt.Sprintf("%d MB", p.MemMB)
			line := fmt.Sprintf("%s%-18s%-14s%-11s%s", cursor, icon+" "+nameStyled, state, mem, dimStyle.Render(p.Age))
			b.WriteString(line + "\n")
		}
	}
	b.WriteString("\n")

	met := m.status.Metrics
	b.WriteString(headerStyle.Render("  METRICS") + "\n\n")
	b.WriteString(fmt.Sprintf("  Requests: %s  Freezes: %s  Thaws: %s  Forks: %s  Migrations: %s\n",
		boldStyle.Render(fmt.Sprintf("%d", met.Requests)),
		boldStyle.Render(fmt.Sprintf("%d", met.Freezes)),
		boldStyle.Render(fmt.Sprintf("%d", met.Thaws)),
		boldStyle.Render(fmt.Sprintf("%d", met.Forks)),
		boldStyle.Render(fmt.Sprintf("%d", met.Migrations)),
	))
	if met.AvgFreezeMs > 0 || met.AvgThawMs > 0 {
		b.WriteString(fmt.Sprintf("  Avg freeze: %dms  Avg thaw: %dms  Cold starts: %d  Hibernations: %d\n",
			met.AvgFreezeMs, met.AvgThawMs, met.ColdStarts, met.Hibernations))
	}
	b.WriteString("\n")

	b.WriteString(headerStyle.Render("  EVENTS") + "\n\n")
	maxEvents := 8
	start := 0
	if len(m.events) > maxEvents {
		start = len(m.events) - maxEvents
	}
	if len(m.events) == 0 {
		b.WriteString(dimStyle.Render("  (no events yet)") + "\n")
	}
	for i := len(m.events) - 1; i >= start; i-- {
		e := m.events[i]
		ts := e.Time.Format("15:04:05")
		typ := eventTypeLabel(e.Type)
		dur := ""
		if e.Duration > 0 {
			dur = dimStyle.Render(fmt.Sprintf(" (%d ms)", e.Duration))
		}
		detail := ""
		if e.Detail != "" {
			detail = " " + e.Detail
		}
		proc := ""
		if e.Process != "" {
			proc = " " + boldStyle.Render(e.Process)
		}
		b.WriteString(fmt.Sprintf("  %s %s%s%s%s\n", dimStyle.Render(ts), typ, proc, detail, dur))
	}
	b.WriteString("\n")

	caps := m.status.Caps
	capStr := dimStyle.Render(fmt.Sprintf("  cuda-checkpoint: %s  criu: %s  driver: %s",
		boolStr(caps.CUDACheckpoint), boolStr(caps.CRIU), caps.DriverVersion))
	b.WriteString(capStr + "\n\n")

	if m.err != nil {
		b.WriteString(warnStyle.Render(fmt.Sprintf("  ERROR: %v", m.err)) + "\n\n")
	}

	b.WriteString(helpStyle.Render("  ↑↓:select  f:freeze  t:thaw  x:kill  q:quit"))
	b.WriteString("\n")

	return b.String()
}

func renderBar(used, total int64, width int) string {
	if total <= 0 {
		return strings.Repeat("░", width)
	}
	pct := float64(used) / float64(total)
	filled := int(pct * float64(width))
	if filled > width {
		filled = width
	}
	empty := width - filled

	style := barFull
	if pct > 0.9 {
		style = barCrit
	} else if pct > 0.7 {
		style = barWarn
	}

	return style.Render(strings.Repeat("█", filled)) +
		barEmpty.Render(strings.Repeat("░", empty))
}

func stateStyle(state protocol.ProcessState, name string) (string, string) {
	switch state {
	case protocol.StateActive:
		return activeStyle.Render("●"), activeStyle.Render(name)
	case protocol.StateFrozen:
		return frozenStyle.Render("○"), frozenStyle.Render(name)
	case protocol.StateHibernated:
		return hibStyle.Render("◌"), hibStyle.Render(name)
	default:
		return deadStyle.Render("✕"), deadStyle.Render(name)
	}
}

func stateLabel(state protocol.ProcessState) string {
	switch state {
	case protocol.StateActive:
		return activeStyle.Render("active")
	case protocol.StateFrozen:
		return frozenStyle.Render("frozen")
	case protocol.StateHibernated:
		return hibStyle.Render("hibernated")
	default:
		return deadStyle.Render("dead")
	}
}

func eventTypeLabel(typ string) string {
	switch typ {
	case "freeze":
		return frozenStyle.Render("FREEZE")
	case "thaw":
		return activeStyle.Render("THAW")
	case "run":
		return activeStyle.Render("RUN")
	case "kill":
		return deadStyle.Render("KILL")
	case "exit":
		return deadStyle.Render("EXIT")
	case "fork":
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#C678DD")).Render("FORK")
	case "migrate":
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#E5C07B")).Render("MIGRATE")
	case "hibernate":
		return hibStyle.Render("HIBERNATE")
	case "evict":
		return warnStyle.Render("EVICT")
	case "evict-kill":
		return deadStyle.Render("EVICT-KILL")
	default:
		return dimStyle.Render(strings.ToUpper(typ))
	}
}

func boolStr(b bool) string {
	if b {
		return activeStyle.Render("✓")
	}
	return deadStyle.Render("✗")
}

func Run(c *client.Client) error {
	p := tea.NewProgram(NewModel(c), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
