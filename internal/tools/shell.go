package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// Shell runs a shell command inside the workspace root. It is the sharpest tool
// in the box, so it is intentionally constrained: commands run in the workspace,
// have a hard timeout, and output is truncated.
type Shell struct{ opts Options }

func (t *Shell) Name() string { return "run_shell" }
func (t *Shell) Description() string {
	return "Run a shell command inside the workspace root and return combined stdout/stderr. Use for building, running tests, git inspection. Commands are subject to a timeout."
}
func (t *Shell) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"},"timeout_seconds":{"type":"integer","description":"max seconds, default 60"}},"required":["command"]}`)
}

func (t *Shell) Execute(ctx context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Command        string `json:"command"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	if strings.TrimSpace(a.Command) == "" {
		return "", fmt.Errorf("run_shell: command is required")
	}
	timeout := t.opts.ShellTimeout
	if a.TimeoutSeconds > 0 {
		timeout = time.Duration(a.TimeoutSeconds) * time.Second
	}
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cctx, "bash", "-lc", a.Command)
	cmd.Dir = t.opts.root()
	out, err := cmd.CombinedOutput()
	result := string(out)
	if cctx.Err() == context.DeadlineExceeded {
		result += fmt.Sprintf("\n[command timed out after %s]", timeout)
	} else if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			result += fmt.Sprintf("\n[exit status %d]", ee.ExitCode())
		} else {
			result += "\n[error: " + err.Error() + "]"
		}
	}
	if strings.TrimSpace(result) == "" {
		result = "[no output]"
	}
	return truncate(result, t.opts.maxOut()), nil
}
