package app

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mima/goforge/internal/config"
	"github.com/mima/goforge/internal/llm"
)

func testConfig(t *testing.T, workdir string) config.Config {
	t.Helper()
	cfg := config.Default()
	cfg.WorkDir = workdir
	cfg.IndexFile = filepath.Join(t.TempDir(), "index.json")
	cfg.CheckpointDir = t.TempDir()
	cfg.VectorStore = "memory"
	cfg.MaxSteps = 12
	return cfg
}

func TestHarnessOfflineDemo(t *testing.T) {
	workdir, err := filepath.Abs("../../testdata/sample-repo")
	if err != nil {
		t.Fatal(err)
	}
	cfg := testConfig(t, workdir)
	h, err := New(cfg, llm.NewDemoCodingClient())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.Index(context.Background(), workdir, nil); err != nil {
		t.Fatal(err)
	}
	res, err := h.Run(context.Background(), RunOptions{Task: "Where is the Add function defined and what does it do?"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Interrupted {
		t.Fatal("unexpected interrupt")
	}
	if !strings.Contains(res.Final, "calc.go") && !strings.Contains(res.Final, "Add") {
		t.Fatalf("final answer does not mention target: %s", res.Final)
	}
	if len(res.Messages) < 5 {
		t.Fatalf("expected a multi-step trajectory, got %d messages", len(res.Messages))
	}
	// Must have at least one tool call and one tool observation.
	var sawTool bool
	for _, m := range res.Messages {
		if m.Role == llm.RoleTool {
			sawTool = true
		}
	}
	if !sawTool {
		t.Fatal("no tool observations in trajectory")
	}
}

func TestHumanInTheLoopApproval(t *testing.T) {
	workdir := t.TempDir()
	cfg := testConfig(t, workdir)
	cfg.RequireApproval = true

	args, _ := json.Marshal(map[string]any{"path": "out.txt", "content": "hello from agent"})
	callMsg := llm.Message{
		Role:      llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{{ID: "w1", Name: "write_file", Arguments: args}},
	}
	finalMsg := llm.Message{Role: llm.RoleAssistant, Content: "wrote the file"}
	client := llm.NewMockClient(callMsg, finalMsg)

	h, err := New(cfg, client)
	if err != nil {
		t.Fatal(err)
	}
	runID := "hitl-run"
	res, err := h.Run(context.Background(), RunOptions{Task: "create out.txt", RunID: runID})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Interrupted {
		t.Fatal("expected interrupt before mutating tool")
	}
	if _, statErr := os.Stat(filepath.Join(workdir, "out.txt")); statErr == nil {
		t.Fatal("file should not exist before approval")
	}

	res, err = h.Run(context.Background(), RunOptions{Task: "create out.txt", RunID: runID, Resume: true, Approve: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Interrupted {
		t.Fatal("still interrupted after approval")
	}
	if res.Final != "wrote the file" {
		t.Fatalf("final = %q", res.Final)
	}
	data, err := os.ReadFile(filepath.Join(workdir, "out.txt"))
	if err != nil {
		t.Fatalf("expected file after approval: %v", err)
	}
	if string(data) != "hello from agent" {
		t.Fatalf("content = %q", string(data))
	}
}

func TestHarnessMutatingTools(t *testing.T) {
	workdir := t.TempDir()
	cfg := testConfig(t, workdir)

	writeArgs, _ := json.Marshal(map[string]any{"path": "out.txt", "content": "hello from agent"})
	shellArgs, _ := json.Marshal(map[string]any{"command": "cat out.txt"})
	client := llm.NewMockClient(
		llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "w1", Name: "write_file", Arguments: writeArgs}}},
		llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "s1", Name: "run_shell", Arguments: shellArgs}}},
		llm.Message{Role: llm.RoleAssistant, Content: "done: wrote and verified"},
	)
	h, err := New(cfg, client)
	if err != nil {
		t.Fatal(err)
	}
	res, err := h.Run(context.Background(), RunOptions{Task: "write out.txt then cat it"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Final != "done: wrote and verified" {
		t.Fatalf("final = %q", res.Final)
	}
	data, err := os.ReadFile(filepath.Join(workdir, "out.txt"))
	if err != nil || string(data) != "hello from agent" {
		t.Fatalf("file = %q err=%v", string(data), err)
	}
	// The shell observation must contain the file content (proves round-trip).
	var sawCat bool
	for _, m := range res.Messages {
		if m.Role == llm.RoleTool && strings.Contains(m.Content, "hello from agent") {
			sawCat = true
		}
	}
	if !sawCat {
		t.Fatal("shell tool did not observe file content")
	}
}

func TestDiagram(t *testing.T) {
	cfg := testConfig(t, t.TempDir())
	h, err := New(cfg, llm.NewMockClient())
	if err != nil {
		t.Fatal(err)
	}
	m := h.Graph.Mermaid()
	for _, want := range []string{"retrieve", "agent", "tools", "finalize"} {
		if !strings.Contains(m, want) {
			t.Fatalf("diagram missing %s:\n%s", want, m)
		}
	}
}
