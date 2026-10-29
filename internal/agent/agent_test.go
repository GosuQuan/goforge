package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mima/goforge/internal/llm"
	"github.com/mima/goforge/internal/tools"
)

type echoTool struct {
	calls int
}

func (e *echoTool) Name() string            { return "echo" }
func (e *echoTool) Description() string     { return "echo back" }
func (e *echoTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (e *echoTool) Execute(_ context.Context, args json.RawMessage) (string, error) {
	e.calls++
	return "echo:" + string(args), nil
}

func TestAgentReActLoop(t *testing.T) {
	tool := &echoTool{}
	reg := tools.NewRegistry().Register(tool)

	call1 := llm.Message{
		Role: llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{{
			ID: "c1", Name: "echo", Arguments: json.RawMessage(`{"v":1}`),
		}},
	}
	final := llm.Message{Role: llm.RoleAssistant, Content: "done"}
	mock := llm.NewMockClient(call1, final)

	a := New(mock, reg)
	res, err := a.Run(context.Background(), "please echo")
	if err != nil {
		t.Fatal(err)
	}
	if res.Final != "done" {
		t.Fatalf("final = %q", res.Final)
	}
	if tool.calls != 1 {
		t.Fatalf("tool calls = %d", tool.calls)
	}
	if mock.Calls() != 2 {
		t.Fatalf("llm calls = %d", mock.Calls())
	}
	// messages: system, user, assistant(tool), tool, assistant(final)
	if len(res.Messages) != 5 {
		t.Fatalf("messages = %d: %v", len(res.Messages), res.Messages)
	}
}

func TestAgentMaxSteps(t *testing.T) {
	reg := tools.NewRegistry()
	// A model that always asks for a tool call.
	loop := llm.Message{
		Role:      llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{{ID: "c", Name: "missing", Arguments: json.RawMessage(`{}`)}},
	}
	mock := llm.NewMockClient(loop, loop, loop)
	a := New(mock, reg)
	a.MaxSteps = 2
	res, err := a.Run(context.Background(), "spin")
	if err != nil {
		t.Fatal(err)
	}
	if res.Steps != 2 {
		t.Fatalf("steps = %d", res.Steps)
	}
}
