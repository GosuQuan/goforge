// Package agent contains a plain ReAct agent loop. It is intentionally small:
// in Go the runtime (goroutines, context cancellation, typed structs) already
// gives us what a Python framework's runtime provides, so the loop itself is
// just "call model -> run tools -> feed results back -> repeat".
//
// The graph-based variant used by the CLI lives in internal/app; this package is
// the reusable primitive and is also what tests exercise in isolation.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mima/goforge/internal/llm"
	"github.com/mima/goforge/internal/tools"
)

// Agent runs a ReAct loop against a chat client and a tool registry.
type Agent struct {
	LLM         llm.Client
	Tools       *tools.Registry
	System      string
	MaxSteps    int
	Temperature float64
	MaxTokens   int
	// OnMessage is called for every assistant/tool message as it is produced.
	OnMessage func(llm.Message)
}

// New builds an agent with sane defaults.
func New(client llm.Client, reg *tools.Registry) *Agent {
	return &Agent{
		LLM:      client,
		Tools:    reg,
		System:   DefaultSystemPrompt,
		MaxSteps: 12,
	}
}

// Result is the outcome of a single Run.
type Result struct {
	Final    string
	Messages []llm.Message
	Steps    int
}

// Run executes the agent until the model stops calling tools or MaxSteps is hit.
func (a *Agent) Run(ctx context.Context, input string) (Result, error) {
	if a.LLM == nil {
		return Result{}, fmt.Errorf("agent: nil llm client")
	}
	system := a.System
	if system == "" {
		system = DefaultSystemPrompt
	}
	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 12
	}

	msgs := []llm.Message{llm.System(system), llm.User(input)}
	var specs []llm.ToolSpec
	if a.Tools != nil {
		specs = a.Tools.Specs()
	}

	for step := 0; step < maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return Result{Messages: msgs, Steps: step}, err
		}
		resp, err := a.LLM.Chat(ctx, llm.ChatRequest{
			Messages:    msgs,
			Tools:       specs,
			Temperature: a.Temperature,
			MaxTokens:   a.MaxTokens,
		})
		if err != nil {
			return Result{Messages: msgs, Steps: step}, err
		}
		assistant := resp.Message
		assistant.Role = llm.RoleAssistant
		msgs = append(msgs, assistant)
		if a.OnMessage != nil {
			a.OnMessage(assistant)
		}
		if len(assistant.ToolCalls) == 0 {
			return Result{Final: assistant.Content, Messages: msgs, Steps: step + 1}, nil
		}
		for _, call := range assistant.ToolCalls {
			obs, terr := a.execTool(ctx, call)
			toolMsg := llm.ToolResult(call.ID, call.Name, obs)
			if terr != nil {
				toolMsg.Content = "ERROR: " + terr.Error()
			}
			msgs = append(msgs, toolMsg)
			if a.OnMessage != nil {
				a.OnMessage(toolMsg)
			}
		}
	}
	last := lastAssistant(msgs)
	return Result{Final: last, Messages: msgs, Steps: maxSteps}, nil
}

func (a *Agent) execTool(ctx context.Context, call llm.ToolCall) (string, error) {
	if a.Tools == nil {
		return "", fmt.Errorf("no tools registered")
	}
	return a.Tools.Execute(ctx, call)
}

func lastAssistant(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == llm.RoleAssistant && strings.TrimSpace(msgs[i].Content) != "" {
			return msgs[i].Content
		}
	}
	return ""
}

// ToolCall is a small convenience for tests and scripts.
func ToolCall(id, name string, args map[string]any) llm.ToolCall {
	b, _ := json.Marshal(args)
	return llm.ToolCall{ID: id, Name: name, Arguments: b}
}
