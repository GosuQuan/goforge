package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// DemoCodingClient is a deterministic, offline stand-in for a real model. It
// drives a realistic multi-step trajectory so `goforge run --mock` exercises the
// full stack — graph routing, RAG retrieval, tool execution, and the ReAct
// cycle — with no network and no API key.
//
// Trajectory:
//
//	turn 1 -> tool call: code_search(query = task)
//	turn 2 -> tool call: read_file(top hit from search)
//	turn 3 -> final answer summarising the observed file
type DemoCodingClient struct{}

// NewDemoCodingClient returns the offline demo client.
func NewDemoCodingClient() *DemoCodingClient { return &DemoCodingClient{} }

var searchHitRE = regexp.MustCompile(`\[1\] (\S+) \(lines (\d+)-(\d+)`)

// Chat implements Client.
func (d *DemoCodingClient) Chat(_ context.Context, req ChatRequest) (*ChatResponse, error) {
	assistants := 0
	for _, m := range req.Messages {
		if m.Role == RoleAssistant {
			assistants++
		}
	}
	task := extractTask(req.Messages)

	switch assistants {
	case 0:
		args, _ := json.Marshal(map[string]any{"query": task, "top_k": 6})
		return &ChatResponse{Message: Message{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{{
				ID: "demo_call_search", Name: "code_search", Arguments: args,
			}},
		}}, nil
	case 1:
		obs := lastToolObservation(req.Messages)
		m := searchHitRE.FindStringSubmatch(obs)
		if m == nil {
			return &ChatResponse{Message: Message{
				Role:    RoleAssistant,
				Content: fmt.Sprintf("Offline demo: no indexed code matched %q. Build an index with `goforge index` and retry.", task),
			}}, nil
		}
		path := m[1]
		start := m[2]
		args, _ := json.Marshal(map[string]any{"path": path, "start_line": atoiSafe(start), "end_line": atoiSafe(start) + 40})
		return &ChatResponse{Message: Message{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{{
				ID: "demo_call_read", Name: "read_file", Arguments: args,
			}},
		}}, nil
	default:
		obs := lastToolObservation(req.Messages)
		return &ChatResponse{Message: Message{
			Role: RoleAssistant,
			Content: "Offline demo answer (deterministic client, no model API called).\n\n" +
				snippet(obs, 900) +
				"\n\nThis trajectory ran: code_search -> read_file -> finalize through the graph.",
		}}, nil
	}
}

func extractTask(msgs []Message) string {
	for _, m := range msgs {
		if m.Role != RoleUser {
			continue
		}
		c := m.Content
		if i := strings.Index(c, "Task:\n"); i >= 0 {
			c = c[i+len("Task:\n"):]
		}
		if j := strings.Index(c, "\n\n"); j >= 0 {
			c = c[:j]
		}
		c = strings.TrimSpace(c)
		if c != "" {
			return c
		}
	}
	return "explore the project"
}

func lastToolObservation(msgs []Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == RoleTool {
			return msgs[i].Content
		}
	}
	return ""
}

func snippet(s string, n int) string {
	s = strings.TrimSpace(s)
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "\n... (truncated)"
}

func atoiSafe(s string) int {
	n := 0
	for _, r := range s {
		if r < '0' || r > '9' {
			break
		}
		n = n*10 + int(r-'0')
	}
	return n
}
