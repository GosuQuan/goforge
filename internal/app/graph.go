package app

import (
	"context"
	"fmt"
	"strings"

	"github.com/mima/goforge/internal/agent"
	"github.com/mima/goforge/internal/config"
	"github.com/mima/goforge/internal/graph"
	"github.com/mima/goforge/internal/llm"
	"github.com/mima/goforge/internal/rag"
	"github.com/mima/goforge/internal/tools"
)

// State keys used by the coding graph. Exported-ish via helpers below.
const (
	keyTask     = "task"
	keyMessages = "messages"
	keyContext  = "context"
	keyFinal    = "final"
	keyApproved = "approved"
)

// Node names (also used by the mermaid diagram and the CLI event log).
const (
	NodeRetrieve = "retrieve"
	NodeAgent    = "agent"
	NodeGuard    = "guard"
	NodeTools    = "tools"
	NodeFinalize = "finalize"
)

func messagesOf(state graph.State) []llm.Message {
	if v, ok := state[keyMessages]; ok {
		if m, ok := v.([]llm.Message); ok {
			return m
		}
	}
	return nil
}

func lastMessage(msgs []llm.Message) (llm.Message, bool) {
	if len(msgs) == 0 {
		return llm.Message{}, false
	}
	return msgs[len(msgs)-1], true
}

// buildCodingGraph assembles the LangGraph-style control flow:
//
//	retrieve ──▶ agent ──(tool calls?)──▶ guard ──▶ tools ──┐
//	               │                                        │
//	               └──(no tool calls)──▶ finalize ◀─────────┘ (loop back to agent)
//
// The loop is the ReAct cycle; `guard` is the human-in-the-loop gate.
func buildCodingGraph(client llm.Client, reg *tools.Registry, retriever *rag.Retriever, cp graph.Checkpointer, cfg config.Config) (*graph.CompiledGraph, error) {
	g := graph.New().Named("goforge-coding-agent")
	g.SetReducer(keyMessages, graph.AppendReducer)
	g.WithCheckpointer(cp)
	g.SetMaxSteps(cfg.MaxSteps*2 + 10)

	// retrieve: RAG lookup for the task, injected as context.
	g.AddNode(NodeRetrieve, func(ctx context.Context, st graph.State) (graph.State, error) {
		task := st.GetString(keyTask)
		if retriever == nil || strings.TrimSpace(task) == "" {
			return graph.State{keyContext: ""}, nil
		}
		results, err := retriever.Retrieve(ctx, task, 6)
		if err != nil {
			// Retrieval is best-effort: never fail the run on a bad query.
			return graph.State{keyContext: "(retrieval unavailable: " + err.Error() + ")"}, nil
		}
		context := rag.FormatResults(results, 6000)
		if context == "" {
			context = "(no indexed code matched the task; use grep/glob/read_file to explore)"
		}
		return graph.State{keyContext: context}, nil
	})

	// agent: one LLM turn. Seeds the conversation on the first visit.
	g.AddNode(NodeAgent, func(ctx context.Context, st graph.State) (graph.State, error) {
		msgs := messagesOf(st)
		var update []llm.Message
		if len(msgs) == 0 {
			system := llm.System(agent.DefaultSystemPrompt)
			user := llm.User(buildUserPrompt(st.GetString(keyTask), st.GetString(keyContext)))
			msgs = []llm.Message{system, user}
			update = append(update, system, user)
		}
		resp, err := client.Chat(ctx, llm.ChatRequest{
			Messages:    trimMessages(msgs, 40),
			Tools:       reg.Specs(),
			Temperature: cfg.Temperature,
			MaxTokens:   cfg.MaxTokens,
		})
		if err != nil {
			return nil, fmt.Errorf("agent turn failed: %w", err)
		}
		assistant := resp.Message
		assistant.Role = llm.RoleAssistant
		update = append(update, assistant)
		return graph.State{keyMessages: update}, nil
	})

	// guard: human-in-the-loop gate before mutating tools execute.
	g.AddNode(NodeGuard, func(_ context.Context, st graph.State) (graph.State, error) {
		if !cfg.RequireApproval {
			return nil, nil
		}
		if st.GetBool(keyApproved) {
			return nil, nil
		}
		msgs := messagesOf(st)
		last, ok := lastMessage(msgs)
		if !ok {
			return nil, nil
		}
		var pending []string
		for _, c := range last.ToolCalls {
			if isMutating(c.Name) {
				pending = append(pending, c.Name+" "+truncateRunes(string(c.Arguments), 160))
			}
		}
		if len(pending) == 0 {
			return nil, nil
		}
		return nil, &graph.Interrupt{
			Prompt: "approve mutating tool calls before continuing",
			Data:   map[string]any{"tools": pending},
		}
	})

	// tools: execute the calls the model asked for, append observations.
	g.AddNode(NodeTools, func(ctx context.Context, st graph.State) (graph.State, error) {
		msgs := messagesOf(st)
		last, ok := lastMessage(msgs)
		if !ok || len(last.ToolCalls) == 0 {
			return nil, nil
		}
		var results []llm.Message
		for _, call := range last.ToolCalls {
			obs, err := reg.Execute(ctx, call)
			if err != nil {
				obs = "ERROR: " + err.Error()
			}
			results = append(results, llm.ToolResult(call.ID, call.Name, obs))
		}
		return graph.State{keyMessages: results}, nil
	})

	// finalize: pull the model's final answer out of the transcript.
	g.AddNode(NodeFinalize, func(_ context.Context, st graph.State) (graph.State, error) {
		msgs := messagesOf(st)
		final := ""
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == llm.RoleAssistant && strings.TrimSpace(msgs[i].Content) != "" {
				final = msgs[i].Content
				break
			}
		}
		if final == "" {
			final = "(agent stopped without a final message)"
		}
		return graph.State{keyFinal: final}, nil
	})

	g.SetEntry(NodeRetrieve)
	g.AddEdge(NodeRetrieve, NodeAgent)
	g.AddConditionalEdge(NodeAgent, func(_ context.Context, st graph.State) (string, error) {
		last, ok := lastMessage(messagesOf(st))
		if ok && last.Role == llm.RoleAssistant && len(last.ToolCalls) > 0 {
			return NodeGuard, nil
		}
		return NodeFinalize, nil
	})
	g.AddEdge(NodeGuard, NodeTools)
	g.AddEdge(NodeTools, NodeAgent)
	g.AddEdge(NodeFinalize, graph.END)

	return g.Compile()
}

func buildUserPrompt(task, context string) string {
	var b strings.Builder
	b.WriteString("Task:\n")
	b.WriteString(strings.TrimSpace(task))
	b.WriteString("\n\n")
	if strings.TrimSpace(context) != "" {
		b.WriteString("Relevant code retrieved from the repository (use these as starting points, verify by reading files):\n")
		b.WriteString(context)
		b.WriteString("\n\n")
	}
	b.WriteString("Work step by step using the tools. When finished, give a concise final answer (no tool calls).")
	return b.String()
}

func isMutating(name string) bool {
	switch name {
	case "write_file", "edit_file", "run_shell":
		return true
	default:
		return false
	}
}

func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}

// trimMessages keeps the system message plus the most recent window, so long
// tool loops do not blow past the context window. This is a simple stand-in for
// LangGraph/LangChain "message trimming" or a condenser.
func trimMessages(msgs []llm.Message, max int) []llm.Message {
	if max <= 0 || len(msgs) <= max {
		return msgs
	}
	out := make([]llm.Message, 0, max)
	// Preserve the first system message.
	if msgs[0].Role == llm.RoleSystem {
		out = append(out, msgs[0])
	}
	tail := msgs[len(msgs)-(max-1):]
	out = append(out, tail...)
	return out
}
