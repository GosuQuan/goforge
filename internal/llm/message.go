// Package llm defines a minimal, provider-agnostic chat abstraction used by the
// agent runtime. It ships with an OpenAI-compatible client (works with OpenAI,
// DeepSeek, Moonshot, vLLM, Ollama's OpenAI endpoint, ...) and a deterministic
// mock client so the whole harness can run fully offline in CI and demos.
package llm

import (
	"encoding/json"
	"strings"
)

// Role is the author of a chat message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a model request to invoke a tool. Arguments is raw JSON so that it
// can be validated by the tool before decoding.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Message is a single entry in a conversation.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// System builds a system message.
func System(content string) Message { return Message{Role: RoleSystem, Content: content} }

// User builds a user message.
func User(content string) Message { return Message{Role: RoleUser, Content: content} }

// Assistant builds an assistant message.
func Assistant(content string) Message { return Message{Role: RoleAssistant, Content: content} }

// ToolResult builds a tool observation message tied to a call ID.
func ToolResult(callID, name, content string) Message {
	return Message{Role: RoleTool, ToolCallID: callID, Name: name, Content: content}
}

// Text renders the message for logging.
func (m Message) Text() string {
	var b strings.Builder
	b.WriteString(string(m.Role))
	b.WriteString(": ")
	b.WriteString(m.Content)
	for _, tc := range m.ToolCalls {
		b.WriteString("\n  -> ")
		b.WriteString(tc.Name)
		b.WriteString("(")
		b.Write(tc.Arguments)
		b.WriteString(")")
	}
	return b.String()
}
