package llm

import (
	"context"
	"encoding/json"
	"errors"
)

// ToolSpec describes a tool to the model using JSON Schema for its arguments.
type ToolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ChatRequest is a single completion request.
type ChatRequest struct {
	Messages    []Message
	Tools       []ToolSpec
	Temperature float64
	MaxTokens   int
}

// Usage reports token accounting when the provider returns it.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatResponse is a single completion result.
type ChatResponse struct {
	Message Message
	Usage   Usage
}

// Client is the single method the agent runtime depends on.
type Client interface {
	Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)
}

// ErrEmptyResponse is returned when a provider sends no usable choice.
var ErrEmptyResponse = errors.New("llm: provider returned no message")
