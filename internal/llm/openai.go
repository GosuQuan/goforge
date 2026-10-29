package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAIOptions configures an OpenAI-compatible chat client.
type OpenAIOptions struct {
	BaseURL     string // e.g. https://api.openai.com/v1 or https://api.deepseek.com
	APIKey      string
	Model       string
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration
	HTTPClient  *http.Client
}

// OpenAIClient talks to any /chat/completions endpoint that follows the OpenAI
// schema (OpenAI, DeepSeek, Moonshot, vLLM, LM Studio, Ollama, ...).
type OpenAIClient struct {
	opts OpenAIOptions
	hc   *http.Client
}

// NewOpenAIClient validates options and returns a client.
func NewOpenAIClient(opts OpenAIOptions) (*OpenAIClient, error) {
	if opts.BaseURL == "" {
		opts.BaseURL = "https://api.openai.com/v1"
	}
	opts.BaseURL = strings.TrimRight(opts.BaseURL, "/")
	if opts.Model == "" {
		return nil, fmt.Errorf("llm: model is required")
	}
	if opts.Timeout == 0 {
		opts.Timeout = 120 * time.Second
	}
	hc := opts.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: opts.Timeout}
	}
	return &OpenAIClient{opts: opts, hc: hc}, nil
}

// Model returns the configured model name.
func (c *OpenAIClient) Model() string { return c.opts.Model }

type wireMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type wireRequest struct {
	Model       string        `json:"model"`
	Messages    []wireMessage `json:"messages"`
	Tools       []wireTool    `json:"tools,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

type wireResponse struct {
	Choices []struct {
		Message struct {
			Role      string         `json:"role"`
			Content   string         `json:"content"`
			ToolCalls []wireToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func toWireMessages(msgs []Message) []wireMessage {
	out := make([]wireMessage, 0, len(msgs))
	for _, m := range msgs {
		wm := wireMessage{Role: string(m.Role), Content: m.Content, ToolCallID: m.ToolCallID, Name: m.Name}
		if len(m.Content) == 0 {
			wm.Content = nil
		}
		for _, tc := range m.ToolCalls {
			var wt wireToolCall
			wt.ID = tc.ID
			wt.Type = "function"
			wt.Function.Name = tc.Name
			wt.Function.Arguments = string(tc.Arguments)
			wm.ToolCalls = append(wm.ToolCalls, wt)
		}
		out = append(out, wm)
	}
	return out
}

// Chat implements Client.
func (c *OpenAIClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	body := wireRequest{
		Model:       c.opts.Model,
		Messages:    toWireMessages(req.Messages),
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}
	if body.Temperature == 0 {
		body.Temperature = c.opts.Temperature
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = c.opts.MaxTokens
	}
	for _, t := range req.Tools {
		var wt wireTool
		wt.Type = "function"
		wt.Function.Name = t.Name
		wt.Function.Description = t.Description
		wt.Function.Parameters = t.Parameters
		body.Tools = append(body.Tools, wt)
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.opts.BaseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.opts.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
	}

	resp, err := c.hc.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: http: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("llm: read body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("llm: status %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}

	var wr wireResponse
	if err := json.Unmarshal(data, &wr); err != nil {
		return nil, fmt.Errorf("llm: decode: %w", err)
	}
	if wr.Error != nil {
		return nil, fmt.Errorf("llm: provider error: %s", wr.Error.Message)
	}
	if len(wr.Choices) == 0 {
		return nil, ErrEmptyResponse
	}
	msg := Message{Role: RoleAssistant, Content: wr.Choices[0].Message.Content}
	for _, tc := range wr.Choices[0].Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	return &ChatResponse{Message: msg, Usage: wr.Usage}, nil
}
