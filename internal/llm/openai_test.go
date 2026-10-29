package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAIClientParsesToolCalls(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing auth header")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
				{"id":"c1","type":"function","function":{"name":"echo","arguments":"{\"x\":1}"}}
			]}}],
			"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
		}`))
	}))
	defer srv.Close()

	c, err := NewOpenAIClient(OpenAIOptions{BaseURL: srv.URL, APIKey: "test-key", Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), ChatRequest{
		Messages: []Message{User("hi")},
		Tools:    []ToolSpec{{Name: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/chat/completions" {
		t.Fatalf("path = %q", gotPath)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.Name != "echo" || string(tc.Arguments) != `{"x":1}` {
		t.Fatalf("bad tool call: %+v", tc)
	}
	if resp.Usage.TotalTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestOpenAIClientHandlesError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer srv.Close()
	c, _ := NewOpenAIClient(OpenAIOptions{BaseURL: srv.URL, Model: "m"})
	if _, err := c.Chat(context.Background(), ChatRequest{Messages: []Message{User("x")}}); err == nil {
		t.Fatal("expected error")
	}
}

func TestMockClientExhaustion(t *testing.T) {
	m := NewMockClient(Message{Role: RoleAssistant, Content: "one"})
	if _, err := m.Chat(context.Background(), ChatRequest{}); err != nil {
		t.Fatal(err)
	}
	resp, err := m.Chat(context.Background(), ChatRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Role != RoleAssistant {
		t.Fatalf("role = %s", resp.Message.Role)
	}
}
