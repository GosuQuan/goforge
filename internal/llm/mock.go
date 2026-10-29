package llm

import (
	"context"
	"fmt"
	"sync"
)

// MockClient returns a program of pre-scripted responses. It lets the whole
// harness (graph + agent + tools + RAG) run with zero network and zero API key,
// which is what CI and the offline demo use.
//
// If the program runs out, it returns a plain assistant message explaining it.
type MockClient struct {
	mu        sync.Mutex
	responses []ChatResponse
	i         int
	// OnExhausted, when set, is called with the incoming request once the
	// script is empty, instead of returning the default message.
	OnExhausted func(req ChatRequest) ChatResponse
}

// NewMockClient builds a mock from assistant messages (tool calls included).
func NewMockClient(msgs ...Message) *MockClient {
	m := &MockClient{}
	for _, msg := range msgs {
		m.responses = append(m.responses, ChatResponse{Message: msg})
	}
	return m
}

// Enqueue appends more scripted responses.
func (m *MockClient) Enqueue(msgs ...Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, msg := range msgs {
		m.responses = append(m.responses, ChatResponse{Message: msg})
	}
}

// Calls reports how many times Chat was invoked (useful in tests).
func (m *MockClient) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.i
}

// Chat implements Client.
func (m *MockClient) Chat(_ context.Context, req ChatRequest) (*ChatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.i < len(m.responses) {
		resp := m.responses[m.i]
		m.i++
		return &resp, nil
	}
	if m.OnExhausted != nil {
		r := m.OnExhausted(req)
		m.i++
		return &r, nil
	}
	m.i++
	return &ChatResponse{Message: Message{
		Role:    RoleAssistant,
		Content: fmt.Sprintf("mock: no more scripted responses (request had %d messages)", len(req.Messages)),
	}}, nil
}
