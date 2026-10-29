package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/mima/goforge/internal/rag"
)

// CodeSearch exposes the RAG retriever as a tool: "search the codebase by
// meaning/lexical similarity". This is what turns the harness from a plain
// file-reader into a retrieval-augmented coding agent.
type CodeSearch struct {
	opts      Options
	retriever *rag.Retriever
}

func (t *CodeSearch) Name() string { return "code_search" }
func (t *CodeSearch) Description() string {
	return "Semantic/lexical search over the indexed codebase. Returns the most relevant code chunks with file paths and line ranges. Use it first to locate where something lives before reading whole files."
}
func (t *CodeSearch) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"natural language or identifiers to search for"},"top_k":{"type":"integer","description":"number of chunks, default 6"}},"required":["query"]}`)
}

func (t *CodeSearch) Execute(ctx context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	results, err := t.retriever.Retrieve(ctx, a.Query, a.TopK)
	if err != nil {
		return "", fmt.Errorf("code_search: %w", err)
	}
	if len(results) == 0 {
		return fmt.Sprintf("no code found for %q (is the index built? run `goforge index`)", a.Query), nil
	}
	return truncate(rag.FormatResults(results, 4000), t.opts.maxOut()), nil
}
