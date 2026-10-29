package rag

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// Retriever couples an Embedder with a VectorStore to serve queries.
type Retriever struct {
	Embedder   Embedder
	Store      VectorStore
	DefaultTop int
}

// NewRetriever builds a retriever (default top-k 6).
func NewRetriever(e Embedder, s VectorStore) *Retriever {
	return &Retriever{Embedder: e, Store: s, DefaultTop: 6}
}

// Retrieve embeds the query and returns the top-k most similar chunks.
func (r *Retriever) Retrieve(ctx context.Context, query string, topK int) ([]Result, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("rag: empty query")
	}
	if topK <= 0 {
		topK = r.DefaultTop
	}
	vecs, err := r.Embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("rag: embedder returned no vector")
	}
	// Over-fetch, then re-rank with a small language prior: when searching a
	// codebase, a matching Go file is usually more useful than a matching
	// README. This is a cheap, explainable re-ranker; a real deployment would
	// use a cross-encoder or a provider rerank API.
	fetch := topK * 4
	if fetch < 20 {
		fetch = 20
	}
	results, err := r.Store.Query(ctx, vecs[0], fetch)
	if err != nil {
		return nil, err
	}
	codeIntent := hasCodeIntent(query)
	for i := range results {
		w := languageWeight(results[i].Chunk.Language)
		// If the query clearly asks about code, demote prose/docs further so a
		// README that merely mentions the identifier does not outrank the file
		// that actually defines it.
		if codeIntent && (results[i].Chunk.Language == "markdown" || results[i].Chunk.Language == "text") {
			w *= 0.30
		}
		results[i].Score *= w
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func languageWeight(lang string) float32 {
	switch lang {
	case "go", "python", "javascript", "typescript", "java", "rust", "ruby", "c", "cpp":
		return 1.08
	case "markdown":
		return 0.80
	case "yaml", "json", "sql", "shell":
		return 0.90
	default:
		return 0.95
	}
}

// hasCodeIntent detects queries that are about code rather than docs. It is a
// tiny, transparent heuristic — good enough to stop a README from outranking the
// file that defines a function, without pretending to be semantic search.
func hasCodeIntent(query string) bool {
	words := []string{
		"function", "func", "method", "class", "struct", "interface",
		"defined", "define", "implement", "implementation", "return", "returns",
		"variable", "const", "package", "import", "bug", "error", "compile",
		"test", "api", "endpoint", "handler", "type", "signature", "call",
	}
	q := strings.ToLower(query)
	for _, w := range words {
		if strings.Contains(q, w) {
			return true
		}
	}
	return false
}

// FormatResults renders results as an LLM-friendly context block with stable
// `path:start-end` citations the model can reference.
func FormatResults(results []Result, maxChars int) string {
	var b strings.Builder
	budget := maxChars
	for i, r := range results {
		header := fmt.Sprintf("[%d] %s (lines %d-%d, score %.3f)\n", i+1, r.Chunk.Path, r.Chunk.StartLine, r.Chunk.EndLine, r.Score)
		body := r.Chunk.Content
		if maxChars > 0 {
			remaining := budget - len(header)
			if remaining <= 0 {
				break
			}
			if len(body) > remaining {
				body = body[:remaining] + "\n... (truncated)"
			}
			budget -= len(header) + len(body)
		}
		b.WriteString(header)
		b.WriteString("```")
		b.WriteString(strings.TrimSpace(languageFence(r.Chunk.Language)))
		b.WriteString("\n")
		b.WriteString(strings.TrimRight(body, "\n"))
		b.WriteString("\n```\n\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func languageFence(lang string) string {
	switch lang {
	case "go", "python", "javascript", "typescript", "java", "rust", "ruby", "c", "cpp", "sql", "shell", "yaml", "json", "markdown":
		return lang
	default:
		return ""
	}
}
